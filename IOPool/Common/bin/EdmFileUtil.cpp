//----------------------------------------------------------------------
// EdmFileUtil.cpp
//

#include <algorithm>
#include <unistd.h>
#include <exception>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <boost/program_options.hpp>
#include "IOPool/Common/bin/CollUtil.h"
#include "DataFormats/Provenance/interface/BranchType.h"
#include "FWCore/Catalog/interface/InputFileCatalog.h"
#include "FWCore/Catalog/interface/SiteLocalConfig.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/RootAutoLibraryLoader/interface/RootAutoLibraryLoader.h"
#include "FWCore/Services/src/SiteLocalConfigService.h"
#include "FWCore/Utilities/interface/Adler32Calculator.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"

#include "TFile.h"
#include "TError.h"

int main(int argc, char* argv[]) {

  gErrorIgnoreLevel = kError;

  // Add options here

  boost::program_options::options_description desc("Allowed options");
  desc.add_options()
    ("help,h", "print help message")
    ("file,f", boost::program_options::value<std::vector<std::string> >(), "data file (-f or -F required)")
    ("Files,F", boost::program_options::value<std::string>(), "text file containing names of data files, one per line")
    ("catalog,c", boost::program_options::value<std::string>(), "catalog")
    ("decodeLFN,d", "Convert LFN to PFN")
    ("uuid,u", "Print uuid")
    ("adler32,a", "Print adler32 checksum.")
    ("allowRecovery", "Allow root to auto-recover corrupted files")
    ("JSON,j", "JSON output format.  Any arguments listed below are ignored")
    ("ls,l", "list file content")
    ("print,P", "Print all")
    ("verbose,v", "Verbose printout")
    ("printBranchDetails,b", "Call Print()sc for all branches")
    ("tree,t", boost::program_options::value<std::string>(), "Select tree used with -P and -b options")
    ("events,e", "Print list of all Events, Runs, and LuminosityBlocks in the file sorted by run number, luminosity block number, and event number.  Also prints the entry numbers and whether it is possible to use fast copy with the file.");

  // What trees do we require for this to be a valid collection?
  std::vector<std::string> expectedTrees;
  expectedTrees.push_back(edm::poolNames::metaDataTreeName());
  expectedTrees.push_back(edm::poolNames::eventTreeName());

  boost::program_options::positional_options_description p;
  p.add("file", -1);

  boost::program_options::variables_map vm;


  try {
      boost::program_options::store(boost::program_options::command_line_parser(argc, argv).
                                    options(desc).positional(p).run(), vm);
  } catch (boost::program_options::error const& x) {
      std::cerr << "Option parsing failure:\n"
                << x.what() << "\n\n";
      std::cerr << desc << "\n";
      return 1;
  }

  boost::program_options::notify(vm);

  if (vm.count("help")) {
    std::cout << desc << "\n";
    return 1;
  }

  int rc = 0;
  try {
    std::auto_ptr<edm::SiteLocalConfig> slcptr(new edm::service::SiteLocalConfigService(edm::ParameterSet()));
    boost::shared_ptr<edm::serviceregistry::ServiceWrapper<edm::SiteLocalConfig> > slc(new edm::serviceregistry::ServiceWrapper<edm::SiteLocalConfig>(slcptr));
    edm::ServiceToken slcToken = edm::ServiceRegistry::createContaining(slc);
    edm::ServiceRegistry::Operate operate(slcToken);

    std::vector<std::string> in = (vm.count("file") ? vm["file"].as<std::vector<std::string> >() : std::vector<std::string>());
    if (vm.count("Files")) {
      std::ifstream ifile(vm["Files"].as<std::string>().c_str());
      std::istream_iterator<std::string> beginItr(ifile);
      if (ifile.fail()) {
        std::cout << "File '" << vm["Files"].as<std::string>() << "' not found, not opened, or empty\n";
        return 1;
      }
      std::istream_iterator<std::string> endItr;
      copy(beginItr, endItr, back_inserter(in));
    }
    if (in.empty()) {
      std::cout << "Data file(s) not set.\n";
      std::cout << desc << "\n";
      return 1;
    }
    std::string catalogIn = (vm.count("catalog") ? vm["catalog"].as<std::string>() : std::string());
    bool decodeLFN = vm.count("decodeLFN");
    bool uuid = vm.count("uuid");
    bool adler32 = vm.count("adler32");
    bool allowRecovery = vm.count("allowRecovery");
    bool json = vm.count("JSON");
    bool more = !json;
    bool verbose = more && (vm.count("verbose") > 0 ? true : false);
    bool events = more && (vm.count("events") > 0 ? true : false);
    bool ls = more && (vm.count("ls") > 0 ? true : false);
    bool tree = more && (vm.count("tree") > 0 ? true : false);
    bool print = more && (vm.count("print") > 0 ? true : false);
    bool printBranchDetails = more && (vm.count("printBranchDetails") > 0 ? true : false);
    bool onlyDecodeLFN = decodeLFN && !(uuid || adler32 || allowRecovery || json || events || tree || ls || print || printBranchDetails);
    std::string selectedTree = tree ? vm["tree"].as<std::string>() : edm::poolNames::eventTreeName().c_str();

    if (events) {
      try {
        edmplugin::PluginManager::configure(edmplugin::standard::config());
      } catch(std::exception& e) {
        std::cout << "exception caught in EdmFileUtil while configuring the PluginManager\n" << e.what();
        return 1;
      }
      edm::RootAutoLibraryLoader::enable();
    }

    edm::InputFileCatalog catalog(in, catalogIn, true);
    std::vector<std::string> const& filesIn = catalog.fileNames();

    if (json) {
      std::cout << '[' << std::endl;
    }

    // now run..
    // Allow user to input multiple files
    for(unsigned int j = 0; j < in.size(); ++j) {

      // We _only_ want the LFN->PFN conversion. No need to open the file,
      // just check the catalog and move on
      if (onlyDecodeLFN) {
        std::cout << filesIn[j] << std::endl;
        continue;
      }

      // open a data file
      if (!json) std::cout << in[j] << "\n";
      std::string const& lfn = in[j];
      TFile *tfile = edm::openFileHdl(filesIn[j]);
      if (tfile == 0) return 1;

      std::string const& pfn = filesIn[j];

      if (verbose) std::cout << "ECU:: Opened " << pfn << std::endl;

      std::string datafile = decodeLFN ? pfn : lfn;

      // First check that this file is not auto-recovered
      // Stop the job unless specified to do otherwise

      bool isRecovered = tfile->TestBit(TFile::kRecovered);
      if (isRecovered) {
        if (allowRecovery) {
          if (!json) {
            std::cout << pfn << " appears not to have been closed correctly and has been autorecovered \n";
            std::cout << "Proceeding anyway\n";
          }
        } else {
          std::cout << pfn << " appears not to have been closed correctly and has been autorecovered \n";
          std::cout << "Stopping. Use --allowRecovery to try ignoring this\n";
          return 1;
        }
      } else {
        if (verbose) std::cout << "ECU:: Collection not autorecovered. Continuing\n";
      }

      // Ok. Do we have the expected trees?
      for (unsigned int i = 0; i < expectedTrees.size(); ++i) {
        TTree *t = (TTree*) tfile->Get(expectedTrees[i].c_str());
        if (t == 0) {
          std::cout << "Tree " << expectedTrees[i] << " appears to be missing. Not a valid collection\n";
          std::cout << "Exiting\n";
          return 1;
        } else {
          if (verbose) std::cout << "ECU:: Found Tree " << expectedTrees[i] << std::endl;
        }
      }

      if (verbose) std::cout << "ECU:: Found all expected trees\n";

      std::ostringstream auout;
      if (adler32) {
        unsigned int const EDMFILEUTILADLERBUFSIZE = 10*1024*1024; // 10MB buffer
        char buffer[EDMFILEUTILADLERBUFSIZE];
        size_t bufToRead = EDMFILEUTILADLERBUFSIZE;
        uint32_t a = 1, b = 0;
        size_t fileSize = tfile->GetSize();
        tfile->Seek(0, TFile::kBeg);

        for (size_t offset = 0; offset < fileSize;
              offset += EDMFILEUTILADLERBUFSIZE) {
            // true on last loop
            if (fileSize - offset < EDMFILEUTILADLERBUFSIZE)
              bufToRead = fileSize - offset;
            tfile->ReadBuffer((char*)buffer, bufToRead);
            cms::Adler32(buffer, bufToRead, a, b);
        }
        uint32_t adler32sum = (b << 16) | a;
        if (json) {
          auout << ",\"adler32sum\":" << adler32sum;
        } else {
          auout << ", " << std::hex << adler32sum << " adler32sum";
        }
      }

      if (uuid) {
        TTree *paramsTree = (TTree*)tfile->Get(edm::poolNames::metaDataTreeName().c_str());
        if (json) {
          auout << ",\"uuid\":\"" << edm::getUuid(paramsTree) << '"';
        } else {
          auout << ", " << edm::getUuid(paramsTree) << " uuid";
        }
      }

      // Ok. How many events?
      int nruns = edm::numEntries(tfile, edm::poolNames::runTreeName());
      int nlumis = edm::numEntries(tfile, edm::poolNames::luminosityBlockTreeName());
      int nevents = edm::numEntries(tfile, edm::poolNames::eventTreeName());
      if (json) {
        std::cout << "{\"file\":\"" << datafile << '"'
                  << ",\"runs\":" << nruns
                  << ",\"lumis\":" << nlumis
                  << ",\"events\":" << nevents
                  << ",\"bytes\":" << tfile->GetSize()
                  << auout.str()
                  << '}' << std::endl;
      } else {
        std::cout << datafile << " ("
                  << nruns << " runs, "
                  << nlumis << " lumis, "
                  << nevents << " events, "
                  << tfile->GetSize() << " bytes"
                  << auout.str()
                  << ")" << std::endl;
      }

      if (json) {
        // Remainder of arguments not supported in JSON yet.
        continue;
      }

      // Look at the collection contents
      if (ls) {
        if (tfile != 0) tfile->ls();
      }

      // Print out each tree
      if (print) {
        TTree *printTree = (TTree*)tfile->Get(selectedTree.c_str());
        if (printTree == 0) {
          std::cout << "Tree " << selectedTree << " appears to be missing. Could not find it in the file.\n";
          std::cout << "Exiting\n";
          return 1;
        }
        edm::printBranchNames(printTree);
      }

      if (printBranchDetails) {
        TTree *printTree = (TTree*)tfile->Get(selectedTree.c_str());
        if (printTree == 0) {
          std::cout << "Tree " << selectedTree << " appears to be missing. Could not find it in the file.\n";
          std::cout << "Exiting\n";
          return 1;
        }
        edm::longBranchPrint(printTree);
      }

      // Print out event lists
      if (events) {
        edm::printEventLists(tfile);
      }

      tfile->Close();
    }
    if (json) {
      std::cout << ']' << std::endl;
    }
  }
  catch (cms::Exception const& e) {
    std::cout << "cms::Exception caught in "
              <<"EdmFileUtil"
              << '\n'
              << e.explainSelf();
    rc = 1;
  }
  catch (std::exception const& e) {
    std::cout << "Standard library exception caught in "
              << "EdmFileUtil"
              << '\n'
              << e.what();
    rc = 1;
  }
  catch (...) {
    std::cout << "Unknown exception caught in "
              << "EdmFileUtil";
    rc = 2;
  }
  return 0;
}
