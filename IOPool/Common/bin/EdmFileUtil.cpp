//----------------------------------------------------------------------
// EdmFileUtil.cpp
//
// Author: Chih-hsiang Cheng, LLNL
//         Chih-Hsiang.Cheng@cern.ch
//
// March 13, 2006
//

#include <exception>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <boost/program_options.hpp>
#include "IOPool/Common/bin/CollUtil.h"
#include "DataFormats/Provenance/interface/BranchType.h"
#include "FWCore/RootAutoLibraryLoader/interface/RootAutoLibraryLoader.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Catalog/interface/InputFileCatalog.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"

#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/PluginManager/interface/PluginManager.h"

#include "TFile.h"


int main(int argc, char* argv[]) {

  // Add options here

  boost::program_options::options_description desc("Allowed options");
  desc.add_options()
    ("help,h", "print help message")
    ("file,f", boost::program_options::value<std::vector<std::string> >(), "data file (Required)")
    ("catalog,c", boost::program_options::value<std::string>(), "catalog")
    ("ls,l", "list file content")
    ("print,P", "Print all")
    ("uuid,u", "Print uuid")
    ("verbose,v","Verbose printout")
    ("decodeLFN,d", "Convert LFN to PFN")
    ("printBranchDetails,b","Call Print()sc for all branches")
    ("tree,t", boost::program_options::value<std::string>(), "Select tree used with -P and -b options")
    ("allowRecovery","Allow root to auto-recover corrupted files") 
    ("events,e", "Print list of all Events, Runs, and LuminosityBlocks in the file sorted by run number, luminosity block number, and event number.  Also prints the entry numbers and whether it is possible to use fast copy with the file.");

  // What trees do we require for this to be a valid collection?
  std::vector<std::string> expectedTrees;
  expectedTrees.push_back(edm::poolNames::metaDataTreeName());
  expectedTrees.push_back(edm::poolNames::eventTreeName());

  boost::program_options::positional_options_description p;
  p.add("file", -1);

  boost::program_options::variables_map vm;


  try
    {
      boost::program_options::store(boost::program_options::command_line_parser(argc, argv).
				    options(desc).positional(p).run(), vm);
    }
  catch (boost::program_options::error const& x)
    {
      std::cerr << "Option parsing failure:\n"
		<< x.what() << "\n\n";
      std::cerr << desc << "\n";
      return 1;
    }





  boost::program_options::notify(vm);    

  bool verbose= vm.count("verbose") > 0 ? true : false;

  if (vm.count("help")) {
    std::cout << desc << "\n";
    return 1;
  }

  if (!vm.count("file")) {
    std::cout << "Data file not set.\n";
    std::cout << desc << "\n";
    return 1;
  }

  //dl  std::string datafile = vm["file"].as<std::string>(); 
  try {
    edmplugin::PluginManager::configure(edmplugin::standard::config());
  } catch(cms::Exception& e) {
    std::cout << "cms::Exception caught in "
    <<"EdmFileUtil"
    << '\n'
    << e.what();
    return 1;
  }
  
  
  edm::RootAutoLibraryLoader::enable();

  int rc = 0;
  try {
    std::string config =
      "import FWCore.ParameterSet.Config as cms\n"
      "process = cms.Process('EdmFileUtil')\n"
      "process.SiteLocalConfigService = cms.Service('SiteLocalConfigService')\n";

    //create the services
    edm::ServiceToken tempToken = edm::ServiceRegistry::createServicesFromConfig(config);

    //make the services available
    edm::ServiceRegistry::Operate operate(tempToken);

    // now run..
    edm::ParameterSet pset;
    std::vector<std::string> in = vm["file"].as<std::vector<std::string> >();
    std::string catalogIn = (vm.count("catalog") ? vm["catalog"].as<std::string>() : std::string());
    
    pset.addUntrackedParameter<std::vector<std::string> >("fileNames", in);
    pset.addUntrackedParameter<std::string>("catalog", catalogIn);
    
    edm::PoolCatalog poolcat;
    edm::InputFileCatalog catalog(pset, poolcat);
    std::vector<std::string> const& filesIn = catalog.fileNames();

    // Allow user to input multiple files
    for(unsigned int j = 0; j < in.size(); ++j) {
      
      // We _only_ want the LFN->PFN conversion. No need to open the file, 
      // just check the catalog and move on
      if ( vm.count("decodeLFN") ) {
	std::cout << filesIn[j] << std::endl;
	continue;
      }

      // open a data file
      std::cout << in[j] << "\n"; 
      std::string datafile=filesIn[j];
      TFile *tfile= edm::openFileHdl(datafile);
      
      if ( tfile == 0 ) return 1;
      if ( verbose ) std::cout << "ECU:: Opened " << datafile << std::endl;
      
      // First check that this file is not auto-recovered
      // Stop the job unless specified to do otherwise
      
      bool isRecovered = tfile->TestBit(TFile::kRecovered);
      if ( isRecovered ) {
	std::cout << datafile << " appears not to have been closed correctly and has been autorecovered \n";
	if ( vm.count("allowRecovery") ) {
	  std::cout << "Proceeding anyway\n";
	}
	else{
	  std::cout << "Stopping. Use --allowRecovery to try ignoring this\n";
	  return 1;
	}
      }
      else{
	if ( verbose ) std::cout << "ECU:: Collection not autorecovered. Continuing\n";
      }
      
      // Ok. Do we have the expected trees?
      for ( unsigned int i = 0; i < expectedTrees.size(); ++i) {
	TTree *t = (TTree*) tfile->Get(expectedTrees[i].c_str());
	if ( t==0 ) {
	  std::cout << "Tree " << expectedTrees[i] << " appears to be missing. Not a valid collection\n";
	  std::cout << "Exiting\n";
	  return 1;
	}
	else{
	if ( verbose ) std::cout << "ECU:: Found Tree " << expectedTrees[i] << std::endl;
	}
      }
      
      if ( verbose ) std::cout << "ECU:: Found all expected trees\n"; 
      
      // Ok. How many events?
      int nevts= edm::numEntries(tfile,edm::poolNames::eventTreeName());
      std::cout << tfile->GetName() << " ( " << nevts << " events, " 
		<< tfile->GetSize() << " bytes )" << std::endl;
      
      // Look at the collection contents
      if ( vm.count("ls")) {
	if ( tfile != 0 ) tfile->ls();
      }
      
      std::string selectedTree = vm.count("tree") ? vm["tree"].as<std::string>() :
                                                    edm::poolNames::eventTreeName().c_str();

      // Print out each tree
      if ( vm.count("print") ) {
        TTree *printTree=(TTree*)tfile->Get(selectedTree.c_str());
        if (printTree == 0) {
          std::cout << "Tree " << selectedTree << " appears to be missing. Could not find it in the file.\n";
          std::cout << "Exiting\n";
          return 1;
        }
        edm::printBranchNames(printTree);
      }
      
      if ( vm.count("printBranchDetails") ) {
        TTree *printTree=(TTree*)tfile->Get(selectedTree.c_str());
        if (printTree == 0) {
          std::cout << "Tree " << selectedTree << " appears to be missing. Could not find it in the file.\n";
          std::cout << "Exiting\n";
          return 1;
        }
	edm::longBranchPrint(printTree);
      }
      
      if ( vm.count("uuid") ) {
	TTree *paramsTree=(TTree*)tfile->Get(edm::poolNames::metaDataTreeName().c_str());
	edm::printUuids(paramsTree);
      }

      // Print out event lists 
      if ( vm.count("events") ) {
	edm::printEventLists(tfile);
      }
      tfile->Close();
    }
  }
  
  catch (cms::Exception& e) {
    std::cout << "cms::Exception caught in "
              <<"EdmFileUtil"
              << '\n'
              << e.explainSelf();
    rc = 1;
  }
  catch (std::exception& e) {
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

