
#include <iostream>
#include <string>
#include <vector>
#include <exception>

#include <boost/program_options.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/fstream.hpp>

#include "TFile.h"
#include "TError.h"

#include "Utilities/StorageFactory/interface/Storage.h"
#include "Utilities/StorageFactory/interface/StorageFactory.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/Services/src/SiteLocalConfigService.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Catalog/interface/InputFileCatalog.h"
#include "FWCore/Catalog/interface/SiteLocalConfig.h"

static int copy_files(const boost::program_options::variables_map& vm) {
  std::unique_ptr<edm::SiteLocalConfig> slcptr =
      std::make_unique<edm::service::SiteLocalConfigService>(edm::ParameterSet());
  auto slc = std::make_shared<edm::serviceregistry::ServiceWrapper<edm::SiteLocalConfig> >(std::move(slcptr));
  edm::ServiceToken slcToken = edm::ServiceRegistry::createContaining(slc);
  edm::ServiceRegistry::Operate operate(slcToken);

  auto in = (vm.count("file") ? vm["file"].as<std::vector<std::string> >() : std::vector<std::string>());

  if (in.size() < 2) {
    std::cerr << "Not enough arguments!" << std::endl;
    std::cerr << "Usage: edmCopyUtil [file1] [file2] [file3] dest_dir" << std::endl;
    return 1;
  }

  boost::filesystem::path destdir(in.back());
  in.pop_back();

  if (!boost::filesystem::is_directory(destdir)) {
    std::cerr << "Last argument must be destination directory; " << destdir << " is not a directory!" << std::endl;
    return 1;
  }

  std::string catalogIn = (vm.count("catalog") ? vm["catalog"].as<std::string>() : std::string());
  edm::InputFileCatalog catalog(in, catalogIn, true);
  std::vector<std::string> const& filesIn = catalog.fileNames();

  for (unsigned int j = 0; j < in.size(); ++j) {
    boost::filesystem::path pathOut = destdir;
    pathOut /= boost::filesystem::path(in[j]).filename();

    boost::filesystem::ofstream ofs;
    ofs.exceptions(std::ofstream::failbit | std::ofstream::badbit);
    ofs.open(pathOut);

    std::unique_ptr<Storage> s = StorageFactory::get()->open(filesIn[j]);
    assert(s);  // StorageFactory should throw if file open fails.

    static unsigned int const COPYBUFSIZE = 10 * 1024 * 1024;  // 10MB buffer
    std::vector<char> buffer;
    buffer.reserve(COPYBUFSIZE);

    IOSize n;
    while ((n = s->read(&buffer[0], COPYBUFSIZE))) {  // Note Storage throws on error
      ofs.write(&buffer[0], n);
    }
    ofs.close();
    s->close();
  }

  return 0;
}

int main(int argc, char* argv[]) {
  gErrorIgnoreLevel = kError;

  boost::program_options::options_description desc("Allowed options");
  desc.add_options()("help,h", "print help message")(
      "catalog,c", boost::program_options::value<std::string>(), "catalog");
  boost::program_options::options_description hidden("Hidden options");
  hidden.add_options()("file", boost::program_options::value<std::vector<std::string> >(), "files to transfer");

  boost::program_options::positional_options_description p;
  p.add("file", -1);

  boost::program_options::options_description cmdline_options;
  cmdline_options.add(desc).add(hidden);

  boost::program_options::variables_map vm;

  try {
    boost::program_options::store(
        boost::program_options::command_line_parser(argc, argv).options(cmdline_options).positional(p).run(), vm);
  } catch (boost::program_options::error const& x) {
    std::cerr << "Option parsing failure:\n" << x.what() << "\n\n";
    std::cerr << desc << "\n";
    return 1;
  }

  boost::program_options::notify(vm);

  if (vm.count("help")) {
    std::cout << desc << "\n";
    return 1;
  }

  int rc;
  try {
    rc = copy_files(vm);
  } catch (cms::Exception const& e) {
    std::cout << "cms::Exception caught in "
              << "EdmFileUtil" << '\n'
              << e.explainSelf();
    rc = 1;
  } catch (std::exception const& e) {
    std::cout << "Standard library exception caught in "
              << "EdmFileUtil" << '\n'
              << e.what();
    rc = 1;
  } catch (...) {
    std::cout << "Unknown exception caught in "
              << "EdmFileUtil";
    rc = 2;
  }
  return rc;
}
