#include "DetectorDescription/Parser/interface/FIPConfiguration.h"

#include <ext/alloc_traits.h>
#include <stddef.h>
#include <iostream>

#include "DetectorDescription/Parser/interface/DDLParser.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "xercesc/util/XercesVersion.hpp"

class DDCompactView;

using namespace XERCES_CPP_NAMESPACE;

FIPConfiguration::FIPConfiguration( DDCompactView& cpv )
  : configHandler_( cpv ),
    cpv_( cpv )
{}

FIPConfiguration::~FIPConfiguration( void )
{}

const std::vector<std::string>&
FIPConfiguration::getFileList( void ) const
{
  return files_;
}

const std::vector<std::string>&
FIPConfiguration::getURLList( void ) const
{
  return urls_;
}

bool
FIPConfiguration::doValidation( void ) const
{
  return configHandler_.doValidation();
}

std::string
FIPConfiguration::getSchemaLocation( void ) const
{
  return configHandler_.getSchemaLocation();
}

void
FIPConfiguration::dumpFileList(void) const
{
  std::cout << "File List:" << std::endl;
  std::cout << "  number of files=" << files_.size() << std::endl;
  for (const auto & file : files_)
    std::cout << file << std::endl;
}

//-----------------------------------------------------------------------
//  Here the Xerces parser is used to process the content of the 
//  configuration file.
//-----------------------------------------------------------------------

int
FIPConfiguration::readConfig( const std::string& filename, bool fullPath )
{
  std::string absoluteFileName (filename);
  if (!fullPath) {
    edm::FileInPath fp(filename);
    // config file
    absoluteFileName = fp.fullPath();
  }

  // Set the parser to use the handler for the configuration file.
  // This makes sure the Parser is initialized and gets a handle to it.
  DDLParser ddlp(cpv_);
  ddlp.getXMLParser()->setContentHandler(&configHandler_);
  ddlp.getXMLParser()->parse(absoluteFileName.c_str());
  const std::vector<std::string>& vURLs = configHandler_.getURLs();
  const std::vector<std::string>& vFiles = configHandler_.getFileNames();
  size_t maxInd = vFiles.size();
  size_t ind = 0;
  // ea. file listed in the config
  for(; ind < maxInd ; ++ind)
  {
    edm::FileInPath fp(vURLs[ind] + "/" + vFiles[ind]);
    //    std::cout << "FileInPath says..." << fp.fullPath() << std::endl;
    files_.push_back(fp.fullPath());
    urls_.push_back("");
  }

  //   std::vector<std::string> fnames = configHandler_.getFileNames();
  //   std::cout << "there are " << fnames.size() << " files." << std::endl;
  //   for (size_t i = 0; i < fnames.size(); ++i)
  //     std::cout << "url=" << configHandler_.getURLs()[i] << " file=" << configHandler_.getFileNames()[i] << std::endl;
  return 0;
}

int
FIPConfiguration::readConfig( const std::string& filename )
{
  return readConfig( filename, false );
}
