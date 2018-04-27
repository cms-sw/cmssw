#include "DetectorDescription/Parser/interface/FIPConfiguration.h"
#include "DetectorDescription/Parser/interface/DDLParser.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "xercesc/util/XercesVersion.hpp"

#include <cstddef>
#include <iostream>
#include <memory>

using namespace XERCES_CPP_NAMESPACE;

FIPConfiguration::FIPConfiguration( DDCompactView& cpv )
  : configHandler_( cpv ),
    cpv_( cpv )
{}

const std::vector<std::string>&
FIPConfiguration::getFileList( void ) const
{
  return files_;
}

void
FIPConfiguration::dumpFileList(void) const
{
  std::cout << "File List:" << std::endl;
  std::cout << "  number of files=" << files_.size() << std::endl;
  for( const auto & file : files_ )
    std::cout << file << std::endl;
}

//-----------------------------------------------------------------------
//  Here the Xerces parser is used to process the content of the 
//  configuration file.
//-----------------------------------------------------------------------

int
FIPConfiguration::readConfig( const std::string& filename, bool fullPath )
{
  std::string absoluteFileName( filename );
  if( !fullPath ) {
    absoluteFileName = edm::FileInPath( filename ).fullPath();
  }

  // Set the parser to use the handler for the configuration file.
  // This makes sure the Parser is initialized and gets a handle to it.
  DDLParser ddlp( cpv_ );
  ddlp.getXMLParser()->setContentHandler( &configHandler_ );
  ddlp.getXMLParser()->parse( absoluteFileName.c_str());
  const std::vector<std::string>& vURLs = configHandler_.getURLs();
  const std::vector<std::string>& vFiles = configHandler_.getFileNames();
  size_t maxInd = vFiles.size();
  size_t ind = 0;
  // ea. file listed in the config
  for(; ind < maxInd ; ++ind)
  {
    edm::FileInPath fp(vURLs[ind] + "/" + vFiles[ind]);
    files_.emplace_back(fp.fullPath());
  }

  return 0;
}

int
FIPConfiguration::readConfig( const std::string& filename )
{
  return readConfig( filename, false );
}
