/**
   Editted By     On
   Michael Case   Sun Nov 13 2005
 **/
#include "GeometryReaders/XMLIdealGeometryESSource/interface/GeometryConfiguration.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "DetectorDescription/Parser/interface/DDLParser.h"
#include "DetectorDescription/Base/interface/DDdebug.h"

#include <string>
#include <vector>

GeometryConfiguration::GeometryConfiguration() : configHandler_ (){ }

GeometryConfiguration::~GeometryConfiguration() { }

/// Return the Schema Location.
std::string GeometryConfiguration::getSchemaLocation() const {
  return configHandler_.getSchemaLocation();
}

/// Return a flag whether to do xml validation or not.
bool GeometryConfiguration::doValidation() const {
  return configHandler_.doValidation();
}

/// Return a list of files as a vector of strings.
const std::vector < std::string >  & GeometryConfiguration::getFileList(void) const {
  return files_;
}

/// Return a list of urls as a vector of strings.
/**
   The EDM should not allow URLs because of provenance.
   This vector will always be empty.
**/
const std::vector < std::string >  & GeometryConfiguration::getURLList(void) const
{
  return urls_;
}

/// Print out the list of files.
void GeometryConfiguration::dumpFileList(void) const {
  std::cout << "File List:" << std::endl;
  std::cout << "  number of files=" << files_.size() << std::endl;
  for (std::vector<std::string>::const_iterator it = files_.begin(); it != files_.end(); it++)
    std::cout << *it << std::endl;
}

int GeometryConfiguration::readConfig( const std::string& fname ) {

  DCOUT('X', "readConfig() about to read " + fname );
  //get hold of the DDLParser
  DDLParser * parser = DDLParser::instance();

  //set the content handler for the xerces:sax2parser
  parser->getXMLParser()->setContentHandler(&configHandler_);

  //parse the configuration with the xerces:sax2parser
  parser->getXMLParser()->parse(fname.c_str());

  //the handler keeps record of all files it processed.
  const std::vector<std::string>& vURLs = configHandler_.getURLs();
  const std::vector<std::string>& vFiles = configHandler_.getFileNames();

  //change the files to be full path names
  //since we are bypassing the original intent, we need to provide a vector
  //of empty strings for the urls to match the files.
  size_t maxInd = vFiles.size();
  size_t ind = 0;
  for ( ; ind < maxInd ; ind++) {
    edm::FileInPath fp(vURLs[ind] + "/" + vFiles[ind]);
    files_.push_back(fp.fullPath());
    urls_.push_back("");
  }
  std::cout << "======== Geometry Configuration read ==========" << std::endl;
  return 0;
}

