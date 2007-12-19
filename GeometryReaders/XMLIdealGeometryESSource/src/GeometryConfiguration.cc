/**
   Editted By     On
   Michael Case   Sun Nov 13 2005
 **/
#include "GeometryReaders/XMLIdealGeometryESSource/interface/GeometryConfiguration.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "DetectorDescription/Parser/interface/DDLParser.h"
#include "DetectorDescription/Base/interface/DDdebug.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <string>
#include <vector>

GeometryConfiguration::GeometryConfiguration( const edm::ParameterSet& pset ) : dummyLocation_("") { 
  relFiles_ = pset.getParameter<std::vector<std::string> >("geomXMLFiles");
  for (std::vector<std::string>::const_iterator rit = relFiles_.begin(), ritEnd = relFiles_.end();
      rit != ritEnd; ++rit ) {
    edm::FileInPath fp(*rit);
    files_.push_back(fp.fullPath());
    emptyStrings_.push_back("");
  }
}

GeometryConfiguration::~GeometryConfiguration() { }

/// Return the Schema Location.
std::string GeometryConfiguration::getSchemaLocation() const {
  edm::LogError("GeometryConfiguration") << " This sub-class of DDLDocumentProvider does not USE XML parsing!!!" << std::endl;
  return dummyLocation_;
}

/// Return a flag whether to do xml validation or not.
bool GeometryConfiguration::doValidation() const {
  LogDebug("GeometryConfiguration") << " the doValidation() method not valid for this DDLDocumentProvider" << std::endl;
  return false;
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
  LogDebug("GeometryConfiguration") << " the getURLList of this DDLDocumentProvider empty strings" << std::endl;
  //  return relFiles_;
  return emptyStrings_;
}

/// Print out the list of files.
void GeometryConfiguration::dumpFileList(void) const {
  std::cout << "File List:" << std::endl;
  std::cout << "  number of files=" << files_.size() << std::endl;
  for (std::vector<std::string>::const_iterator it = files_.begin(), itEnd = files_.end(); it != itEnd; ++it)
    std::cout << *it << std::endl;
}

int GeometryConfiguration::readConfig( const std::string& fname ) {
  edm::LogWarning("GeometryConfiguration") << " The readConfig of this DDLDocumentProvider is not valid!" << std::endl;
  return 0;
}

