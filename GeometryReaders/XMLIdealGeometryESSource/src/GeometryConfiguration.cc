#include "GeometryReaders/XMLIdealGeometryESSource/interface/GeometryConfiguration.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <string>
#include <vector>

GeometryConfiguration::GeometryConfiguration( const edm::ParameterSet& pset )
{ 
  relFiles_ = pset.getParameter<std::vector<std::string> >( "geomXMLFiles" );
  for( auto const& rit : relFiles_ ) {
    files_.emplace_back( edm::FileInPath(rit).fullPath());
  }
}

/// Return a list of files as a vector of strings.
const std::vector < std::string > &
GeometryConfiguration::getFileList() const {
  return files_;
}

/// Print out the list of files.
void
GeometryConfiguration::dumpFileList() const {
  edm::LogVerbatim("GeometryConfiguration") << "File List:\n"
					    << "  number of files=" << files_.size() << "\n";
  for (const auto & file : files_)
    edm::LogVerbatim("GeometryConfiguration") << file << "\n";
}

