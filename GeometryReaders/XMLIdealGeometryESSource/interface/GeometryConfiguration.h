#ifndef GEOMETRY_READERS_GEOMETRY_CONFIGURATION_H
#define GEOMETRY_READERS_GEOMETRY_CONFIGURATION_H

#include "DetectorDescription/Parser/interface/DDLDocumentProvider.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>
#include <vector>

/**
   This class provides the filenames to the DDLParser from the
   parameter-set passed by XMLIdealGeometryESSource.
   The list of XML files is in the parameter-set for provenance.
 **/
class GeometryConfiguration: public DDLDocumentProvider {

 public:
  GeometryConfiguration( const edm::ParameterSet & p );

  /// Print out the list of files.
  void dumpFileList() const override;

  /// Return a list of files as a vector of strings.
  const std::vector < std::string > & getFileList() const override;

 private:
  std::vector< std::string > files_;
  std::vector< std::string > relFiles_;
};

#endif
