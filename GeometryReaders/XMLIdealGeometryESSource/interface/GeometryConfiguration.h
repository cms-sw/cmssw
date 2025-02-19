#ifndef GeometryConfiguration_H
#define GeometryConfiguration_H

#include "DetectorDescription/Parser/interface/DDLDocumentProvider.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class DDLParser;

#include <string>
#include <vector>
#include <memory>

/**
   May 23, 2006:  Michael Case:
   This class provides the filenames to the DDLParser from the
   parameter-set passed by XMLIdealGeometryESSource.  This removes 
   the dependency on the Configuration Language of the DDD and moves
   the list of XML files to the parameter-set for provenance.
 **/
class GeometryConfiguration: public DDLDocumentProvider {

 public:
  GeometryConfiguration( const edm::ParameterSet & p );

  virtual ~GeometryConfiguration();

  /// Print out the list of files.
  virtual void dumpFileList(void) const;

  /// Return a list of files as a vector of strings.
  virtual const std::vector < std::string >  & getFileList(void) const;

  /// Return a list of urls as a vector of strings.
  /**
     The EDM should not allow URLs because of provenance.
     This vector will always be empty.
   **/
  virtual const std::vector < std::string >  & getURLList(void) const;
 
  /// Return a flag whether to do xml validation or not.
  virtual bool doValidation() const;

  /// Return the Schema Location.
  virtual std::string getSchemaLocation() const;

  /// Reads in a configuration file and parses it
  int readConfig(const std::string& filename);

 protected:

 private:
  std::vector< std::string > files_;
  std::vector< std::string > relFiles_;
  std::vector< std::string > emptyStrings_;
  std::string dummyLocation_;
};

#endif
