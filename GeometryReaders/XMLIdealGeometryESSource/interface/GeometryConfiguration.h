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
class GeometryConfiguration : public DDLDocumentProvider {
public:
  GeometryConfiguration(const edm::ParameterSet& p);

  ~GeometryConfiguration() override;

  /// Print out the list of files.
  void dumpFileList(void) const override;

  /// Return a list of files as a vector of strings.
  const std::vector<std::string>& getFileList(void) const override;

  /// Return a list of urls as a vector of strings.
  /**
     The EDM should not allow URLs because of provenance.
     This vector will always be empty.
   **/
  const std::vector<std::string>& getURLList(void) const override;

  /// Return a flag whether to do xml validation or not.
  bool doValidation() const override;

  /// Return the Schema Location.
  std::string getSchemaLocation() const override;

  /// Reads in a configuration file and parses it
  int readConfig(const std::string& filename) override;

protected:
private:
  std::vector<std::string> files_;
  std::vector<std::string> relFiles_;
  std::vector<std::string> emptyStrings_;
  std::string dummyLocation_;
};

#endif
