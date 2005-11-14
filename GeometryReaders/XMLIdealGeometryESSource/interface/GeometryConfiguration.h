#ifndef GeometryConfiguration_H
#define GeometryConfiguration_H

#include "DetectorDescription/Core/interface/DDLDocumentProvider.h"
#include "DetectorDescription/Parser/interface/DDLSAX2ConfigHandler.h"

class DDLParser;

#include <string>
#include <vector>
#include <memory>

/**
 **/
class GeometryConfiguration: public DDLDocumentProvider {

 public:
  GeometryConfiguration();

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

  DDLSAX2ConfigHandler configHandler_;
  std::vector< std::string > files_;
  std::vector< std::string > urls_;

};

#endif
