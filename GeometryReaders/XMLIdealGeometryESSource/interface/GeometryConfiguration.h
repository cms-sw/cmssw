#ifndef GeometryConfiguration_H
#define GeometryConfiguration_H

#include "DetectorDescription/Core/interface/DDLDocumentProvider.h"
class DDLParser;

#include <string>
#include <vector>
#include <memory>

/**
 */
class GeometryConfiguration: public DDLDocumentProvider {


 public:
  explicit GeometryConfiguration(std::string  fname, DDLParser & parser);
  virtual ~GeometryConfiguration(){}

  /// Print out the list of files.
  virtual void dumpFileList(void) const;

  /// Return a list of files as a vector of strings.
  virtual const std::vector < std::string >  & getFileList(void) const;

  /// Return a list of urls as a vector of strings.
  virtual const std::vector < std::string >  & getURLList(void) const;
 
 /// Return a flag whether to do xml validation or not.
  virtual bool doValidation() const;

  /// Return the Schema Location.
  virtual std::string getSchemaLocation() const;

 protected:

 private:
  GeometryConfiguration(){}
  int readConfig(const std::string& filename) { return 0;}

private:

  std::auto_ptr<DDLDocumentProvider> m_config;

  std::string configfile;
  std::string configpath;

  std::vector< std::string > myFilenames;
  std::vector< std::string > myURLs;

};

#endif
