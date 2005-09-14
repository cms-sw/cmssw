#ifndef DDL_Configuration_H
#define DDL_Configuration_H

// ---------------------------------------------------------------------------
//  Includes
// ---------------------------------------------------------------------------
#include "DetectorDescription/Parser/interface/DDLDocumentProvider.h"

class DDLParser;
class DDLSAX2Handler;
class DDLSAX2ConfigHandler;

#include <string>
#include <vector>
#include <map>

/// DDLConfiguration reads in the configuration file for the DDParser.
/** @class DDLConfiguration
 * @author Michael Case
 *
 *  DDLConfiguration.h  -  description
 *  -------------------
 *  begin: Mon Feb 24 2003
 *  email: case@ucdhep.ucdavis.edu
 *
 */
class DDLConfiguration : public DDLDocumentProvider {

  //  friend DDLParser;

 public:

  explicit DDLConfiguration(DDLParser *);
  DDLConfiguration();
  virtual ~DDLConfiguration();

  /// Read in the configuration file.
  int readConfig(const std::string& filename);

  /// Return a list of files as a std::vector of strings.
  virtual const std::vector < std::string >&  getFileList(void) const;

  /// Return a list of urls as a std::vector of strings.
  virtual const std::vector < std::string >&  getURLList(void) const;

  /// Print out the list of files.
  virtual void dumpFileList(void) const;

  /// Return whether Validation should be on or off and where the DDL SchemaLocation is.
  virtual bool doValidation() const;

  /// Return the designation for where to look for the schema.
  std::string getSchemaLocation() const;

 protected:

 private:
  DDLParser * m_parser;
  DDLSAX2Handler* errHandler_;
  DDLSAX2ConfigHandler * sch_;
};

#endif
