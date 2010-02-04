#ifndef DDL_Configuration_H
#define DDL_Configuration_H

// ---------------------------------------------------------------------------
//  Includes
// ---------------------------------------------------------------------------
#include "DetectorDescription/Parser/interface/DDLDocumentProvider.h"
#include "DetectorDescription/Parser/interface/DDLSAX2ConfigHandler.h"
#include <xercesc/util/XercesDefs.hpp>
#include <xercesc/sax2/SAX2XMLReader.hpp>

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
  typedef XERCES_CPP_NAMESPACE::SAX2XMLReader SAX2XMLReader;

  explicit DDLConfiguration(DDLParser *, DDCompactView&);
  DDLConfiguration(DDCompactView&);
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
  SAX2XMLReader*  parser_;
  DDLSAX2ConfigHandler configHandler_;
  DDCompactView& cpv_;
};

#endif
