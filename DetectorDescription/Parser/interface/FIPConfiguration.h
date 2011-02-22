#ifndef DetectorDescription_Parser_XMLConfiguration_H
#define DetectorDescription_Parser_XMLConfiguration_H

// ---------------------------------------------------------------------------
//  Includes
// ---------------------------------------------------------------------------
#include "DetectorDescription/Parser/interface/DDLDocumentProvider.h"
#include "DetectorDescription/Parser/interface/DDLSAX2ConfigHandler.h"

// From DD Core
#include "DetectorDescription/Core/interface/DDCompactView.h"

class DDLParser;
class DDLSAX2Handler;
class DDLSAX2ConfigHandler;

#include <string>
#include <vector>
#include <map>

/// FIPConfiguration reads in the configuration file for the DDParser.
/** @class FIPConfiguration
 * @author Michael Case
 *
 *  FIPConfiguration.h  -  description
 *  -------------------
 *  begin: Sun Nov 13, 2005
 *  email: case@ucdhep.ucdavis.edu
 *
 */
class FIPConfiguration : public DDLDocumentProvider {


 public:

  FIPConfiguration( DDCompactView& cpv);
  virtual ~FIPConfiguration();

  /// Read in the configuration file.
  int readConfig(const std::string& filename);

  /// Return a list of files as a std::vector of strings.
  virtual const std::vector < std::string >&  getFileList(void) const;

  /// Return a list of urls as a std::vector of strings.
  /**
     This implementation does not provide a meaningful url list.
   **/
  virtual const std::vector < std::string >&  getURLList(void) const;

  /// Print out the list of files.
  virtual void dumpFileList(void) const;

  /// Return whether Validation should be on or off and where the DDL SchemaLocation is.
  virtual bool doValidation() const;

  /// Return the designation for where to look for the schema.
  std::string getSchemaLocation() const;

 protected:

 private:
  DDLSAX2ConfigHandler configHandler_;
  std::vector<std::string> files_;
  std::vector<std::string> urls_;
  DDCompactView& cpv_;
};

#endif
