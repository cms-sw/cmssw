#ifndef DETECTOR_DESCRIPTION_PARSER_FIP_CONFIGURATION_H
#define DETECTOR_DESCRIPTION_PARSER_FIP_CONFIGURATION_H

#include <map>
#include <string>
#include <vector>

#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Parser/interface/DDLDocumentProvider.h"
#include "DetectorDescription/Parser/interface/DDLSAX2ConfigHandler.h"

class DDCompactView;
class DDLParser;
class DDLSAX2ConfigHandler;
class DDLSAX2Handler;

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
  FIPConfiguration(DDCompactView& cpv);
  ~FIPConfiguration() override;

  /// Read in the configuration file.
  int readConfig(const std::string& filename) override;

  /// Read in the configuration file.
  int readConfig(const std::string& filename, bool fullPath);

  /// Return a list of files as a std::vector of strings.
  const std::vector<std::string>& getFileList(void) const override;

  /// Return a list of urls as a std::vector of strings.
  /**
     This implementation does not provide a meaningful url list.
   **/
  const std::vector<std::string>& getURLList(void) const override;

  /// Print out the list of files.
  void dumpFileList(void) const override;

  /// Return whether Validation should be on or off and where the DDL SchemaLocation is.
  bool doValidation() const override;

  /// Return the designation for where to look for the schema.
  std::string getSchemaLocation() const override;

private:
  DDLSAX2ConfigHandler configHandler_;
  std::vector<std::string> files_;
  std::vector<std::string> urls_;
  DDCompactView& cpv_;
};

#endif
