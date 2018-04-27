#ifndef DETECTOR_DESCRIPTION_PARSER_FIP_CONFIGURATION_H
#define DETECTOR_DESCRIPTION_PARSER_FIP_CONFIGURATION_H

#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Parser/interface/DDLDocumentProvider.h"
#include "DetectorDescription/Parser/interface/DDLSAX2ConfigHandler.h"

#include <string>
#include <vector>

/// FIPConfiguration reads in the configuration file for the DDParser.
/** @class FIPConfiguration
 * @author Michael Case
 *
 */
class FIPConfiguration : public DDLDocumentProvider
{
 public:

  FIPConfiguration( DDCompactView& cpv );

  /// Read in the configuration file.
  int readConfig( const std::string& filename );

  /// Read in the configuration file.
  int readConfig( const std::string& filename, bool fullPath );

  /// Return a list of files as a std::vector of strings.
  const std::vector < std::string >&  getFileList() const override;

  /// Print out the list of files.
  void dumpFileList() const override;

 private:
  
  DDLSAX2ConfigHandler configHandler_;
  std::vector<std::string> files_;
  DDCompactView& cpv_;
};

#endif
