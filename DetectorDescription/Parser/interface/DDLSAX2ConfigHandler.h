#ifndef DETECTOR_DESCRIPTION_PARSER_DDL_SAX2_CONFIG_HANDLER_H
#define DETECTOR_DESCRIPTION_PARSER_DDL_SAX2_CONFIG_HANDLER_H

#include "DetectorDescription/Parser/interface/DDLSAX2Handler.h"
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"

#include <vector>
#include <string>
#include <xercesc/sax2/Attributes.hpp>

/// DDLSAX2ConfigHandler is the handler for the configuration file.
/** @class DDLSAX2ConfigHandler
 * @author Michael Case
 * 
 *  DDLSAX2ConfigHandler.h  -  description
 *  -------------------
 *  begin: Mon Oct 22 2001
 *  email: case@ucdhep.ucdavis.edu
 *
 *  This handler is used by the DDLParser to process configuration files.
 */
class DDLSAX2ConfigHandler : public DDLSAX2Handler {
public:
  DDLSAX2ConfigHandler(DDCompactView& cpv);
  ~DDLSAX2ConfigHandler() override;

  // -----------------------------------------------------------------------
  //  Handlers for the SAX ContentHandler interface
  // -----------------------------------------------------------------------
  void startElement(const XMLCh* uri, const XMLCh* localname, const XMLCh* qname, const Attributes& attrs) override;

  const std::vector<std::string>& getFileNames() const;
  const std::vector<std::string>& getURLs() const;
  const std::string getSchemaLocation() const;
  const bool doValidation() const;

private:
  bool doValidation_;
  std::vector<std::string> files_;
  std::vector<std::string> urls_;
  std::string schemaLocation_;
  DDCompactView& cpv_;
};

#endif
