#include "DetectorDescription/Parser/interface/DDLSAX2ConfigHandler.h"
#include "DetectorDescription/Parser/interface/DDLSAX2Handler.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDRoot.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Utilities/Xerces/interface/XercesStrUtils.h"

#include <string>
#include <vector>

using namespace cms::xerces;

DDLSAX2ConfigHandler::DDLSAX2ConfigHandler( DDCompactView& cpv)
  : doValidation_(false),
    files_(),
    urls_(),
    schemaLocation_(),
    cpv_(cpv)
{}

DDLSAX2ConfigHandler::~DDLSAX2ConfigHandler( void )
{}

// ---------------------------------------------------------------------------
//  DDLSAX2Handler: Implementation of the SAX DocumentHandler interface
//  
//  This is kind-of sneaky-cheating.  Basically it ignores all elements except 
//  File elements, and displays attributes up to the name attribute because that
//  is the only one it cares about at this time.  Later we would need a mechanism
//  to exclude and include sections based on the configuration if we proceed with
//  that level of selectivity.
//
//  The file name is passed back to DDLParser via SetDDLFileName in order to 
//  process this list of files later.
// ---------------------------------------------------------------------------
void
DDLSAX2ConfigHandler::startElement( const XMLCh* const uri,
				    const XMLCh* const localname,
				    const XMLCh* const qname,
				    const Attributes& attrs )
{
  if( XMLString::equals( qname, uStr("File").ptr()))
  {
    std::string name = toString(attrs.getValue(uStr("name").ptr()));
    std::string url = toString(attrs.getValue(uStr("url").ptr()));

    files_.emplace_back(name);
    urls_.emplace_back(url);
  }
  else if( XMLString::equals( qname, uStr("Root").ptr()))
  {
    std::string fileName = toString(attrs.getValue(uStr("fileName").ptr()));
    std::string logicalPartName = toString(attrs.getValue(uStr("logicalPartName").ptr()));

    fileName = fileName.substr(0, fileName.find("."));
    DDLogicalPart root(DDName(logicalPartName,fileName));
    DDRootDef::instance().set(root);
    cpv_.setRoot(root);
  }
  else if( XMLString::equals( qname, uStr("Schema").ptr()))
  {
    schemaLocation_ = toString(attrs.getValue(uStr("schemaLocation").ptr()));
    doValidation_ = XMLString::equals(attrs.getValue(uStr("validation").ptr()), uStr("true").ptr());
  }
}

const std::vector<std::string>&
DDLSAX2ConfigHandler::getFileNames( void ) const
{
  return files_;
}

const std::vector<std::string>&
DDLSAX2ConfigHandler::getURLs( void ) const
{
  return urls_;
}

const std::string
DDLSAX2ConfigHandler::getSchemaLocation( void ) const
{
  return schemaLocation_;
}

const bool
DDLSAX2ConfigHandler::doValidation( void ) const
{
  return doValidation_;
}
