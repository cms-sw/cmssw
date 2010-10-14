/***************************************************************************
                          DDLSAX2ConfigHandler.cc  -  description
                             -------------------
    begin                : Mon Oct 22 2001
    email                : case@ucdhep.ucdavis.edu
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *           DDDParser sub-component of DDD                                *
 *                                                                         *
 ***************************************************************************/

#include "DetectorDescription/Parser/interface/DDLParser.h"
#include "DetectorDescription/Parser/src/StrX.h"
#include "DetectorDescription/Parser/interface/DDLSAX2ConfigHandler.h"

#include "DetectorDescription/Base/interface/DDdebug.h"
#include "DetectorDescription/Core/interface/DDRoot.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"

#include <iostream>

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

  ++elementCount_;
  attrCount_ += attrs.getLength();

  std::string myelemname(StrX(qname).localForm());
  DCOUT_V('P', "DetectorDescription/Parser/interface/DDLSAX2ConfigHandler::startElement" << myelemname << " started...");

  unsigned int numAtts = attrs.getLength();
  unsigned int i = 0;
  if (myelemname == "File")
  {
    std::string name="", url="";
    while ( i < numAtts )
    {
      std::string myattname(StrX(attrs.getLocalName(i)).localForm());
      std::string myvalue(StrX(attrs.getValue(i)).localForm());

      if (myattname == "name")
	name=myvalue;
      if (myattname == "url")
	url=myvalue;
      ++i;
    }
    DCOUT('P', "file name = " << name << " and url = " << url);
    files_.push_back(name);
    urls_.push_back(url);
  }
  else if (myelemname == "Root")
  {
    std::string fileName="", logicalPartName="";
    while ( i < numAtts )
    {
      std::string myattname(StrX(attrs.getLocalName(i)).localForm());
      std::string myvalue(StrX(attrs.getValue(i)).localForm());

      if (myattname == "fileName")
	fileName = myvalue;
      if (myattname == "logicalPartName")
	logicalPartName = myvalue;
      ++i;
    }

    fileName = fileName.substr(0, fileName.find("."));
    //      std::cout << fileName << ":" << logicalPartName << " is the ROOT" << std::endl;
    DDLogicalPart root(DDName(logicalPartName,fileName));
    DDRootDef::instance().set(root);//DDName(logicalPartName, fileName));
    /// bad, just testing...
    //      DDCompactView cpv;
    //DDName rt(DDName(logicalPartName, fileName));
    cpv_.setRoot(root);
    DCOUT_V('P', std::string("DetectorDescription/Parser/interface/DDLSAX2ConfigHandler::startElement.  Setting DDRoot LogicalPart=") + logicalPartName + std::string(" in ") + fileName);  

  }
  else if (myelemname == "Schema")
  {
    while ( i < numAtts )
    {
      std::string myattname(StrX(attrs.getLocalName(i)).localForm());
      std::string myvalue(StrX(attrs.getValue(i)).localForm());
      if (myattname == "schemaLocation")
	schemaLocation_ = myvalue;
      else if (myattname == "validation")
	doValidation_ = (myvalue == "true" ? true : false);
      ++i;
    }
  }
  //  std::cout <<  "DetectorDescription/Parser/interface/DDLSAX2ConfigHandler::startElement " << myelemname << " completed..." << std::endl;
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
