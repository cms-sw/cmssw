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

namespace std{} using namespace std;

// ---------------------------------------------------------------------------
//  Includes
// ---------------------------------------------------------------------------
#include "DetectorDescription/Parser/interface/DDLParser.h"
#include "DetectorDescription/Parser/interface/DDLSAX2ConfigHandler.h"

// Xerces C++ dependencies.
#include <xercesc/sax2/Attributes.hpp>
#include <xercesc/sax/SAXParseException.hpp>
#include <xercesc/sax/SAXException.hpp>

// DDCore dependencies
#include "DetectorDescription/Base/interface/DDdebug.h"
//  This is frustrating.  I only need DDInit at this level to get a handle on DDRootDef.
//#include "DetectorDescription/Core/interface/DDInit.h"
#include "DetectorDescription/Core/interface/DDRoot.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"

#include <iostream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
//  DDLSAX2ConfigHandler: Constructors and Destructor
// ---------------------------------------------------------------------------
DDLSAX2ConfigHandler::DDLSAX2ConfigHandler() : doValidation_(false), files_(), urls_(), schemaLocation_()
{
}

DDLSAX2ConfigHandler::~DDLSAX2ConfigHandler() { }

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
void DDLSAX2ConfigHandler::startElement(const XMLCh* const uri
					, const XMLCh* const localname
					, const XMLCh* const qname
					, const Attributes& attrs)
{

  elementCount_++;
  attrCount_ += attrs.getLength();

  std::string myelemname = std::string(XMLString::transcode(qname));
  DCOUT_V('P', "DDLSAX2ConfigHandler::startElement" << myelemname << " started...");

  unsigned int numAtts = attrs.getLength();
  unsigned int i = 0;
  if (myelemname == "File")
    {
      std::string name="", url="";
      while ( i < numAtts )
	{
          std::string myattname = std::string(XMLString::transcode(attrs.getLocalName(i)));
          std::string myvalue = std::string(XMLString::transcode(attrs.getValue(i)));

          if (myattname == "name")
	    name=myvalue;
	  if (myattname == "url")
	    url=myvalue;
          i++;
	}
      DCOUT('P', "file name = " << name << " and url = " << url);
      files_.push_back(name);
      urls_.push_back(url);
//       if (url[url.size() - 1] == '/')
// 	{
// 	  //	beingParsed->SetDDLFileName(url+name);
// 	  fileNames_.push_back(url+name);
// 	}
//       else
// 	{
// 	  //	beingParsed->SetDDLFileName(url+"/"+name);
// 	  fileNames_.push_back(url+"/"+name);
// 	}
    }
  else if (myelemname == "Root")
    {
      std::string fileName="", logicalPartName="";
      while ( i < numAtts )
	{
          std::string myattname = std::string(XMLString::transcode(attrs.getLocalName(i)));
          std::string myvalue = std::string(XMLString::transcode(attrs.getValue(i)));

	  if (myattname == "fileName")
	    fileName = myvalue;
	  if (myattname == "logicalPartName")
	    logicalPartName = myvalue;
	  i++;
	}

      fileName = fileName.substr(0, fileName.find("."));
      std::cout << fileName << ":" << logicalPartName << " is the ROOT" << std::endl;
      DDLogicalPart root;
      DDRootDef::instance().set(DDName(logicalPartName, fileName));

      DCOUT_V('P', std::string("DDLSAX2ConfigHandler::startElement.  Setting DDRoot LogicalPart=") + logicalPartName + std::string(" in ") + fileName);  

    }
  else if (myelemname == "Schema")
    {
      while ( i < numAtts )
	{
          std::string myattname = std::string(XMLString::transcode(attrs.getLocalName(i)));
          std::string myvalue = std::string(XMLString::transcode(attrs.getValue(i)));
	  if (myattname == "schemaLocation")
	    schemaLocation_ = myvalue;
          else if (myattname == "validation")
            doValidation_ = (myvalue == "true" ? true : false);
	  i++;
	}
    }
  DCOUT_V('P', "DDLSAX2ConfigHandler::startElement" << myelemname << " completed...");
}

std::vector<std::string>& DDLSAX2ConfigHandler::getFileNames()
{
  return files_;
}

std::vector<std::string>& DDLSAX2ConfigHandler::getURLs()
{
  return urls_;
}

std::string DDLSAX2ConfigHandler::getSchemaLocation()
{
  return schemaLocation_;
}

bool DDLSAX2ConfigHandler::doValidation()
{
  return doValidation_;
}
