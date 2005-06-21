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
#include "DetectorDescription/DDParser/interface/DDLParser.h"
#include "DetectorDescription/DDParser/interface/DDLSAX2ConfigHandler.h"

// Xerces C++ dependencies.
#include <xercesc/sax2/Attributes.hpp>
#include <xercesc/sax/SAXParseException.hpp>
#include <xercesc/sax/SAXException.hpp>

// DDCore dependencies
#include "DetectorDescription/DDBase/interface/DDdebug.h"
//  This is frustrating.  I only need DDInit at this level to get a handle on DDRootDef.
//#include "DetectorDescription/DDCore/interface/DDInit.h"
#include "DetectorDescription/DDCore/interface/DDRoot.h"
#include "DetectorDescription/DDCore/interface/DDLogicalPart.h"

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

  string myelemname = string(XMLString::transcode(qname));
  DCOUT_V('P', "DDLSAX2ConfigHandler::startElement" << myelemname << " started...");

  unsigned int numAtts = attrs.getLength();
  unsigned int i = 0;
  if (myelemname == "File")
    {
      string name="", url="";
      while ( i < numAtts )
	{
          string myattname = string(XMLString::transcode(attrs.getLocalName(i)));
          string myvalue = string(XMLString::transcode(attrs.getValue(i)));

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
      string fileName="", logicalPartName="";
      while ( i < numAtts )
	{
          string myattname = string(XMLString::transcode(attrs.getLocalName(i)));
          string myvalue = string(XMLString::transcode(attrs.getValue(i)));

	  if (myattname == "fileName")
	    fileName = myvalue;
	  if (myattname == "logicalPartName")
	    logicalPartName = myvalue;
	  i++;
	}

      fileName = fileName.substr(0, fileName.find("."));
      cout << fileName << ":" << logicalPartName << " is the ROOT" << endl;
      DDLogicalPart root;
      DDRootDef::instance().set(DDName(logicalPartName, fileName));

      DCOUT_V('P', string("DDLSAX2ConfigHandler::startElement.  Setting DDRoot LogicalPart=") + logicalPartName + string(" in ") + fileName);  

    }
  else if (myelemname == "Schema")
    {
      while ( i < numAtts )
	{
          string myattname = string(XMLString::transcode(attrs.getLocalName(i)));
          string myvalue = string(XMLString::transcode(attrs.getValue(i)));
	  if (myattname == "schemaLocation")
	    schemaLocation_ = myvalue;
          else if (myattname == "validation")
            doValidation_ = (myvalue == "true" ? true : false);
	  i++;
	}
    }
  DCOUT_V('P', "DDLSAX2ConfigHandler::startElement" << myelemname << " completed...");
}

vector<string>& DDLSAX2ConfigHandler::getFileNames()
{
  return files_;
}

vector<string>& DDLSAX2ConfigHandler::getURLs()
{
  return urls_;
}

string DDLSAX2ConfigHandler::getSchemaLocation()
{
  return schemaLocation_;
}

bool DDLSAX2ConfigHandler::doValidation()
{
  return doValidation_;
}
