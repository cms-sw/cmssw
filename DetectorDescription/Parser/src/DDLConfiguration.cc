/***************************************************************************
                          DDLConfiguration.cc  -  description
                             -------------------
    begin                : Mon Feb 24 2003
    email                : case@ucdhep.ucdavis.edu
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *           DDLConfiguration sub-component of DDD                                *
 *                                                                         *
 ***************************************************************************/
//using namespace xercesc_2_3;
//--------------------------------------------------------------------------
//  Includes
//--------------------------------------------------------------------------
// Parser parts
#include "DetectorDescription/Parser/interface/DDLParser.h"
#include "DetectorDescription/Parser/interface/DDLConfiguration.h"
#include "DetectorDescription/Parser/interface/DDLSAX2ConfigHandler.h"

// DDCore Dependencies
#include "DetectorDescription/Base/interface/DDdebug.h"
#include "DetectorDescription/Base/interface/DDException.h"
#include "DetectorDescription/Parser/interface/StrX.h"

// Xerces dependencies
#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/sax2/SAX2XMLReader.hpp>
#include <xercesc/sax2/XMLReaderFactory.hpp>
#include <xercesc/sax/SAXException.hpp>

#include <string>
#include <iostream>
#include <map>

namespace std{} using namespace std;

//--------------------------------------------------------------------------
//  DDLConfiguration:  Default constructor and destructor.
//--------------------------------------------------------------------------
DDLConfiguration::~DDLConfiguration()
{ 
  delete sch_;
  delete errHandler_;
}

DDLConfiguration::DDLConfiguration()
{ 
  m_parser = DDLParser::instance(); // I just want to make sure Xerces gets initialized!
  sch_ = new DDLSAX2ConfigHandler;
  errHandler_ = new DDLSAX2Handler;
  //  cout << "made a DDLSAX2ConfigHandler at " << sch_ << endl;
  //  cout << "made a DDLSAX2Handler at " << errHandler_ << endl;
}

DDLConfiguration::DDLConfiguration(DDLParser * ip)
{ 
  m_parser = ip; //
  sch_ = new DDLSAX2ConfigHandler;
  errHandler_ = new DDLSAX2Handler;
  //  cout << "made a DDLSAX2ConfigHandler at " << sch_ << endl;
  //  cout << "made a DDLSAX2Handler at " << errHandler_ << endl;
}

const vector<string>&  DDLConfiguration::getFileList(void) const
{
  return sch_->getFileNames();
}

const vector<string>&  DDLConfiguration::getURLList(void) const
{
  return sch_->getURLs();
}

bool DDLConfiguration::doValidation() const { return sch_->doValidation(); }

string DDLConfiguration::getSchemaLocation() const { return sch_->getSchemaLocation(); }

void DDLConfiguration::dumpFileList(void) const {
  cout << "File List:" << endl;
  vector<string> vst = getFileList();  // why do I need to do this?
  cout << "  number of files=" << vst.size() << endl;
  for (vector<string>::const_iterator it = vst.begin(); it != vst.end(); it++)
    cout << *it << endl;
}

//-----------------------------------------------------------------------
//  Here the Xerces parser is used to process the content of the 
//  configuration file.
//  FIX:  Right now, each config file passed to this will simply increase the 
//  size of the list of files.  So if this default DDLDocumentProvider is
//  called repeatedly (i.e. the same instance of it) then the file list MAY
//  repeat files.  It is the Parser which checks for maintains a list of real
//  files.
//-----------------------------------------------------------------------
int DDLConfiguration::readConfig(const string& filename)
{
  DCOUT('P', "DDLConfiguration::ReadConfig(): started");

  //  configFileName_ = filename;

  // Set the parser to use the handler for the configuration file.
  // This makes sure the Parser is initialized and gets a handle to it.
  m_parser->getXMLParser()->setContentHandler(sch_);
  m_parser->getXMLParser()->setErrorHandler(errHandler_);

  try {
    m_parser->getXMLParser()->parse(filename.c_str());
  }
  catch (const XMLException& toCatch) {
    cout << "\nXMLException: parsing '" << filename << "'\n"
	 << "Exception message is: \n"
	 << string(StrX(toCatch.getMessage()).localForm()) << "\n" ;
    return -1;
  }
  catch (...)
    {
      cout << "\nUnexpected exception during parsing: '" << filename << "'\n";
      return 4;
    }

//   vector<string> fnames = sch_->getFileNames();
//   cout << "there are " << fnames.size() << " files." << endl;
//   for (size_t i = 0; i < fnames.size(); i++)
//     cout << "url=" << sch_->getURLs()[i] << " file=" << sch_->getFileNames()[i] << endl;
  return 0;
}
