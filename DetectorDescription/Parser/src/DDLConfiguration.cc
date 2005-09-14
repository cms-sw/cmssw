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
#include "DetectorDescription/Parser/interface/DDLConfiguration.h"
#include "DetectorDescription/Parser/interface/DDLParser.h"
#include "DetectorDescription/Parser/interface/DDLSAX2ConfigHandler.h"
#include "DetectorDescription/Parser/interface/StrX.h"
#include "DetectorDescription/Base/interface/DDdebug.h"
#include "DetectorDescription/Base/interface/DDException.h"


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
  //  std::cout << "made a DDLSAX2ConfigHandler at " << sch_ << std::endl;
  //  std::cout << "made a DDLSAX2Handler at " << errHandler_ << std::endl;
}

DDLConfiguration::DDLConfiguration(DDLParser * ip)
{ 
  m_parser = ip; //
  sch_ = new DDLSAX2ConfigHandler;
  errHandler_ = new DDLSAX2Handler;
  //  std::cout << "made a DDLSAX2ConfigHandler at " << sch_ << std::endl;
  //  std::cout << "made a DDLSAX2Handler at " << errHandler_ << std::endl;
}

const std::vector<std::string>&  DDLConfiguration::getFileList(void) const
{
  return sch_->getFileNames();
}

const std::vector<std::string>&  DDLConfiguration::getURLList(void) const
{
  return sch_->getURLs();
}

bool DDLConfiguration::doValidation() const { return sch_->doValidation(); }

std::string DDLConfiguration::getSchemaLocation() const { return sch_->getSchemaLocation(); }

void DDLConfiguration::dumpFileList(void) const {
  std::cout << "File List:" << std::endl;
  std::vector<std::string> vst = getFileList();  // why do I need to do this?
  std::cout << "  number of files=" << vst.size() << std::endl;
  for (std::vector<std::string>::const_iterator it = vst.begin(); it != vst.end(); it++)
    std::cout << *it << std::endl;
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
int DDLConfiguration::readConfig(const std::string& filename)
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
    std::cout << "\nXMLException: parsing '" << filename << "'\n"
	 << "Exception message is: \n"
	 << std::string(StrX(toCatch.getMessage()).localForm()) << "\n" ;
    return -1;
  }
  catch (...)
    {
      std::cout << "\nUnexpected exception during parsing: '" << filename << "'\n";
      return 4;
    }

//   std::vector<std::string> fnames = sch_->getFileNames();
//   std::cout << "there are " << fnames.size() << " files." << std::endl;
//   for (size_t i = 0; i < fnames.size(); i++)
//     std::cout << "url=" << sch_->getURLs()[i] << " file=" << sch_->getFileNames()[i] << std::endl;
  return 0;
}
