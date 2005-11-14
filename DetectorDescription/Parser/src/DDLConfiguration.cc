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
  //  parser_->getXMLParser()->setContentHandler(0);  
}

DDLConfiguration::DDLConfiguration() : configHandler_()
{ 
  parser_ = DDLParser::instance();
  //  std::cout << "Making a DDLConfiguration with configHandler_ at " << &configHandler_ << std::endl;
}

DDLConfiguration::DDLConfiguration(DDLParser * ip) : configHandler_()
{ 
  parser_ = ip;
}

const std::vector<std::string>&  DDLConfiguration::getFileList(void) const
{
  return configHandler_.getFileNames();
}

const std::vector<std::string>&  DDLConfiguration::getURLList(void) const
{
  return configHandler_.getURLs();
}

bool DDLConfiguration::doValidation() const { return configHandler_.doValidation(); }

std::string DDLConfiguration::getSchemaLocation() const { return configHandler_.getSchemaLocation(); }

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
//-----------------------------------------------------------------------
int DDLConfiguration::readConfig(const std::string& filename)
{
  DCOUT('P', "DDLConfiguration::ReadConfig(): started");

  //  configFileName_ = filename;

  // Set the parser to use the handler for the configuration file.
  // This makes sure the Parser is initialized and gets a handle to it.
  // Set these to the flags for the configuration file.
//   parser_->getXMLParser()->setFeature(XMLString::transcode("http://xml.org/sax/features/validation"),true);   // optional
//   parser_->getXMLParser()->setFeature(XMLString::transcode("http://xml.org/sax/features/namespaces"),true);   // optional
//   if (parser_->getXMLParser()->getFeature(XMLString::transcode("http://xml.org/sax/features/validation")) == true)
//     parser_->getXMLParser()->setFeature(XMLString::transcode("http://apache.org/xml/features/validation/dynamic"), true);

  parser_->getXMLParser()->setContentHandler(&configHandler_);

  try {
    parser_->getXMLParser()->parse(filename.c_str());
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

  //   std::vector<std::string> fnames = configHandler_.getFileNames();
  //   std::cout << "there are " << fnames.size() << " files." << std::endl;
  //   for (size_t i = 0; i < fnames.size(); i++)
  //     std::cout << "url=" << configHandler_.getURLs()[i] << " file=" << configHandler_.getFileNames()[i] << std::endl;
  return 0;
}
