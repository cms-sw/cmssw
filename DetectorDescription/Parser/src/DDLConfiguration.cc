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

//--------------------------------------------------------------------------
//  Includes
//--------------------------------------------------------------------------
// Parser parts
#include "DetectorDescription/Parser/interface/DDLConfiguration.h"
#include "DetectorDescription/Parser/interface/DDLParser.h"
#include "DetectorDescription/Base/interface/DDdebug.h"


// Xerces dependencies
#include <xercesc/sax2/XMLReaderFactory.hpp>
#include <xercesc/sax/SAXException.hpp>

#include <iostream>

//--------------------------------------------------------------------------
//  DDLConfiguration:  Default constructor and destructor.
//--------------------------------------------------------------------------
DDLConfiguration::~DDLConfiguration()
{
  //  parser_->getXMLParser()->setContentHandler(0);  
}

DDLConfiguration::DDLConfiguration(DDCompactView& cpv) : configHandler_( cpv ), cpv_(cpv)
{ 
  //  parser_ = DDLParser::instance();
  //  std::cout << "Making a DDLConfiguration with configHandler_ at " << &configHandler_ << std::endl;
}

DDLConfiguration::DDLConfiguration(DDLParser * ip, DDCompactView& cpv) : configHandler_( cpv ), cpv_(cpv)
{ 
  //  parser_ = ip; do NOTHING with the incomming pointer for now...
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
  for (std::vector<std::string>::const_iterator it = vst.begin(); it != vst.end(); ++it)
    std::cout << *it << std::endl;
}

//-----------------------------------------------------------------------
//  Here the Xerces parser is used to process the content of the 
//  configuration file.
//-----------------------------------------------------------------------
int DDLConfiguration::readConfig(const std::string& filename)
{
  DCOUT('P', "DetectorDescription/Parser/interface/DDLConfiguration::ReadConfig(): started");

  //  configFileName_ = filename;

  // Set the parser to use the handler for the configuration file.
  // This makes sure the Parser is initialized and gets a handle to it.
  // Set these to the flags for the configuration file.

  parser_->setContentHandler(&configHandler_);
  parser_->parse(filename.c_str());

  return 0;
}
