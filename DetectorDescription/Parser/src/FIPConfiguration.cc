
/***************************************************************************
                          FIPConfiguration.cc  -  description
                             -------------------
    begin                : Sun Nov 13 2005
    email                : case@ucdhep.ucdavis.edu
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *           FIPConfiguration sub-component of DDD                                *
 *                                                                         *
 ***************************************************************************/

//--------------------------------------------------------------------------
//  Includes
//--------------------------------------------------------------------------
// Parser parts
#include "DetectorDescription/Parser/interface/FIPConfiguration.h"
#include "DetectorDescription/Parser/interface/DDLParser.h"
#include "DetectorDescription/Base/interface/DDdebug.h"


// Xerces dependencies
#include <xercesc/util/XercesDefs.hpp>
#include <xercesc/sax2/SAX2XMLReader.hpp>
#include <xercesc/sax2/XMLReaderFactory.hpp>
#include <xercesc/sax/SAXException.hpp>

// EDM Dependencies
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Utilities/interface/EDMException.h"

// STL
#include <iostream>

using namespace XERCES_CPP_NAMESPACE;

//--------------------------------------------------------------------------
//  FIPConfiguration:  Default constructor and destructor.
//--------------------------------------------------------------------------
FIPConfiguration::~FIPConfiguration()
{
  //  parser_->getXMLParser()->setContentHandler(0);  
}

FIPConfiguration::FIPConfiguration(DDCompactView& cpv) : configHandler_(cpv), cpv_(cpv)
{ 
  //  parser_ = DDLParser::instance();
  //  std::cout << "Making a FIPConfiguration with configHandler_ at " << &configHandler_ << std::endl;
}

const std::vector<std::string>&  FIPConfiguration::getFileList(void) const
{
  return files_;
}

const std::vector<std::string>&  FIPConfiguration::getURLList(void) const
{
  return urls_;
}

bool FIPConfiguration::doValidation() const { return configHandler_.doValidation(); }

std::string FIPConfiguration::getSchemaLocation() const { return configHandler_.getSchemaLocation(); }

void FIPConfiguration::dumpFileList(void) const {
  std::cout << "File List:" << std::endl;
  std::cout << "  number of files=" << files_.size() << std::endl;
  for (std::vector<std::string>::const_iterator it = files_.begin(); it != files_.end(); ++it)
    std::cout << *it << std::endl;
}

//-----------------------------------------------------------------------
//  Here the Xerces parser is used to process the content of the 
//  configuration file.
//-----------------------------------------------------------------------
int FIPConfiguration::readConfig(const std::string& filename)
{
  DCOUT('P', "FIPConfiguration::ReadConfig(): started");

  // Set the parser to use the handler for the configuration file.
  // This makes sure the Parser is initialized and gets a handle to it.
  DDLParser ddlp(cpv_);
  ddlp.getXMLParser()->setContentHandler(&configHandler_);
  edm::FileInPath fp(filename);
  // config file
  std::string absoluteFileName (filename);
  absoluteFileName = fp.fullPath();
  ddlp.getXMLParser()->parse(absoluteFileName.c_str());
  const std::vector<std::string>& vURLs = configHandler_.getURLs();
  const std::vector<std::string>& vFiles = configHandler_.getFileNames();
  size_t maxInd = vFiles.size();
  size_t ind = 0;
  // ea. file listed in the config
  for(; ind < maxInd ; ++ind) {
    edm::FileInPath fp(vURLs[ind] + "/" + vFiles[ind]);
    //    std::cout << "FileInPath says..." << fp.fullPath() << std::endl;
    files_.push_back(fp.fullPath());
    urls_.push_back("");
  }

  //   std::vector<std::string> fnames = configHandler_.getFileNames();
  //   std::cout << "there are " << fnames.size() << " files." << std::endl;
  //   for (size_t i = 0; i < fnames.size(); ++i)
  //     std::cout << "url=" << configHandler_.getURLs()[i] << " file=" << configHandler_.getFileNames()[i] << std::endl;
  return 0;
}
