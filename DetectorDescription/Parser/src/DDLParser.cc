/***************************************************************************
                          DDLParser.cc  -  description
                             -------------------
    begin                : Mon Oct 22 2001
    email                : case@ucdhep.ucdavis.edu
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *           DDDParser sub-component of DDD                                *
 *                                                                         *
 ***************************************************************************/

//--------------------------------------------------------------------------
//  Includes
//--------------------------------------------------------------------------
// Parser parts
#include "DetectorDescription/Parser/interface/DDLParser.h"
#include "DetectorDescription/Parser/interface/DDLDocumentProvider.h"
#include "DetectorDescription/Parser/interface/DDLConfiguration.h"
#include "DetectorDescription/Parser/interface/DDLSAX2FileHandler.h"
#include "DetectorDescription/Parser/interface/DDLSAX2ConfigHandler.h"
#include "DetectorDescription/Parser/interface/DDLSAX2ExpressionHandler.h"
#include "DetectorDescription/Parser/interface/DDLElementRegistry.h"

// DDCore Dependencies
#include "DetectorDescription/Base/interface/DDdebug.h"
#include "DetectorDescription/Base/interface/DDException.h"
#include "DetectorDescription/Parser/interface/StrX.h"
#include "DetectorDescription/Algorithm/src/AlgoInit.h"

// Xerces dependencies
#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/sax2/SAX2XMLReader.hpp>
#include <xercesc/sax2/XMLReaderFactory.hpp>
#include <xercesc/sax/SAXException.hpp>

#include <string>
#include <iostream>
#include <map>

#include "SealUtil/SealTimer.h"

namespace std{} using namespace std;

//--------------------------------------------------------------------------
//  DDLParser:  Default constructor and destructor.
//--------------------------------------------------------------------------
/// Destructor terminates the XMLPlatformUtils (as required by Xerces)
DDLParser::~DDLParser()
{ 

  // clean up and leave
  XMLPlatformUtils::Terminate();

  DCOUT_V('P', "DDLParser::~DDLParser(): destruct DDLParser"); 
}

/// Constructor initializes XMLPlatformUtils (as required by Xerces.
DDLParser::DDLParser(seal::Context* ) : nFiles_(0) //, handler_(0)
{ 
  // Initialize the XML4C2 system
  static seal::SealTimer tddlparser("DDLParser:Construct");

  try
    {
      XMLPlatformUtils::Initialize();
      AlgoInit();
    }

  catch (const XMLException& toCatch)
    {
      std::string e("\nDDLParser(): Error during initialization! Message:");
      e += std::string(StrX(toCatch.getMessage()).localForm()) + std::string ("\n");
      throw (DDException(e));
    }

  SAX2Parser_ = XMLReaderFactory::createXMLReader();

  // Set these to the flags for the configuration file.
  SAX2Parser_->setFeature(XMLString::transcode("http://xml.org/sax/features/validation"),true);   // optional
  SAX2Parser_->setFeature(XMLString::transcode("http://xml.org/sax/features/namespaces"),true);   // optional
  if (SAX2Parser_->getFeature(XMLString::transcode("http://xml.org/sax/features/validation")) == true)
    SAX2Parser_->setFeature(XMLString::transcode("http://apache.org/xml/features/validation/dynamic"), true);
  
  DCOUT_V('P', "DDLParser::DDLParser(): new (and only) DDLParser"); 
}

// ---------------------------------------------------------------------------
//  DDLSAX2Parser: Implementation of the DDLParser sitting on top of the Xerces
//  C++ XML Parser
//  
//--------------------------------------------------------------------------
// Initialize singleton pointer.
DDLParser* DDLParser::instance_ = 0;

// deprecated!
DDLParser* DDLParser::Instance()
{
  return instance();
}

// MEC: trying to keep COBRA stuff...
#ifdef USECOBRA
#include "Utilities/GenUtil/interface/TopLevelContext.h"
#include "Utilities/GenUtil/interface/GenericComponent.h"

// Fake singleton for backward compatibility

DDLParser* DDLParser::instance()
{
  if ( instance_ == 0 ) {
    // legacy sigleton
    // configure and build...
    seal::Context & theAppContext = frappe::topLevelContext();
    frappe::Configurator config(theAppContext);
    config.configure<DDLParserI>(frappe::Key<DDLParser>::name());
    instance_ = dynamic_cast<DDLParser*>(&config.component<DDLParserI>());
  }
  return instance_;
}
#endif

#ifndef USECOBRA
DDLParser* DDLParser::instance()
{
  if ( instance_ == 0 ) {
    // legacy sigleton
    // configure and build...
//     seal::Context & theAppContext = frappe::topLevelContext();
//     frappe::Configurator config(theAppContext);
//     config.configure<DDLParserI>(frappe::Key<DDLParser>::name());
//     instance_ = dynamic_cast<DDLParser*>(&config.component<DDLParserI>());
    instance_ = new DDLParser();
  }
  return instance_;
}
#endif

// MEC: EDMProto  temp?
void DDLParser::setInstance( DDLParser* p ) {
  if ( instance_ == 0 ) {
    instance_ = p;
  } else {
    std::cout << "DDL Parser instance already set!!" << std::endl;
  }
}

/**  This method allows external "users" to use the current DDLParser on their own.
 *   by giving them access to the SAX2XMLReader.  This may not be a good idea!  The
 *   reason that I 
 */ 
SAX2XMLReader* DDLParser::getXMLParser() { return SAX2Parser_; }

DDLSAX2FileHandler* DDLParser::getDDLSAX2FileHandler() { return dynamic_cast < DDLSAX2FileHandler* > (SAX2Parser_->getContentHandler()); }

// Set the filename for the config file.
// Remove this soon!
int DDLParser::SetConfig (const std::string& filename)
{
  std::cout << "DDLParser::SetConfig IS DEPRECATED.  Please check interface." << std::endl;
  configFileName_ = filename;
  return 0;
}

size_t DDLParser::isFound(const std::string& filename)
{
  FileNameHolder::const_iterator it = fileNames_.begin();
  size_t i = 0;
  bool foundFile = false;
  while (it != fileNames_.end() && !foundFile) //  for (; it != fileNames_.end(); it++) 
    {
      if (it->second.first == filename)
	{
	  foundFile = true;
	}
      else i++;
      it++;
    }
  if (foundFile)
    return i;
  return 0;
}

bool DDLParser::isParsed(const std::string& filename)
{
  size_t found = isFound(filename);
  if (found)
    return parsed_[found];
  return false;
}

bool DDLParser::parseOneFile(const std::string& filename, const std::string& url)
{
  size_t foundFile = isFound(filename);

  if (!foundFile || !parsed_[foundFile])
    {

      int fIndex = foundFile;
      //std::cout << "fIndex= " << fIndex << std::endl;
      if (!foundFile) //parsed_[foundFile])
	{
	  seal::SealTimer tddlfname("DDLParser:"+filename.substr(filename.rfind('/')+1));
	  pair <std::string, std::string> pss;
	  pss.first = filename;
	  if (url.size() && url.substr(url.size() - 1, 1) == "/")
	    pss.second = url+filename;
	  else
	    pss.second = url+ "/" + filename;
	  fIndex = nFiles_++;
	  fileNames_[fIndex] = pss;
	  parsed_[fIndex] = false;
	}
      //std::cout << "fIndex= " << fIndex << std::endl;
      DDLSAX2ExpressionHandler* myExpHandler(0);
      currFileName_ = fileNames_[fIndex].second;
      try
	{
	  myExpHandler = new DDLSAX2ExpressionHandler;
	  SAX2Parser_->setContentHandler(myExpHandler);
	  std::cout << "Parsing: " << fileNames_[fIndex].second << std::endl;
	  parseFile ( fIndex );

	  delete myExpHandler;
	}
      catch (const XMLException& toCatch) {
	std::cout << "\nDDLParser::ParseOneFile, PASS1: XMLException while processing files... \n"
	     << "Exception message is: \n"
	     << StrX(toCatch.getMessage()) << "\n" ;
	if (myExpHandler) delete myExpHandler;
	XMLPlatformUtils::Terminate();
	throw (DDException("  See XMLException above. "));
      }
    
      // PASS 2:

      DCOUT_V('P', "DDLParser::ParseOneFile(): PASS2: Just before setting Xerces content and error handlers... ");

      DDLSAX2FileHandler* myFileHandler(0);
      try
	{ 
	  seal::SealTimer t("DDLParser2:"+filename.substr(filename.rfind('/')+1));

	  myFileHandler = new DDLSAX2FileHandler;
	  SAX2Parser_->setContentHandler(myFileHandler);

	  // No need to validate (regardless of user's doValidation
	  // because the files have already been validated on the first pass.
	  // This optimization suggested by Martin Listd::endl.
	  SAX2Parser_->setFeature(XMLString::transcode("http://xml.org/sax/features/validation"), false);   // optional
	  SAX2Parser_->setFeature(XMLString::transcode("http://xml.org/sax/features/namespaces"), false);   // optional
	  SAX2Parser_->setFeature(XMLString::transcode("http://apache.org/xml/features/validation/dynamic"), false);

	  parseFile ( fIndex );
	  parsed_[fIndex] = true;
	  
	  delete myFileHandler;
	}
      catch (const XMLException& toCatch) {
	std::cout << "\nDDLParser::ParseOneFile, PASS2: XMLException while processing files... \n"
	     << "Exception message is: \n"
	     << StrX(toCatch.getMessage()) << "\n" ;
	if (myFileHandler) delete myFileHandler;
	XMLPlatformUtils::Terminate();
	throw (DDException("  See XMLException above."));
      }
    }
  else // was found and is parsed...
    {
      DCOUT('P', "\nWARNING: DDLParser::ParseOneFile() file " + filename 
	   + " was already parsed as " + fileNames_[foundFile].second);
      return true;
    }
  return false;
}

std::vector < std::string >  DDLParser::getFileList(void) 
{
  //  std::cout << "start getFileList" << std::endl;
  std::vector<std::string> flist;
  for (FileNameHolder::const_iterator fit = fileNames_.begin(); fit != fileNames_.end(); fit++)
    {
      //      std::cout << "about to push_back " << std::endl;
      //      std::cout << "fit->second.second = " << fit->second.second << std::endl;
      //      std::cout << "fit->second.first = " << fit->second.first << std::endl;
      flist.push_back(fit->second.first); // was .second (mec: 2003:02:19
    }
  //  std::cout << "about to return the list" << std::endl;
  return flist;
}

void DDLParser::dumpFileList(void) {
  dumpFileList(std::cout);
}

void DDLParser::dumpFileList(ostream& co) {
  co << "File List:" << std::endl;
  for (FileNameHolder::const_iterator it = fileNames_.begin(); it != fileNames_.end(); it++)
    co << it->second.second << std::endl;
}

// Deprecated. One implication is that I may not NEED configFileName_ and should not 
// keep it around any how, once this is gone!
int DDLParser::StartParsing()
{
  std::string tconfig = configFileName_;
  if (tconfig.size() == 0)
    {
      DDException e(std::string("DDLParser::StartParsing() can not be called without first DDLParser::SetConfig(...)"));
      throw(e);
    }
  else
    {
      DDLConfiguration tdp;
      /// FIX after removing this deprecated function, fix DDLConfiguration to not use int and fail flag.
      int failflag = tdp.readConfig(tconfig);
      if (!failflag)
	{
	  failflag = parse( tdp );
	}
      return failflag;
    }
  return 1; // control should never reach here
}

int DDLParser::parse(const DDLDocumentProvider& dp)
{
  std::cout << "Start Parsing.  Validation is set to " << dp.doValidation() << "." << std::endl;
  // prep for pass 1 through DDD XML
  DDLSAX2Handler* errHandler = new DDLSAX2Handler;
  SAX2Parser_->setErrorHandler(errHandler);
  //  std::cout << "after setErrorHandler)" << std::endl;
  if (dp.doValidation())
    { 
      seal::SealTimer t("DDLParser:Validation");

      std::string tval=dp.getSchemaLocation();
      const char* tch = tval.c_str();
      SAX2Parser_->setProperty(XMLString::transcode("http://apache.org/xml/properties/schema/external-schemaLocation"),XMLString::transcode(tch));
      SAX2Parser_->setFeature(XMLString::transcode("http://xml.org/sax/features/validation"), true);   // optional
      SAX2Parser_->setFeature(XMLString::transcode("http://xml.org/sax/features/namespaces"), true);   // optional
      SAX2Parser_->setFeature(XMLString::transcode("http://apache.org/xml/features/validation/dynamic"), true);
    }
  else
    {
      seal::SealTimer t("DDLParser:NoValidation");
      SAX2Parser_->setFeature(XMLString::transcode("http://xml.org/sax/features/validation"), false);   // optional
      SAX2Parser_->setFeature(XMLString::transcode("http://xml.org/sax/features/namespaces"), false);   // optional
      SAX2Parser_->setFeature(XMLString::transcode("http://apache.org/xml/features/validation/dynamic"), false);
    }

  //  This need be only done once, so might as well to it here.
  DDLSAX2ExpressionHandler* myExpHandler = new DDLSAX2ExpressionHandler;
  size_t fileIndex = 0;
  std::vector<std::string> fullFileName;

  for (; fileIndex < (dp.getFileList()).size(); fileIndex++)
    { 
      std::string ts = dp.getURLList()[fileIndex];
      std::string tf = dp.getFileList()[fileIndex];
      if (ts[ts.size() - 1] == '/')
	fullFileName.push_back( ts + tf );
      else
	fullFileName.push_back( ts + "/" + tf );
      //      std::cout << "full file name" << fullFileName[fileIndex] << std::endl;
    }

    seal::SealTimer * t = new seal::SealTimer("DDLParser1");

    for (std::vector<std::string>::const_iterator fnit = fullFileName.begin(); 
	 fnit != fullFileName.end();
	 fnit++)
      {
	size_t foundFile = isFound(myExpHandler->extractFileName( *fnit )); 
	
	//       FileNameHolder::const_iterator it = fileNames_.begin();
	//       size_t i = 0;      
	//       while (it != fileNames_.end() && !foundFile)
	// 	{
	// 	  if (it->second.first == myExpHandler->extractFileName( *fnit ))
	// 	    {
	// 	      foundFile = true;
	// 	    }
	// 	  else i++;
	// 	  it++;
	// 	}
	if (!foundFile)
	  {
	    pair <std::string, std::string> pss;
	    pss.first = myExpHandler->extractFileName( *fnit );
	    pss.second = *fnit;
	    fileNames_[nFiles_++] = pss;
	    parsed_[nFiles_ - 1]=false;
	  }
	//       else if (parsed_[foundFile])
	// 	{
	// 	  std::cout << "INFO:  DDLParser::Parse File " << *fnit << " was already processed as " 
	// 	       << fileNames_[foundFile].second << std::endl;
	// 	}
      }
    
  // Start processing the files found in the config file.
  
  // PASS 1:  This was added later (historically) to implement the DDD
  // requirement for Expressions.
  DCOUT('P', "DDLParser::ParseAllFiles(): PASS1: Just before setting Xerces content and error handlers... ");
  
  
  // Because we know we will be parsing a bunch of the same files, let us set the "reuse grammar flag"
  // THIS DID NOT DO WHAT I WANTED AND EXPECTED.  I GOT VERY STRANGE ERRORS!
  // if (SAX2Parser_->getFeature(XMLString::transcode("http://xml.org/sax/features/validation")))
  //    SAX2Parser_->setFeature(XMLString::transcode("http://apache.org/xml/features/validation/reuse-grammar"), true);

  try
    {
      SAX2Parser_->setContentHandler(myExpHandler);
      //std::cout << "1st PASS: Parsing " << fileNames_.size() << " files." << std::endl;
      //std::vector<std::string>::const_iterator currFile = fileNames_.begin(); currFile != fileNames_.end(); currFile++)
      for (size_t i = 0; i < fileNames_.size(); i++)
	{
	  if (!parsed_[i])
	    {
	      //std::cout << "Parsing: " << fileNames_[i].second << std::endl;
	      parseFile(i);
	    }
	}
      myExpHandler->dumpElementTypeCounter();
    }
  catch (const XMLException& toCatch) {
    std::cout << "\nPASS1: XMLException while processing files... \n"
	 << "Exception message is: \n"
	 << StrX(toCatch.getMessage()) << "\n" ;
    if (myExpHandler) delete myExpHandler;
    XMLPlatformUtils::Terminate();
    // FIX use this after DEPRECATED stuff removed:    throw(DDException("See XML Exception above"));
    return -1;
  }
  catch (DDException& e) {
    std::cout << "unexpected: " << std::endl;
    std::cout << e << std::endl;
    if (myExpHandler) delete myExpHandler;
    // FIX use this after DEPRECATED stuff removed    throw(e);
    return 4;
  }
  delete myExpHandler;
  delete t;
  // PASS 2:

  DCOUT('P', "DDLParser::ParseAllFiles(): PASS2: Just before setting Xerces content and error handlers... ");
  t = new seal::SealTimer("DDLParser2");
  DDLSAX2FileHandler* myFileHandler(0);
  try
    {
      myFileHandler = new DDLSAX2FileHandler;
      SAX2Parser_->setContentHandler(myFileHandler);
      // *T no need to re-do, all the same!


      // No need to validate (regardless of user's doValidation
      // because the files have already been validated on the first pass.
      // This optimization suggested by Martin Listd::endl.
      SAX2Parser_->setFeature(XMLString::transcode("http://xml.org/sax/features/validation"), false);   // optional
      SAX2Parser_->setFeature(XMLString::transcode("http://xml.org/sax/features/namespaces"), false);   // optional
      SAX2Parser_->setFeature(XMLString::transcode("http://apache.org/xml/features/validation/dynamic"), false);

      // Process files again.
      //std::cout << "Parsing " << fileNames_.size() << " files." << std::endl;
      for (size_t i = 0; i < fileNames_.size(); i++)
	//std::vector<std::string>::const_iterator currFile = fileNames_.begin(); currFile != fileNames_.end(); currFile++)
	{
	  parseFile(i);
	  parsed_[i] = true;
	  pair<std::string, std::string> namePair = fileNames_[i];
	  std::cout << "Completed parsing file " << namePair.second << "." << std::endl;
	}
      //myFileHandler->dumpElementTypeCounter();

      delete myFileHandler;
    }
  catch (DDException& e) {
    std::string s(e.what());
    s+="\n\t see above:  DDLParser::parse  Exception " ;
    std::cout << s << std::endl;
    if (myFileHandler) delete myFileHandler;
    // remove/fix after DEPRECATED stuff...
    delete errHandler;
    return -1;
    //    throw(e);
  }
  catch (const XMLException& toCatch) {
    std::cout << "\nPASS2: XMLException while processing files... \n"
	 << "Exception message is: \n"
	 << StrX(toCatch.getMessage()) << "\n" ;
    if (myFileHandler) delete myFileHandler;
    XMLPlatformUtils::Terminate();
    // FIX - use this after DEPRECATED stuff removed:    throw(DDException("See XMLException above"));
    delete errHandler;
    return -1;
  }
  delete errHandler;
  delete t;
  // FIX remove after DEPRECATED stuff removed
  return 0;
}

void DDLParser::parseFile(const int& numtoproc) 
{
  
  if (!parsed_[numtoproc])
    {
      const std::string & fname = fileNames_[numtoproc].second;
      seal::SealTimer t("DDLParser:"+fname.substr(fname.rfind('/')+1));

      try
	{
	  currFileName_ = fname;
	  SAX2Parser_->parse(currFileName_.c_str());
	}
      catch (const XMLException& toCatch)
	{
	  std::string e("\nWARNING: DDLParser::parseFile, File: '");
	  e += currFileName_ + "'\n"
	    + "Exception message is: \n"
	    + std::string(StrX(toCatch.getMessage()).localForm()) + "\n";
	  throw(DDException(e));
	}
      catch (DDException& e)
	{
	  std::string s(e.what());
	  s += "\nERROR: Unexpected exception during parsing: '"
	    + currFileName_ + "'\n"
	    + "Exception message is shown above.\n";
	  throw(DDException(e));
	}
    }
  else
    {
// 	DDException e("\nWARNING: File ");
// 	e+= fileNames_[numtoproc].first 
// 	  + " has already been processed as " 
// 	  + fileNames_[numtoproc].second 
// 	  + "\n";
// 	throw(e);
      DCOUT('P', "\nWARNING: File " + fileNames_[numtoproc].first 
	   + " has already been processed as " + fileNames_[numtoproc].second);
    }
}

// Return the name of the Current file being processed by the parser.
std::string DDLParser::getCurrFileName()
{
  return currFileName_;
}

#include "DetectorDescription/Parser/interface/DDLConfiguration.h"
// to make client independent of the implementation
DDLDocumentProvider * DDLParser::newConfig() const {
  return new  DDLConfiguration(const_cast<DDLParser*>(this));
} 
