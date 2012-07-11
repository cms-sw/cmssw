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

#include "DetectorDescription/Parser/interface/DDLParser.h"
#include "DetectorDescription/Parser/interface/DDLDocumentProvider.h"

#include "DetectorDescription/Base/interface/DDdebug.h"
#include "DetectorDescription/Algorithm/src/AlgoInit.h"

#include <xercesc/framework/MemBufInputSource.hpp>
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include <iostream>

using namespace std;

using namespace XERCES_CPP_NAMESPACE;

/// Constructor MUST associate a DDCompactView storage.
DDLParser::DDLParser( DDCompactView& cpv )
  : cpv_( cpv ),
    nFiles_( 0 )
{
  XMLPlatformUtils::Initialize();
  AlgoInit();
  SAX2Parser_  = XMLReaderFactory::createXMLReader();
  
  SAX2Parser_->setFeature(XMLUni::fgSAX2CoreValidation, false);   // optional
  SAX2Parser_->setFeature(XMLUni::fgSAX2CoreNameSpaces, false);   // optional
  // Specify other parser features, e.g.
  //  SAX2Parser_->setFeature(XMLUni::fgXercesSchemaFullChecking, false);
  
  expHandler_  = new DDLSAX2ExpressionHandler(cpv);
  fileHandler_ = new DDLSAX2FileHandler(cpv);
  errHandler_  = new DDLSAX2Handler();
  SAX2Parser_->setErrorHandler(errHandler_); 
  SAX2Parser_->setContentHandler(fileHandler_); 
  
  DCOUT_V('P', "DDLParser::DDLParser(): new (and only) DDLParser"); 
}

/// Destructor terminates the XMLPlatformUtils (as required by Xerces)
DDLParser::~DDLParser( void )
{ 
  // clean up and leave
  delete expHandler_;
  delete fileHandler_;
  delete errHandler_;
  XMLPlatformUtils::Terminate();
  DCOUT_V('P', "DDLParser::~DDLParser(): destruct DDLParser"); 
}

/**  This method allows external "users" to use the current DDLParser on their own.
 *   by giving them access to the SAX2XMLReader.  This may not be a good idea!  The
 *   reason that I 
 */ 
SAX2XMLReader*
DDLParser::getXMLParser( void )
{
  return SAX2Parser_;
}

DDLSAX2FileHandler*
DDLParser::getDDLSAX2FileHandler( void )
{ 
  return fileHandler_; 
}

size_t
DDLParser::isFound( const std::string& filename )
{
  FileNameHolder::const_iterator it = fileNames_.begin();
  size_t i = 1;
  bool foundFile = false;
  while( it != fileNames_.end() && !foundFile )
  {
    if( it->second.first == filename )
    {
      foundFile = true;
    }
    else ++i;
    ++it;
  }
  if( foundFile )
    return i;
  return 0;
}

bool
DDLParser::isParsed( const std::string& filename )
{
  size_t found = isFound(filename);
  if (found)
    return parsed_[found];
  return false;
}

// Must receive a filename and path relative to the src directory of a CMSSW release
// e.g. DetectorDescription/test/myfile.xml
bool
DDLParser::parseOneFile( const std::string& fullname ) //, const std::string& url)
{
  //  std::string filename = expHandler_->extractFileName(fullname);
  std::string filename = extractFileName(fullname);
  //  std::cout << "parseOneFile - fullname = " << fullname << std::endl;
  //  std::cout << "parseOneFile - filename = " << filename << std::endl;
  edm::FileInPath fp(fullname);
  std::string absoluteFileName = fp.fullPath();
  size_t foundFile = isFound(filename);
  if (!foundFile)
  {
    pair <std::string, std::string> pss;
    pss.first = filename;
    pss.second = absoluteFileName; //url+filename;
    int fIndex = nFiles_;
    fileNames_[nFiles_] = pss;
    ++nFiles_;
    parsed_[fIndex]=false;

    currFileName_ = fileNames_[fIndex].second;

    // in cleaning up try-catch blocks 2007-06-26 I decided to remove
    // this because of CMSSW rules. but keep the commented way I used to
    // do it...
    //       DO NOT UNCOMMENT FOR ANY RELEASE; ONLY FOR DEBUGGING! try
    //         {
    SAX2Parser_->setContentHandler(expHandler_);
    expHandler_->setNameSpace( getNameSpace(filename) );
    //	  std::cout << "0) namespace = " << getNameSpace(filename) << std::endl;
    LogDebug ("DDLParser") << "ParseOneFile() Parsing: " << fileNames_[fIndex].second << std::endl;
    parseFile ( fIndex );

    //         }
    //       DO NOT UNCOMMENT FOR ANY RELEASE; ONLY FOR DEBUGGING! catch (const XMLException& toCatch) {
    //         edm::LogError ("DDLParser") << "\nDDLParser::ParseOneFile, PASS1: XMLException while processing files... \n"
    //              << "Exception message is: \n"
    //              << StrX(toCatch.getMessage()) << "\n" ;
    //         XMLPlatformUtils::Terminate();
    //         throw (DDException("  See XMLException above. "));
    //       }

    // PASS 2:

    DCOUT_V('P', "DDLParser::ParseOneFile(): PASS2: Just before setting Xerces content and error handlers... ");

    //       DO NOT UNCOMMENT FOR ANY RELEASE; ONLY FOR DEBUGGING! try
    //         {

    SAX2Parser_->setContentHandler(fileHandler_);
    //      std::cout << "currFileName = " << currFileName_ << std::endl;
    fileHandler_->setNameSpace( getNameSpace(extractFileName(currFileName_)) );
    //      std::cout << "1)  namespace = " << getNameSpace(currFileName_) << std::endl;
    parseFile ( fIndex );
    parsed_[fIndex] = true;

    //         }
    //       DO NOT UNCOMMENT FOR ANY RELEASE; ONLY FOR DEBUGGING! catch (const XMLException& toCatch) {
    //         edm::LogError ("DDLParser") << "\nDDLParser::ParseOneFile, PASS2: XMLException while processing files... \n"
    //              << "Exception message is: \n"
    //              << StrX(toCatch.getMessage()) << "\n" ;
    //         XMLPlatformUtils::Terminate();
    //         throw (DDException("  See XMLException above."));
    //       }
  }
  else // was found and is parsed...
  {
    DCOUT('P', " WARNING: DDLParser::ParseOneFile() file " + filename
	  + " was already parsed as " + fileNames_[foundFile].second);
    return true;
  }
  return false;
}

//  This is for parsing the content of a blob stored in the conditions system of CMS.
void
DDLParser::parse( const std::vector<unsigned char>& ablob, unsigned int bsize )
{
  char* dummy(0);
  MemBufInputSource  mbis( &*ablob.begin(), bsize, dummy );
  SAX2Parser_->parse(mbis);
}

std::vector < std::string >
DDLParser::getFileList( void ) 
{
  std::vector<std::string> flist;
  for (FileNameHolder::const_iterator fit = fileNames_.begin(); fit != fileNames_.end(); ++fit)
  {
    flist.push_back(fit->second.first); // was .second (mec: 2003:02:19
  }
  return flist;
}

void
DDLParser::dumpFileList( void )
{
  edm::LogInfo ("DDLParser") << "File List:" << std::endl;
  for (FileNameHolder::const_iterator it = fileNames_.begin(); it != fileNames_.end(); ++it)
    edm::LogInfo ("DDLParser") << it->second.second << std::endl;
}

void
DDLParser::dumpFileList( ostream& co )
{
  co << "File List:" << std::endl;
  for (FileNameHolder::const_iterator it = fileNames_.begin(); it != fileNames_.end(); ++it)
    co << it->second.second << std::endl;
}

int
DDLParser::parse( const DDLDocumentProvider& dp )
{
  //  edm::LogInfo ("DDLParser") << "Start Parsing.  Validation is set to " << dp.doValidation() << "." << std::endl;
  edm::LogInfo ("DDLParser") << "Start Parsing.  Validation is set off for the time being." << std::endl;
  // prep for pass 1 through DDD XML
  //   // Since this block does nothing for CMSSW right now, I have taken it all out
  //   This clean-up involves interface changes such as the removal of doValidation() everywhere (OR NOT 
  //   if I decide to keep it for other testing reasons.)
  //    if (dp.doValidation())
  //      { 
  // //       DCOUT_V('P', "WARNING:  PARSER VALIDATION IS TURNED OFF REGARDLESS OF <Schema... ELEMENT");
  //   SAX2Parser_->setFeature(XMLUni::fgSAX2CoreValidation, true);
  //   SAX2Parser_->setFeature(XMLUni::fgSAX2CoreNameSpaces, true);
  // //       //  SAX2Parser_->setFeature(XMLUni::fgXercesSchemaFullChecking, true);
  //      }
  //    else
  //     {
  SAX2Parser_->setFeature(XMLUni::fgSAX2CoreValidation, false);
  SAX2Parser_->setFeature(XMLUni::fgSAX2CoreNameSpaces, false);
  //       //  SAX2Parser_->setFeature(XMLUni::fgXercesSchemaFullChecking, false);

  //     }

  //  This need be only done once, so might as well to it here.
  size_t fileIndex = 0;
  std::vector<std::string> fullFileName;

  for (; fileIndex < (dp.getFileList()).size(); ++fileIndex)
  { 
    std::string ts = dp.getURLList()[fileIndex];
    std::string tf = dp.getFileList()[fileIndex];
    if ( ts.size() > 0 ) {
      if ( ts[ts.size() - 1] == '/') {
	fullFileName.push_back( ts + tf );
      } else {
	fullFileName.push_back( ts + "/" + tf );
      }
    } else {
      fullFileName.push_back( tf );
    }
  }

  for (std::vector<std::string>::const_iterator fnit = fullFileName.begin(); 
       fnit != fullFileName.end();
       ++fnit)
  {
    size_t foundFile = isFound(extractFileName( *fnit )); 
	
    if (!foundFile)
    {
      pair <std::string, std::string> pss;
      pss.first = extractFileName( *fnit );
      pss.second = *fnit;
      fileNames_[nFiles_++] = pss;
      parsed_[nFiles_ - 1]=false;
    }
  }
    
  // Start processing the files found in the config file.
  
  // PASS 1:  This was added later (historically) to implement the DDD
  // requirement for Expressions.
  DCOUT('P', "DDLParser::parse(): PASS1: Just before setting Xerces content and error handlers... ");
  
  
  // in cleaning up try-catch blocks 2007-06-26 I decided to remove
  // this because of CMSSW rules. but keep the commented way I used to
  // do it...
  //   DO NOT UNCOMMENT FOR ANY RELEASE; ONLY FOR DEBUGGING! try
  //     {
  SAX2Parser_->setContentHandler(expHandler_);
  for (size_t i = 0; i < fileNames_.size(); ++i)
  {
    // 	  seal::SealTimer t("DDLParser: parsing expressions of file " +fileNames_[i].first);
    if (!parsed_[i])
    {
      currFileName_ = fileNames_[i].second;
      //	      std::cout << "currFileName = " << currFileName_ << std::endl;
      expHandler_->setNameSpace( getNameSpace(extractFileName(currFileName_)) );
      //	      std::cout << "2)  namespace = " << getNameSpace(extractFileName(currFileName_)) << std::endl;
      parseFile(i);
    }
  }
  expHandler_->dumpElementTypeCounter();
  //     }
  //   DO NOT UNCOMMENT FOR ANY RELEASE; ONLY FOR DEBUGGING! catch (const XMLException& toCatch) {
  //     edm::LogInfo ("DDLParser") << "\nPASS1: XMLException while processing files... \n"
  // 	 << "Exception message is: \n"
  // 	 << StrX(toCatch.getMessage()) << "\n" ;
  //     XMLPlatformUtils::Terminate();
  //     // FIX use this after DEPRECATED stuff removed:    throw(DDException("See XML Exception above"));
  //     return -1;
  //   }
  // PASS 2:

  DCOUT('P', "DDLParser::parse(): PASS2: Just before setting Xerces content and error handlers... ");

  // in cleaning up try-catch blocks 2007-06-26 I decided to remove
  // this because of CMSSW rules. but keep the commented way I used to
  // do it...
  //   DO NOT UNCOMMENT FOR ANY RELEASE; ONLY FOR DEBUGGING! try
  //     {
  SAX2Parser_->setContentHandler(fileHandler_);

  // No need to validate (regardless of user's doValidation
  // because the files have already been validated on the first pass.
  // This optimization suggested by Martin Liendl.
  //       SAX2Parser_->setFeature(StrX("http://xml.org/sax/features/validation"), false);   // optional
  //       SAX2Parser_->setFeature(StrX("http://xml.org/sax/features/namespaces"), false);   // optional
  //       SAX2Parser_->setFeature(StrX("http://apache.org/xml/features/validation/dynamic"), false);


  // Process files again.
  for (size_t i = 0; i < fileNames_.size(); ++i)
  {
    // 	  seal::SealTimer t("DDLParser: parsing all elements of file " +fileNames_[i].first);
    if (!parsed_[i]) {
      currFileName_ = fileNames_[i].second;
      //	    std::cout << "currFileName = " << currFileName_ << std::endl;
      fileHandler_->setNameSpace( getNameSpace(extractFileName(currFileName_)) );
      //	    std::cout << "3)  namespace = " << getNameSpace(extractFileName(currFileName_)) << std::endl;
      parseFile(i);
      parsed_[i] = true;
      pair<std::string, std::string> namePair = fileNames_[i];
      LogDebug ("DDLParser") << "Completed parsing file " << namePair.second << std::endl;
    }
  }
  //     }
  //   DO NOT UNCOMMENT FOR ANY RELEASE; ONLY FOR DEBUGGING! catch (const XMLException& toCatch) {
  //     edm::LogError ("DDLParser") << "\nPASS2: XMLException while processing files... \n"
  // 	 << "Exception message is: \n"
  // 	 << StrX(toCatch.getMessage()) << "\n" ;
  //     XMLPlatformUtils::Terminate();
  //     return -1;
  //   }
  return 0;
}

void
DDLParser::parseFile( const int& numtoproc ) 
{
  if (!parsed_[numtoproc])
  {
    const std::string & fname = fileNames_[numtoproc].second;

    // in cleaning up try-catch blocks 2007-06-26 I decided to remove
    // this because of CMSSW rules. but keep the commented way I used to
    // do it...
    //       DO NOT UNCOMMENT FOR ANY RELEASE; ONLY FOR DEBUGGING! try
    // 	{
    currFileName_ = fname;
    SAX2Parser_->parse(currFileName_.c_str());
    // 	}
    //       DO NOT UNCOMMENT FOR ANY RELEASE; ONLY FOR DEBUGGING! catch (const XMLException& toCatch)
    // 	{
    // 	  std::string e("\nWARNING: DDLParser::parseFile, File: '");
    // 	  e += currFileName_ + "'\n"
    // 	    + "Exception message is: \n"
    // 	    + std::string(StrX(toCatch.getMessage()).localForm()) + "\n";
    // 	  throw(DDException(e));
    // 	}
  }
  else
  {
    DCOUT('P', "\nWARNING: File " + fileNames_[numtoproc].first 
	  + " has already been processed as " + fileNames_[numtoproc].second);
  }
}

// Return the name of the Current file being processed by the parser.
std::string
DDLParser::getCurrFileName( void )
{
  return currFileName_;
}

void
DDLParser::clearFiles( void )
{
  fileNames_.clear();
  parsed_.clear();
}

std::string
DDLParser::extractFileName( std::string fullname )
{
  std::string ret = "";
  size_t bit = fullname.rfind('/');
  if ( bit < fullname.size() - 2 ) {
    ret=fullname.substr(bit+1);
  }
  return ret;
}

std::string
DDLParser::getNameSpace( const std::string& fname )
{
  size_t j = 0;
  std::string ret="";
  while (j < fname.size() && fname[j] != '.')
    ++j;
  if (j < fname.size() && fname[j] == '.')
    ret = fname.substr(0, j);
  return ret;
}
