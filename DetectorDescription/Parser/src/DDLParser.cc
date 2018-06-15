#include "DetectorDescription/Parser/interface/DDLParser.h"
#include "DetectorDescription/Parser/interface/DDLDocumentProvider.h"
#include "DetectorDescription/Parser/interface/DDLSAX2ExpressionHandler.h"
#include "DetectorDescription/Parser/interface/DDLSAX2FileHandler.h"
#include "DetectorDescription/Parser/interface/DDLSAX2Handler.h"
#include "FWCore/Concurrency/interface/Xerces.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include <xercesc/framework/MemBufInputSource.hpp>
#include <xercesc/sax2/XMLReaderFactory.hpp>
#include <xercesc/util/XMLUni.hpp>

#include <iostream>

class DDCompactView;

XERCES_CPP_NAMESPACE_USE

using namespace std;

/// Constructor MUST associate a DDCompactView storage.
DDLParser::DDLParser( DDCompactView& cpv )
  : cpv_( cpv ),
    nFiles_( 0 )
{
  cms::concurrency::xercesInitialize();
  SAX2Parser_  = XMLReaderFactory::createXMLReader();
  
  SAX2Parser_->setFeature(XMLUni::fgSAX2CoreValidation, false);   // optional
  SAX2Parser_->setFeature(XMLUni::fgSAX2CoreNameSpaces, false);   // optional

  elementRegistry_ = new DDLElementRegistry();
  expHandler_  = new DDLSAX2ExpressionHandler(cpv, *elementRegistry_);
  fileHandler_ = new DDLSAX2FileHandler(cpv, *elementRegistry_);
  errHandler_  = new DDLSAX2Handler();
  SAX2Parser_->setErrorHandler(errHandler_); 
  SAX2Parser_->setContentHandler(fileHandler_); 
}

/// Destructor terminates the XMLPlatformUtils (as required by Xerces)
DDLParser::~DDLParser( void )
{ 
  // clean up and leave
  delete expHandler_;
  delete fileHandler_;
  delete errHandler_;
  delete elementRegistry_;
  cms::concurrency::xercesTerminate();
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
DDLParser::parseOneFile( const std::string& fullname )
{
  std::string filename = extractFileName( fullname );
  edm::FileInPath fp( fullname );
  std::string absoluteFileName = fp.fullPath();
  size_t foundFile = isFound( filename );
  if( !foundFile )
  {
    pair< std::string, std::string > pss;
    pss.first = filename;
    pss.second = absoluteFileName; //url+filename;
    int fIndex = nFiles_;
    fileNames_[nFiles_] = pss;
    ++nFiles_;
    parsed_[fIndex] = false;

    currFileName_ = fileNames_[fIndex].second;

    SAX2Parser_->setContentHandler( expHandler_ );
    expHandler_->setNameSpace( getNameSpace( filename ));

    LogDebug ("DDLParser") << "ParseOneFile() Parsing: " << fileNames_[fIndex].second << std::endl;
    parseFile( fIndex );
    expHandler_->createDDConstants();
    // PASS 2:

    SAX2Parser_->setContentHandler(fileHandler_);
    fileHandler_->setNameSpace( getNameSpace( extractFileName( currFileName_ )));
    parseFile ( fIndex );
    parsed_[fIndex] = true;
  }
  else // was found and is parsed...
  {
    return true;
  }
  return false;
}

//  This is for parsing the content of a blob stored in the conditions system of CMS.
void
DDLParser::parse( const std::vector<unsigned char>& ablob, unsigned int bsize )
{
  char* dummy(nullptr);
  MemBufInputSource  mbis( &*ablob.begin(), bsize, dummy );
  SAX2Parser_->parse(mbis);
  expHandler_->createDDConstants();
  
}

int
DDLParser::parse( const DDLDocumentProvider& dp )
{
  edm::LogInfo ("DDLParser") << "Start Parsing.  Validation is set off for the time being." << std::endl;
  // prep for pass 1 through DDD XML
  SAX2Parser_->setFeature( XMLUni::fgSAX2CoreValidation, false );
  SAX2Parser_->setFeature( XMLUni::fgSAX2CoreNameSpaces, false );

  //  This need be only done once, so might as well to it here.
  size_t fileIndex = 0;
  std::vector<std::string> fullFileName;
  const std::vector < std::string >& fileList = dp.getFileList();
  const std::vector < std::string >& urlList = dp.getURLList();
  
  for(; fileIndex < fileList.size(); ++fileIndex )
  { 
    std::string ts = urlList[fileIndex];
    std::string tf = fileList[fileIndex];
    if ( !ts.empty() ) {
      if ( ts[ts.size() - 1] == '/') {
	fullFileName.emplace_back( ts + tf );
      } else {
	fullFileName.emplace_back( ts + "/" + tf );
      }
    } else {
      fullFileName.emplace_back( tf );
    }
  }

  for( const auto& fnit : fullFileName ) 
  {
    size_t foundFile = isFound( extractFileName( fnit )); 
	
    if( !foundFile )
    {
      pair <std::string, std::string> pss;
      pss.first = extractFileName( fnit );
      pss.second = fnit;
      fileNames_[nFiles_++] = pss;
      parsed_[nFiles_ - 1] = false;
    }
  }
    
  // Start processing the files found in the config file.
  assert( fileNames_.size() == nFiles_ );

  // PASS 1:  This was added later (historically) to implement the DDD
  // requirement for Expressions.
  
  SAX2Parser_->setContentHandler(expHandler_);
  for( size_t i = 0; i < nFiles_; ++i )
  {
    if( !parsed_[i])
    {
      currFileName_ = fileNames_[i].second;
      expHandler_->setNameSpace( getNameSpace( extractFileName( currFileName_ )));
      parseFile(i);
    }
  }
  expHandler_->createDDConstants();

  // PASS 2:

  SAX2Parser_->setContentHandler(fileHandler_);

  // No need to validate (regardless of user's doValidation
  // because the files have already been validated on the first pass.
  // This optimization suggested by Martin Liendl.

  // Process files again.
  for( size_t i = 0; i < nFiles_; ++i )
  {
    if( !parsed_[i]) {
      currFileName_ = fileNames_[i].second;
      fileHandler_->setNameSpace( getNameSpace( extractFileName( currFileName_ )));
      parseFile(i);
      parsed_[i] = true;
      pair<std::string, std::string> namePair = fileNames_[i];
      LogDebug ("DDLParser") << "Completed parsing file " << namePair.second << std::endl;
    }
  }
  return 0;
}

void
DDLParser::parseFile( const int& numtoproc ) 
{
  if (!parsed_[numtoproc])
  {
    const std::string & fname = fileNames_[numtoproc].second;

    currFileName_ = fname;
    SAX2Parser_->parse(currFileName_.c_str());
  }
}

void
DDLParser::clearFiles( void )
{
  fileNames_.clear();
  parsed_.clear();
}

std::string const
DDLParser::extractFileName( const std::string& fullname )
{
  return( fullname.substr( fullname.rfind( '/' ) + 1 ));
}

std::string const
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
