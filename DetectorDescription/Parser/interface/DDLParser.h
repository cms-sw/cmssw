#ifndef DETECTOR_DESCRIPTION_PARSER_DDL_PARSER_H
#define DETECTOR_DESCRIPTION_PARSER_DDL_PARSER_H

#include "DetectorDescription/Parser/interface/DDLSAX2ExpressionHandler.h"
#include "DetectorDescription/Parser/interface/DDLSAX2FileHandler.h"
#include "DetectorDescription/Parser/interface/DDLSAX2Handler.h"
#include "FWCore/Concurrency/interface/Xerces.h"
#include <xercesc/sax/SAXException.hpp>
#include <xercesc/sax2/SAX2XMLReader.hpp>
#include <xercesc/sax2/XMLReaderFactory.hpp>
#include <xercesc/util/XercesDefs.hpp>
#include <xercesc/util/XercesVersion.hpp>
#include <iosfwd>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include <cstddef>

class DDCompactView;
class DDLDocumentProvider;
class DDLElementRegistry;

/// DDLParser is the main class of Detector Description Language Parser.
/** @class DDLParser
 * @author Michael Case
 *
 *  DDLParser.h  -  description
 *  -------------------
 *  begin: Mon Oct 22 2001
 *  email: case@ucdhep.ucdavis.edu
 *
 *  Singleton which controls the parsing of XML files (DDL).  It guarantees
 *  that a given filename will only be parsed once regardless of its path.
 *  It now relies on a DDLDocumentProvider class which provides a list of
 *  file names and URLs to be parsed.
 *
 *  It uses the Xerces C++ Parser from the Apache Group straight-forwardly.
 *  One main thing to note is that only one DDLParser can ever be made.  This
 *  allows for sub-components of the parser to easily find out information from
 *  the parser during run-time.
 *
 *  There is an interface to parse just one file.  If one uses this method
 *  and does not use the default DDLDocumentProvider 
 *  user is responsible for also setting the DDRootDef.
 *  
 *  Modification:
 *    2003-02-13: Michael Case, Stepan Wynhoff and Martin Liendl
 *    2003-02-24: same.
 *         DDLParser will use DDLDocumentProvider (abstract).  One of these
 *         and will be defaulted to DDLConfiguration.  This will read
 *         the "configuration.xml" file provided and will be used by the Parser
 *         to "get" the files.
 *   2005-11-13:  Michael Case
 *         removed some of the un-necessary methods that were deprecated.
 *   2010-01 to 2010-04 sometime: Michael Case
 *         removed singleton-ness.  MUST have a DDCompactView to refer to
 *         and no more default constructor at the moment.
 *   2010-07-29:  removed DDLConfiguration; use FIPConfiguration, it is easier.
 *         for CMSSW Framework example see XMLIdealGeometryESSource (different
 *         DDLDocumentProvider completely
 */
class DDLParser
{
 public:
  typedef XERCES_CPP_NAMESPACE::SAX2XMLReader SAX2XMLReader;

  typedef std::map< int, std::pair<std::string, std::string> > FileNameHolder;
  
  DDLParser( DDCompactView& cpv );

  DDLParser() = delete; 
  
  ~DDLParser();

  /// Parse all files. Return is meaningless.
  int parse( const DDLDocumentProvider& dp );

  /// Process a single files.
  /** 
   *  This method allows a user to add to an existing DDD by
   *  parsing a new XML file.  Ideally, these would be in addition
   *  to an existing DDD configuration which was processed using
   *  Parse(...).  
   *
   *  The idea is based on whether users decide that the configuration
   *  will only hold "standard geometry files" and that any ancillary 
   *  parameter files, filters and so forth will be unknown to the main
   *  configuration file.  For me, this seems to go against the principle
   *  of knowing what files are relevant because now, there is no central
   *  way to find out (externally) what XML files generate the DDD in memory.
   *
   *  On the other hand, if on any run, a dumpFileList is run, then 
   *  the user will at least know what files were used from where in 
   *  a given run.
   *
   *  2007-04-06:  filename is now relative to src directory of checkout
   *               (or share).  edm:FileInPath used internally.
   **/
  bool parseOneFile(const std::string& filename);

  // I ASSUME I take ownership of this blob
  //  void parse( std::vector<unsigned char>* ablob, unsigned int bsize ) ;
  //old way  void parse( const std::vector<unsigned char>& ablob, unsigned int bsize ) ;
  void parse( const std::vector<unsigned char>& ablob, unsigned int bsize ) ;

  /// Get the SAX2Parser from the DDLParser.  USE WITH CAUTION.  Set your own handler, etc.
  SAX2XMLReader* getXMLParser();

  /// To get the parent this class allows access to the handler.
  /**
   *  In order to retrieve the name of the parent element from DDLSAX2Handlers.
   */
  DDLSAX2FileHandler* getDDLSAX2FileHandler();
  
  /// Clear the file list - see Warning!
  /**
   *  This could result in mangled geometry if the Core has not been cleared.
   **/
  void clearFiles();

 private:
  /// Parse File.  Just to hold some common looking code.
  void parseFile (const int& numtoproc);

  /// Is the file already known by the DDLParser?  Returns 0 if not found, and index if found.
  size_t isFound(const std::string& filename);
  
  /// Is the file already parsed?
  bool isParsed(const std::string& filename);

  std::string const extractFileName( const std::string& fullname );

  std::string const getNameSpace( const std::string& fname );
  
  /// reference to storage
  DDCompactView& cpv_;

  /// List of files to be processed, obtained from the DDLDocumentProvider.
  FileNameHolder fileNames_;

  /// Parse status of a given file.
  std::map<int, bool> parsed_;

  /// Number of files + 1.
  size_t nFiles_;

  /// Which file is currently being processed.
  std::string currFileName_;

  /// SAX2XMLReader is one way of parsing.
  SAX2XMLReader* SAX2Parser_;

  DDLElementRegistry* elementRegistry_;
  DDLSAX2FileHandler* fileHandler_;
  DDLSAX2ExpressionHandler* expHandler_;
  DDLSAX2Handler* errHandler_;
};

#endif
