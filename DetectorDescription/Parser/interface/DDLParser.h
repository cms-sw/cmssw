#ifndef DDL_Parser_H
#define DDL_Parser_H

// ---------------------------------------------------------------------------
//  Includes
// ---------------------------------------------------------------------------
#include "DetectorDescription/Core/interface/DDLParserI.h"

#include "DetectorDescription/Parser/interface/DDLSAX2ConfigHandler.h"
#include "DetectorDescription/Parser/interface/DDLSAX2FileHandler.h"

// Xerces C++ dependencies
#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/sax2/SAX2XMLReader.hpp>
#include <xercesc/sax2/XMLReaderFactory.hpp>
#include <xercesc/sax/SAXException.hpp>

#include <string>
#include <vector>
#include <map>
#include <iosfwd>

class DDLDocumentProvider;

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
 *  There is a new interface to parse just one file.  If one uses this method
 *  and does not use the default DDLDocumentProvider (DDLConfiguration) the
 *  user is responsible for also setting the DDRootDef or else I do not know
 *  how one would get a CompactView from the DDD.
 *  
 *  Modification:
 *    2003-02-13: Michael Case, Stepan Wynhoff and Martin Listd::endl
 *    2003-02-24: same.
 *         DDLParser will use DDLDocumentProvider (abstract).  One of these
 *         and will be defaulted to DDLConfiguration.  This will read
 *         the "configuration.xml" file provided and will be used by the Parser
 *         to "get" the files.
 *
 */
class DDLParser : public DDLParserI {

 public:

  typedef std::map< int, std::pair<std::string, std::string> > FileNameHolder;
  static DDLParser* instance();

  // MEC: EDMProto temporary? But we need it for 
  static void setInstance( DDLParser* p );

  ///DEPRECATED, PLEASE stop using.
  static DDLParser* Instance();

  /// unique (and default) constructor
  DDLParser(seal::Context* ic=0);

  ~DDLParser();

  /// DEPRECATED: Parse all files in the given configuration.
  int StartParsing();


  /// New Parse all files. FIX - After deprecated stuff removed, make this void
  int parse( const DDLDocumentProvider& dp );

  /// Deprecated... temporarily keep for backward compatibility.
  int SetConfig (const std::string& filename);

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
   **/
  bool parseOneFile(const std::string& filename, const std::string& url);

  /// Return list of files
  std::vector<std::string> getFileList();

  /// Print out the list of files.
  void dumpFileList();
  void dumpFileList(std::ostream& co);

  /// Report which file currently being processed (or last processed).
  std::string getCurrFileName();

  /// Get the SAX2Parser from the DDLParser.  USE WITH CAUTION.  Set your own handler, etc.
  /**
   *  I wanted (needed?) to do this for the DDLConfiguration to do the parsing separately.
   *  Since these two classes are so connected I wonder if I should remove this, make
   *  DDLConfiguration a friend of this guy and let it access the SAX2XMLReader directly.
   */
  SAX2XMLReader* getXMLParser();

  /// To get the parent this class allows access to the handler.
  /**
   *  In order to retrieve the name of the parent element from DDLSAX2Handlers.
   */
  DDLSAX2FileHandler* getDDLSAX2FileHandler();

  /// Is the file already known by the DDLParser?  Returns 0 if not found, and index if found.
  size_t isFound(const std::string& filename);

  /// Is the file already parsed?
  bool isParsed(const std::string& filename);

  // it may not belong here...
  virtual DDLDocumentProvider * newConfig() const;

 protected:

  /// Parse File.  Just to hold some common looking code.
  void parseFile (const int& numtoproc);

 private:
  /// For Singleton behavior.
  static DDLParser* instance_;

  /// List of files to be processed, obtained from the DDLDocumentProvider.
  FileNameHolder fileNames_;

  /// Parse status of a given file.
  std::map<int, bool> parsed_;

  /// Number of files + 1.
  int nFiles_;

  /// Configuration file name.  Only necessary until deprecated methods removed.
  std::string configFileName_;

  /// Which file is currently being processed.
  std::string currFileName_;

  /// SAX2XMLReader is one way of parsing.
  SAX2XMLReader* SAX2Parser_;

  //  DDLSAX2ConfigHandler* handler_;

  
};

#endif
