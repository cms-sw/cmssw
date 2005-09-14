#ifndef DDL_ParserI_H
#define DDL_ParserI_H
#include <vector>
#include <string>
#include<iosfwd> 

namespace seal {
  class Context;
}

class DDLDocumentProvider;
/// public abstract interface to DDLParser
/**
 **/
class DDLParserI {

 public:


  virtual ~DDLParserI(){}

  /// New Parse all files. FIX - After deprecated stuff removed, make this void
  virtual int parse( const DDLDocumentProvider& dp )=0;

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
  virtual bool parseOneFile(const std::string& filename, const std::string& url)=0;

  /// Return list of files
  virtual std::vector<std::string> getFileList()=0;

  /// Print out the list of files.
  virtual void dumpFileList()=0;
  virtual void dumpFileList(std::ostream& co)=0;

  /// Report which file currently being processed (or last processed).
  virtual std::string getCurrFileName()=0;

  /// Is the file already known by the DDLParser?  Returns 0 if not found, and index if found.
  virtual size_t isFound(const std::string& filename)=0;

  /// Is the file already parsed?
  virtual bool isParsed(const std::string& filename)=0;


  virtual DDLDocumentProvider * newConfig() const=0; 


};

#endif
