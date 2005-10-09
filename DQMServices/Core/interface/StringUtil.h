#ifndef StringUtil_h
#define StringUtil_h

#include <string>
#include <vector>
#include <sys/types.h>
#include <regex.h>

// Class with string utilities; Use to convert strings from a unix-like format
// with wildcards (eg. C1/C2/*, C1/C2/h?, etc) to a format that the internal DQM
// logic understands: <directory pathname>:<h1>,<h2>,...
class StringUtil
{
 public:
  StringUtil(void){}
  ~StringUtil(void){}

 private:
  // if a search patten contains a wildcard (*, ?), the regex library 
  // needs a preceding "." if it is to behave like a unix "?" or "*" wildcard;
  // this function replaces "*" by ".*" and "?" by ".?"
  std::string addDots2WildCards(const std::string & s) const;
  // find all "<subs>" in string <s>, replace by "<repl>"
  std::string replace_string(const std::string& s, const std::string& subs, 
			     const std::string & repl) const;

  // called when encountering errors in monitoring object unpacking
  void errorObjUnp(const std::vector<std::string> & desc) const;

 protected:
  
  // yes if we have a match (<pattern> can include unix-like wildcards "*", "?")
  bool matchit(const std::string & s, const std::string & pattern) const;
  // return <pathname>/<filename>
  // (or just <filename> if <pathname> corresponds to top folder)
  std::string getUnixName(const std::string & pathname, 
			  const std::string & filename) const;
  // true if pathname corresponds to top folder
  bool isTopFolder(const std::string & pathname) const;

  // Using fullpath as input (e.g. A/B/test.txt) 
  // extract path (e.g. A/B) and filename (e.g. test.txt)
  void unpack(const std::string & fullpath, 
	      std::string & path, std::string & filename) const;

  /* true if <subdir> is a subfolder of <parent_dir> (or same)
     eg. c0/c1/c2 is a subdirectory of c0/c1, but
     c0/c1_1 is not */
  bool isSubdirectory(const std::string & parentdir_fullpath, 
		      const std::string & subdir_fullpath) const;

  // similar to isSubdirectory, with the exception that <subscription> is 
  // of the form: <directory pathname>:<h1>,<h2>,...
  bool belongs2folder(const std::string & folder, 
		      const std::string & subscription) const;

  // structure for unpacking "directory" format: <dir_path>:<obj1>,<obj2>,...
  // if directory appears empty, all owned objects are implied
  struct DirFormat_{
    std::string dir_path;  // full directory pathname
    std::vector<std::string> contents; // vector of monitoring objects
  };
  typedef struct DirFormat_ DirFormat;

  // unpack directory format (name); expected format: see DirFormat definition
  // return success flag
  bool unpackDirFormat(const std::string & name, DirFormat & dir) const;
  // unpack input string into vector<string> by using "separator"
  std::vector<std::string> 
    unpackString(const char* in, const char* separator) const;

};

#endif // define StringUtil_h
