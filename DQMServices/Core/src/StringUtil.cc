#include "DQMServices/Core/interface/StringUtil.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include <iostream>

using namespace std;
using namespace dqm::me_util;

// yes if we have a match (<pattern> can include unix-like wildcards "*", "?")
bool StringUtil::matchit(const string & s, const string & pattern) const
{

  bool ret = false;

  // regex library needs dots in front of wildcards
  // (see addDots2WildCards for details)
  string pattern_ = addDots2WildCards(pattern);
  // end of line special character
  pattern_ += "$";

  regex_t com;

  if(regcomp(&com,pattern_.c_str(),REG_NOSUB|REG_EXTENDED)!=0)
    {
      cerr << " ** Bad regcomp in matchit function !" << endl;
      cerr << " Called with string = " << s << ", pattern = " << pattern
	   << endl;
      return ret;
    }

  // we only want exact matches
  if(regexec(&com,s.c_str(),0,0,0)==0)
    ret = true;

  // free memory allocated by regexec
  regfree(&com);

  return ret;
  
}    

// return <pathname>/<filename>
// (or just <filename> if <pathname> corresponds to top folder)
string 
StringUtil::getUnixName(const string & pathname, const string & filename) const
{
  if(isTopFolder(pathname))
    return filename;
  else 
    return pathname + "/" + filename;
}

// true if pathname corresponds to top folder
bool StringUtil::isTopFolder(const string & pathname) const
{
  if(pathname == "" || pathname == "/" || pathname == ROOT_PATHNAME)
    return true;
  else
    return false;
}

// if a search patten contains a wildcard (*, ?), the regex library 
// needs a preceding "." if it is to behave like a unix "?" or "*" wildcard;
// this function replaces "*" by ".*" and "?" by ".?"
string StringUtil::addDots2WildCards(const string & s) const
{
  string ret = replace_string(s, "*", ".*");
  string ret2 = replace_string(ret, "?", ".?");
  return ret2;
}

// find all "<subs>" in string <s>, replace by "<repl>"
string StringUtil::replace_string(const string & s, const string & subs,  
				  const string & repl) const
{
  // lengths of substring to be replaced & its replacement
  unsigned int subs_length = subs.length();
  unsigned int repl_length = repl.length();
  // next instance of substring that needs to be replaced (if any)
  unsigned int pos = s.find(subs);

  string ret = s;
  // stay here there is a substring (that hasn't been replaced already)
  while(pos != string::npos)
    {
      // replace substring at position <pos>
      ret.replace(pos, subs_length, repl); 
      // shift beginning of search after insertion of <repl>
      unsigned new_pos = pos + repl_length;
      // find new substring (if any)
      pos = ret.find(subs, new_pos);
    }
 
  return ret;
}

// Using fullpath as input (e.g. A/B/test.txt) 
// extract path (e.g. A/B) and filename (e.g. test.txt);
void StringUtil::unpack(const string & fullpath, string & path, 
				   string & filename) const
{
  unsigned n = fullpath.rfind("/");
  if(n == string::npos)
    {
      path = ROOT_PATHNAME;
      filename = fullpath;
    }
  else
    {
      path = fullpath.substr(0, n);
      filename = fullpath.substr(n+1, fullpath.length());
    }
}

/* true if <subdir> is a subfolder of <parentdir> (or same)
   eg. c0/c1/c2 is a subdirectory of c0/c1, but
   c0/c1_1 is not */
bool StringUtil::isSubdirectory(const string & parentdir_fullpath, 
				const string & subdir_fullpath) const
{
  // true if (a) exactly same pathname, or 
  // (b) <subdir> is a subdirectory, ie. there is a slash "/"
  if( subdir_fullpath == parentdir_fullpath)
    return true;

  if(subdir_fullpath.find(parentdir_fullpath + "/") != string::npos)
    return true;
  // this ensures that we won't consider eg. c0/c1_1
  // as a subdirectory of c0/c1

  return false;
}

// similar to isSubdirectory, with the exception that <subscription> is of the form:
// <directory pathname>:<h1>,<h2>,...
bool StringUtil::belongs2folder(const string & folder, 
				const string & subscription) const 
{
  // folder's name will appear: (1) either with a ":" (if there is no subdirectory),
  // or (2) with a "/", if there is one
  string case1 = folder + ":"; 
  string case2 = folder + "/"; 
  if( (subscription.find(case1) != string::npos) || 
      (subscription.find(case2) != string::npos))
    return true;
  else
    return false;
}

// unpack directory format (name); expected format: see DirFormat definition
// return success flag
bool StringUtil::unpackDirFormat(const string & name, DirFormat & dir) const
{
  // split name into <dir> and contents
  vector<string> subs = unpackString(name.c_str(), ":");
  
  if(subs.size() != 2)
    {
      errorObjUnp(subs);
      return false;
    }
  
  cvIt subst = subs.begin();
  // first get the directory pathname
  dir.dir_path = *subst;
  // now get the directory's contents
  ++subst;
  dir.contents = unpackString(subst->c_str(), ",");
  
  return true;
}

#include "DQMServices/Core/interface/Tokenizer.h"
using dqm::Tokenizer;

// unpack input string into vector<string> by using "separator"
vector<string> StringUtil::unpackString(const char *in, const char * separator) 
  const
{
  vector<string> retVal; retVal.clear();
  // return empty vector if in = ""
  if(strcmp(in,"")==0)
    return retVal;

  string names(in);
  // return whole string if separator=""
  if(strcmp(separator,"")==0)
    {
      retVal.push_back(names);
      return retVal;
    }

  Tokenizer obji(separator, names);
  for (Tokenizer::const_iterator on = obji.begin(); on != obji.end(); ++on) 
    {
      //      if(*on == "")continue;
      retVal.push_back(*on);
    }
  return retVal;
}

// called when encountering errors in monitoring object unpacking
void StringUtil::errorObjUnp(const vector<string> & desc) const
{
  cerr << " *** Error! Expected pathname and objects list! " << endl;
  for(vector<string>::const_iterator it = desc.begin(); it != desc.end(); 
      ++it)
    cout << " desc = " << *it << endl;
}

// print out error message
void StringUtil::nullError(const char * name)
{
  if(!name) 
    cerr << " *** Error! nullError called with null pointer "
	 << "(isn't that ironic...) " << endl;
  else
    cerr << " *** Error! Attempt to use null " << name << endl;
}
