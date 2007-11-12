#include "DQMServices/Core/interface/StringUtil.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/Tokenizer.h"

#include <iostream>

using namespace dqm::me_util;

using std::cout; using std::endl; using std::cerr;
using std::string; using std::vector;

// yes if we have a match (<pattern> can include unix-like wildcards "*", "?")
bool StringUtil::matchit(const string & s, const string & pattern)
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
StringUtil::getUnixName(const string & pathname, const string & filename)
{
  if(isTopFolder(pathname))
    return filename;
  else 
    return pathname + "/" + filename;
}

// true if pathname corresponds to top folder
bool StringUtil::isTopFolder(const string & pathname)
{
  if(pathname == "" || pathname == "/" || pathname == ROOT_PATHNAME)
    return true;
  else
    return false;
}

// if a search patten contains a wildcard (*, ?), the regex library 
// needs a preceding "." if it is to behave like a unix "?" or "*" wildcard;
// this function replaces "*" by ".*" and "?" by ".?"
string StringUtil::addDots2WildCards(const string & s)
{
  string ret = replace_string(s, "*", ".*");
  string ret2 = replace_string(ret, "?", ".?");
  string ret3 = replace_string(ret2, "+", "\\+");
  return ret3;
}

// find all "<subs>" in string <s>, replace by "<repl>"
string StringUtil::replace_string(const string & s, const string & subs,  
				  const string & repl)
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
				   string & filename)
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
				const string & subdir_fullpath)
{
  // true if (a) parent directory is root, or 
  // (b) exactly same pathname, or 
  // (c) <subdir> is a subdirectory, ie. there is a slash "/"
  if(parentdir_fullpath == ROOT_PATHNAME)
    return true;

  if( subdir_fullpath == parentdir_fullpath)
    return true;

  string look4this = parentdir_fullpath + "/";
  if((subdir_fullpath.find(look4this) != string::npos)
     && subdir_fullpath.find(look4this) == 0)
    return true;
  // first check ensures that we won't consider eg. c0/c1_1
  // as a subdirectory of c0/c1;
  // second check ensures that we won't consider eg. c0/c1/c2 
  // as a subdirectory of c1

  return false;
}

// similar to isSubdirectory, with the exception that <subscription> is of the form:
// <directory pathname>:<h1>,<h2>,...
bool StringUtil::belongs2folder(const string & folder, 
				       const string & subscription)
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
bool StringUtil::unpackDirFormat(const string & name, DirFormat & dir)
{
  // split name into <dir> and contents
  vector<string> subs; 
  unpackString(name.c_str(), ":", subs);
  
  // size could be either 2 (dir-pathname and objects) or 3 (also tag)
  if(subs.size() < 2 || subs.size() > 3) 
    {
      errorObjUnp(subs);
      return false;
    }
  
  cvIt subst = subs.begin();
  // first get the directory pathname
  dir.dir_path = *subst;
  // now get the directory's contents
  ++subst;
  unpackString(subst->c_str(), ",", dir.contents);
  if(subs.size() == 3)
    {
      // now get to the tag
      ++subst;
      dir.tag = (unsigned int) atoi(subst->c_str());
    }

  return true;
}

using dqm::Tokenizer;

// unpack input string into vector<string> by using "separator"
void StringUtil::unpackString(const char *in, const char * separator,
			      vector<string> & put_here)
{
  put_here.clear();
  // return empty vector if in = ""
  if(strcmp(in,"")==0)
    return;

  string names(in);
  // return whole string if separator=""
  if(strcmp(separator,"")==0)
    {
      put_here.push_back(names);
      return;
    }

  Tokenizer obji(separator, names);
  for (Tokenizer::const_iterator on = obji.begin(); on != obji.end(); ++on) 
    {
      //      if(*on == "")continue;
      put_here.push_back(*on);
    }
}

// called when encountering errors in monitoring object unpacking
void StringUtil::errorObjUnp(const vector<string> & desc)
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

// unpack QReport (with name, value) into ME_name, qtest_name, status, message;
// return success flag; Expected format of QReport is a TNamed variable with
// (a) name in the form: <ME_name>.<QTest_name>
// (b) title (value) in the form: st.<status>.<the message here>
// (where <status> is defined in Core/interface/QTestStatus.h)
bool StringUtil::unpackQReport(string name, string value, string & ME_name, 
			       string & qtest_name, int & status, 
			       string & message)
{
  unsigned _begin = 0;
  unsigned _middle = name.find(".");
  unsigned _end = name.size();

  // first unpack ME and quality test name
  if(_begin != string::npos && _middle != string::npos 
     &&_end != string::npos)
    {  
      ME_name = name.substr(_begin, _middle-_begin);
      qtest_name = name.substr(_middle+1, _end -_middle-1);
    }
  else
    {
      cerr << " *** Unpacking of QReport name = " << name
	   << " has failed " << endl;
      return false;
    }
      
  _begin = value.find("st.") + 3;
  _middle = value.find(".", _begin);
  _end = value.size();
  // then unpack status value and message
  if(_begin != string::npos && _middle != string::npos 
     &&_end != string::npos)
    {  
      string status_string = value.substr(_begin, _middle-_begin);
      status = atoi(status_string.c_str());
      message = value.substr(_middle+1, _end -_middle-1);
    }
  else
    {
      cerr << " *** Unpacking of QReport value = " << value
	   << " has failed " << endl;
      return false;
    }

  // if here, everything is good
  return true;

}

// keep maximum pathname till first wildcard (?, *); put rest in chopped_part
// Examples: 
// "a/b/c/*"    -->  "a/b/c/", chopped_part = "*"
// "a/b/c*"     -->  "a/b/c" , chopped_part = "*"
// "a/b/c/d?/*" -->  "a/b/c/d", chopped_part = "?/*"
string StringUtil::getMaxPathname(const string & search_string, 
				  string & chopped_part)
{
  string ret = search_string;
  chopped_part.clear();
  
  unsigned i = ret.find("?");
  if(i != string::npos)
    {
      chopped_part = ret.substr(i, ret.length()-1);
      ret = ret.substr(0, i);
    }
  
  i = ret.find("*");
  if(i != string::npos)
    {
      chopped_part = ret.substr(i, ret.length()-1) + chopped_part;
      ret = ret.substr(0, i);
    }
  return ret;    
}

/* get parent directory. 
   Examples: 
   (a) A/B/C --> A/B
     (b) A/B/  --> A/B (because last slash implies subdirectories of A/B)
     (c) C     --> ROOT_PATHNAME  (".": top folder)
*/
string StringUtil::getParentDirectory(const string & pathname)
{
  string ret = pathname;
  unsigned i = ret.rfind("/");
  if(i != string::npos)
    ret = ret.substr(0, i);
  else
    ret = dqm::me_util::ROOT_PATHNAME;
  
  return ret;
}

// true if string includes any wildcards ("*" or "?")
bool StringUtil::hasWildCards(const string & pathname)
{
  if(pathname.find("?") != string::npos || 
     pathname.find("*") != string::npos)
    return true;
  else
    return false;
}
