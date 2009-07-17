// -*- C++ -*-
#include <iostream>
#include <string>
#include <algorithm>
#include <cctype>
#include <fstream>
#include <iomanip>

#include "TString.h"

#include "FWCore/FWLite/interface/OptionUtils.h"
#include "FWCore/FWLite/interface/dout.h"
#include "FWCore/FWLite/interface/dumpSTL.icc"

using namespace std;

// Variable definitions
optutl::SIMap     optutl::ns_integerMap;
optutl::SDMap     optutl::ns_doubleMap;
optutl::SSMap     optutl::ns_stringMap;
optutl::SBMap     optutl::ns_boolMap;
optutl::SIVecMap  optutl::ns_integerVecMap;
optutl::SDVecMap  optutl::ns_doubleVecMap;
optutl::SSVecMap  optutl::ns_stringVecMap;

optutl::SBMap     optutl::ns_variableModifiedMap;
optutl::SSMap     optutl::ns_variableDescriptionMap;
optutl::SVec      optutl::ns_fullArgVec;
string            optutl::ns_argv0;
string            optutl::ns_usageString;
bool              optutl::ns_printOptions = false;

optutl::SVec
optutl::parseArguments (int argc, char** argv, bool returnArgs)
{   
   bool callHelp     = false;
   SVec argsVec;
   ns_argv0 = argv[0];
   ns_fullArgVec.push_back (argv[0]);
   for (int loop = 1; loop < argc; ++loop)
   {
      string arg = argv[loop];
      ns_fullArgVec.push_back (arg);
      string::size_type where = arg.find_first_of("=");
      if (string::npos != where)
      {
         if ( setVariableFromString (arg) )
         {
            continue;
         }
         if ( runVariableCommandFromString (arg) )
         {
            continue;
         }
         cerr << "Don't understand: " << arg << endl;
         exit(0);
      } // tag=value strings
      else if (arg.at(0) == '-')
      {
         string::size_type where = arg.find_first_not_of("-");
         if (string::npos == where)
         {
            // a poorly formed option
            cerr << "Don't understand: " << arg << endl;
            exit(0);
            continue;
         }
         lowercaseString (arg);
         char first = arg.at (where);
         // Print the values
         if ('p' == first)
         {
            ns_printOptions = true;
            continue;
         }
         // Exit after printing values
         if ('e' == first || 'h' == first)
         {
            callHelp = true;
            continue;
         }
         // if we're still here, then we've got a problem.
         cerr << "Don't understand: " << arg << endl;
         exit(0);
      } // -arg strings
      if (returnArgs)
      {
         argsVec.push_back (arg);
      } else {
         cerr << "Don't understand: " << arg << endl;
         exit(0);
      }
   } // for loop
   if (callHelp)
   {
      help();
   }
   return argsVec;
}

void 
optutl::setUsageString (const std::string &usage) 
{ 
   ns_usageString = usage; 
}

void
optutl::help()
{
   if (ns_usageString.length())
   {
      cout << ns_argv0 << " - " << ns_usageString << endl;
   } else {
      cout << "dude" << endl;
   }
   printOptionValues();
   exit (0);
}

optutl::OptionType
optutl::hasOption (string key)
{
   lowercaseString (key); 
   // Look through our maps to see if we've got it
   if (ns_integerMap.end()    != ns_integerMap.find (key))    return kInteger;
   if (ns_doubleMap.end()     != ns_doubleMap.find (key))     return kDouble;
   if (ns_stringMap.end()     != ns_stringMap.find (key))     return kString;
   if (ns_boolMap.end()       != ns_boolMap.find (key))       return kBool;
   if (ns_integerVecMap.end() != ns_integerVecMap.find (key)) return kIntegerVector;
   if (ns_doubleVecMap.end()  != ns_doubleVecMap.find (key))  return kDoubleVector;
   if (ns_stringVecMap.end()  != ns_stringVecMap.find (key))  return kStringVector;
   // if we're here, the answer's no.
   return kNone;
}

void 
optutl::split (SVec &retval, string line, string match, 
                    bool ignoreComments)
{
   if (ignoreComments)
   {
      removeComment (line);
   } // if ignoreComments
   retval.clear();
   // find the first non-space
   string::size_type start1 = line.find_first_not_of(kSpaces);
   // Is the first non-space character a '#'
   char firstCh = line[start1];
   if ('#' == firstCh)
   {
      // this line is a comment
      return;
   }

   line += match; // get last word of line
   string::size_type last = string::npos;
   string::size_type current = line.find_first_of(match);
   while (string::npos != current)
   {
      string::size_type pos;
      if (string::npos != last)
      {
         pos = last + 1;
      } else {
         pos = 0;
      }
      string part = line.substr( pos, current - last - 1);
      // don't bother adding 0 length strings
      if (part.length()) 
      {
         retval.push_back(part);
      }
      last = current;
      current = line.find_first_of(match, current + 1);
   } // while we're finding spaces
}

void
optutl::removeComment (string &line)
{
   string::size_type location = line.find ("#");
   if (string::npos != location)
   {
      // we've got a comment.  Strip it out
      line = line.substr (0, location - 1);
   } // if found
}

void
optutl::removeLeadingAndTrailingSpaces (std::string &line)
{
   string::size_type pos = line.find_first_not_of (kSpaces);
   if (string::npos == pos)
   {
      // we don't have anything here at all.  Just quit now
      return;
   }
   if (pos)
   {
      // We have spaces at the beginning.
      line = line.substr (pos);
   }
   pos = line.find_last_not_of (kSpaces);
   if (pos + 1 != line.length())
   {
      // we've got spaces at the end
      line = line.substr (0, pos + 1);
   }
}
 
string
optutl::removeEnding (const string &input, const string &ending)
{
   string::size_type position = input.rfind(ending);
   if (input.length() - ending.length() == position)
   {
      // we've got it
      return input.substr(0, position);
   }
   // If we're still here, it wasn't there
   return input;
}

void 
optutl::findCommand (const string &line,
                          string &command,
                          string &rest)
{
   command = rest = "";
   string::size_type nonspace = line.find_first_not_of (kSpaces);
   if (string::npos == nonspace)
   {
      // we don't have anything here at all.  Just quit now
      return;
   }
   string::size_type space = line.find_first_of (kSpaces, nonspace);
   if (string::npos == space)
   {
      // we only have a command and nothing else
      command = line.substr (nonspace);
      return;
   }
   command = line.substr (nonspace, space - nonspace);
   rest    = line.substr (space + 1);
   removeLeadingAndTrailingSpaces (rest);
}


void
optutl::printOptionValues()
{
   cout << "------------------------------------------------------------------" 
        << left << endl << "Option Values:" << endl;
   // Print the integers next
   if (ns_integerMap.size())
   {
      cout << "  Integer options:" << endl;
   }
   for (SIMapConstIter iter = ns_integerMap.begin(); 
       ns_integerMap.end() != iter; 
       ++iter) 
   {
      const string &description = ns_variableDescriptionMap[ iter->first ];
      cout << "    " << setw(14) << iter->first << " = " << setw(14)
           << iter->second;
      if (description.length())
      {
         cout << " - " << description;
      }
      cout << endl;
   } // for iter

   // Print the doubles next
   if (ns_doubleMap.size())
   {
      cout << "  Double options:" << endl;
   }
   for (SDMapConstIter iter = ns_doubleMap.begin(); 
       ns_doubleMap.end() != iter; 
       ++iter) 
   {
      const string &description = ns_variableDescriptionMap[ iter->first ];
      cout << "    " << setw(14) << iter->first << " = " << setw(14) 
           << iter->second;
      if (description.length())
      {
         cout << " - " << description;
      }
      cout << endl;
   } // for iter

   // Print the bools first
   if (ns_boolMap.size())
   {
      cout << "  Bool options:" << endl;
   }
   for (SBMapConstIter iter = ns_boolMap.begin(); 
       ns_boolMap.end() != iter; 
       ++iter) 
   {
      const string &description = ns_variableDescriptionMap[ iter->first ];
      cout << "    " << setw(14) << iter->first << " = " << setw(14);
      if (iter->second)
      {
         cout << "true";
      } else {
         cout << "false";
      }
      if (description.length())
      {
         cout << " - " << description;
      }
      cout << endl;
   } // for iter

   // Print the strings next
   if (ns_stringMap.size())
   {
      cout << "  String options:" << endl;
   }
   for (SSMapConstIter iter = ns_stringMap.begin(); 
       ns_stringMap.end() != iter; 
       ++iter) 
   {
      const string &description = ns_variableDescriptionMap[ iter->first ];
      cout << "    " << setw(14) << iter->first << " = ";
      const string value = "'" + iter->second + "'";
      cout << setw(14) << value;
      if (description.length())
      {
         cout << " - " << description;
      }
      cout << endl;
   } // for iter

   // Integer Vec
   if (ns_integerVecMap.size())
   {
      cout << "  Integer Vector options:" << endl;
   }
   for (SIVecMapConstIter iter = ns_integerVecMap.begin(); 
       ns_integerVecMap.end() != iter; 
       ++iter) 
   {
      const string &description = ns_variableDescriptionMap[ iter->first ];
      cout << "    " << setw(14) << iter->first << " = ";
      dumpSTL (iter->second); 
      cout << endl;
      if (description.length())
      {
         cout << "      - " << description;
      }
      cout << endl;
   } // for iter

   // Double Vec
   if (ns_doubleVecMap.size())
   {
      cout << "  Double Vector options:" << endl;
   }
   for (SDVecMapConstIter iter = ns_doubleVecMap.begin(); 
       ns_doubleVecMap.end() != iter; 
       ++iter) 
   {
      const string &description = ns_variableDescriptionMap[ iter->first ];
      cout << "    " << setw(14) << iter->first << " = ";
      dumpSTL (iter->second); 
      cout << endl;
      if (description.length())
      {
         cout << "      - " << description;
      }
      cout << endl;
   } // for iter

   // String Vec
   if (ns_stringVecMap.size())
   {
      cout << "  String Vector options:" << endl;
   }
   for (SSVecMapConstIter iter = ns_stringVecMap.begin(); 
       ns_stringVecMap.end() != iter; 
       ++iter) 
   {
      const string &description = ns_variableDescriptionMap[ iter->first ];
      cout << "    " << setw(14) << iter->first << " = ";
      dumpSTL (iter->second); 
      cout << endl;
      if (description.length())
      {
         cout << "      - " << description;
      }
      cout << endl;
   } // for iter

   cout << "------------------------------------------------------------------" 
        << right << endl;
}

void 
optutl::lowercaseString(string& arg)
{
   // assumes 'toLower(ch)' modifies ch
   std::for_each (arg.begin(), arg.end(), optutl::toLower);
   // // assumes 'toLower(ch)' returns the lower case char
   // std::transform (arg.begin(), arg.end(), arg.begin(), 
   //                 optutl::toLower);
}

char 
optutl::toLower (char& ch)
{
   ch = tolower (ch);
   return ch;
}

void
optutl::_checkKey (string &key, const string &description)
{
   // Let's make sure we don't already have this key
   lowercaseString (key);
   if ( ns_variableModifiedMap.end() != ns_variableModifiedMap.find (key) )
   {
      cerr << "optutl::addOption() Error: Key '" << key 
           << "' has already been defined.  Aborting." << endl;
      assert (0);
   } // found a duplicate
   ns_variableModifiedMap[key]     = false;
   ns_variableDescriptionMap[key]  = description;
}

void
optutl::addOption (string key, OptionType type,
                        const string &description)
{
   _checkKey (key, description);
   if (kInteger    == type)
   {
      ns_integerMap[key]    = kDefaultInteger;
      return;
   } 
   if (kDouble     == type)
   {
      ns_doubleMap[key]    = kDefaultDouble;
      return;
   } 
   if (kString     == type)
   {
      ns_stringMap[key]    = kDefaultString;
      return;
   } 
   if (kBool       == type)
   {
      ns_boolMap[key]    = kDefaultBool;
      return;
   }
   if (kIntegerVector == type)
   {
      ns_integerVecMap[key] = kEmptyIVec;
      return;
   } 
   if (kDoubleVector == type)
   {
      ns_doubleVecMap[key] = kEmptyDVec;
      return;
   } 
   if (kStringVector == type)
   {
      ns_stringVecMap[key] = kEmptySVec;
      return;
   } 
}

void
optutl::addOption (string key, OptionType type,
                        const string &description, int defaultValue)
{
   _checkKey (key, description);
   if (kInteger != type)
   {
      cerr << "optutl::addOption() Error: Key '" << key 
           << "' is not defined as an integer but has an integer "
           << "default value. Aborting." << endl;
      assert (0);      
   }
   ns_integerMap[key] = defaultValue;
}

void
optutl::addOption (string key, OptionType type,
                        const string &description, double defaultValue)
{
   _checkKey (key, description);
   if (kDouble != type)
   {
      cerr << "optutl::addOption() Error: Key '" << key 
           << "' is not defined as an double but has an double "
           << "default value. Aborting." << endl;
      assert (0);      
   }
   ns_doubleMap[key] = defaultValue;
}

void
optutl::addOption (string key, OptionType type,
                        const string &description, 
                        const string &defaultValue)
{
   _checkKey (key, description);
   if (kString != type)
   {
      cerr << "optutl::addOption() Error: Key '" << key 
           << "' is not defined as an string but has an string "
           << "default value. Aborting." << endl;
      assert (0);      
   }
   ns_stringMap[key] = defaultValue;
}

void
optutl::addOption (string key, OptionType type,
                        const string &description, 
                        const char* defaultValue)
{
   addOption (key, type, description, (string) defaultValue);
}

void
optutl::addOption (string key, OptionType type,
                        const string &description, bool defaultValue)
{
   _checkKey (key, description);
   if (kBool != type)
   {
      cerr << "optutl::addOption() Error: Key '" << key 
           << "' is not defined as an bool but has an bool "
           << "default value. Aborting." << endl;
      assert (0);      
   }
   ns_boolMap[key] = defaultValue;
}

int &
optutl::integerValue (std::string key)
{
   lowercaseString (key);
   SIMapIter iter = ns_integerMap.find (key);
   if (ns_integerMap.end() == iter)
   {
      cerr << "optutl::integerValue() Error: key '"
           << key << "' not found.  Aborting." << endl;
      assert (0);
   }
   return iter->second;
}

double &
optutl::doubleValue (std::string key)
{
   lowercaseString (key);
   SDMapIter iter = ns_doubleMap.find (key);
   if (ns_doubleMap.end() == iter)
   {
      cerr << "optutl::doubleValue() Error: key '"
           << key << "' not found.  Aborting." << endl;
      assert (0);
   }
   return iter->second;
}

string &
optutl::stringValue (std::string key)
{
   lowercaseString (key);
   SSMapIter iter = ns_stringMap.find (key);
   if (ns_stringMap.end() == iter)
   {
      cerr << "optutl::stringValue() Error: key '"
           << key << "' not found.  Aborting." << endl;
      assert (0);
   }
   return iter->second;
}

bool &
optutl::boolValue (std::string key)
{
   lowercaseString (key);
   SBMapIter iter = ns_boolMap.find (key);
   if (ns_boolMap.end() == iter)
   {
      cerr << "optutl::boolValue() Error: key '"
           << key << "' not found.  Aborting." << endl;
      assert (0);
   }
   return iter->second;
}

optutl::IVec &
optutl::integerVector (std::string key)
{
   lowercaseString (key);
   SIVecMapIter iter = ns_integerVecMap.find (key);
   if (ns_integerVecMap.end() == iter)
   {
      cerr << "optutl::integerVector() Error: key '"
           << key << "' not found.  Aborting." << endl;
      assert (0);
   }
   return iter->second;
}

optutl::DVec &
optutl::doubleVector (std::string key)
{
   lowercaseString (key);
   SDVecMapIter iter = ns_doubleVecMap.find (key);
   if (ns_doubleVecMap.end() == iter)
   {
      cerr << "optutl::doubleVector() Error: key '"
           << key << "' not found.  Aborting." << endl;
      assert (0);
   }
   return iter->second;
}

optutl::SVec &
optutl::stringVector (std::string key)
{
   lowercaseString (key);
   SSVecMapIter iter = ns_stringVecMap.find (key);
   if (ns_stringVecMap.end() == iter)
   {
      cerr << "optutl::stringVector() Error: key '"
           << key << "' not found.  Aborting." << endl;
      assert (0);
   }
   return iter->second;
}

bool
optutl::valueHasBeenModified (const string &key)
{
   SBMapConstIter iter = ns_variableModifiedMap.find (key);
   if (ns_variableModifiedMap.end() == iter)
   {
      // Not found.  Not a valid option
      cerr << "optutl::valueHasBeenModfied () Error: '" 
           << key << "' is not a valid key." << endl;
      return false;
   }
   return iter->second;
}

bool 
optutl::setVariableFromString (const string &arg,
                                    bool dontOverrideChange,
                                    int offset)
{
   string::size_type where = arg.find_first_of("=", offset + 1);
   string varname = arg.substr (offset, where - offset);
   string value   = arg.substr (where + 1);
   lowercaseString (varname);
   // check to make sure this is a valid option
   SBMapConstIter sbiter = ns_variableModifiedMap.find (varname);
   if (ns_variableModifiedMap.end() == sbiter)
   {
      // Not found.  Not a valid option
      return false;
   }
   // if 'dontOverrideChange' is set, then we are being asked to NOT
   // change any variables that have already been changed.
   if (dontOverrideChange && valueHasBeenModified (varname) )
   {
      // don't go any further
      return true;
   }
   // integers
   SIMapIter integerIter = ns_integerMap.find(varname);
   if (ns_integerMap.end() != integerIter)
   {
      // we found it
      // use 'atof' instead of 'atoi' to get scientific notation
      integerIter->second = (int) atof( value.c_str() );
      ns_variableModifiedMap[varname] = true;
      return true;
   }
   // double
   SDMapIter doubleIter = ns_doubleMap.find(varname);
   if (ns_doubleMap.end() != doubleIter)
   {
      // we found it
      doubleIter->second = atof( value.c_str() );
      ns_variableModifiedMap[varname] = true;
      return true;
   }
   // string
   SSMapIter stringIter = ns_stringMap.find(varname);
   if (ns_stringMap.end() != stringIter)
   {
      // we found it
      stringIter->second = value;
      ns_variableModifiedMap[varname] = true;
      return true;
   }
   // bool
   SBMapIter boolIter = ns_boolMap.find(varname);
   if (ns_boolMap.end() != boolIter)
   {
      // we found it
      boolIter->second = 0 != atoi( value.c_str() );
      ns_variableModifiedMap[varname] = true;
      return true;
   }
   // IntegerVec
   SIVecMapIter integerVecIter = ns_integerVecMap.find(varname);
   if (ns_integerVecMap.end() != integerVecIter)
   {
      // we found it
      SVec words;
      split (words, value, ",");
      for (SVecConstIter wordIter = words.begin();
           words.end() != wordIter;
           ++wordIter)
      {
         integerVecIter->second.push_back( (int) atof( wordIter->c_str() ) );
      }
      // we don't want to mark this as modified because we can add
      // many values to this
      // ns_variableModifiedMap[varname] = true;
      return true;
   }
   // DoubleVec
   SDVecMapIter doubleVecIter = ns_doubleVecMap.find(varname);
   if (ns_doubleVecMap.end() != doubleVecIter)
   {
      // we found it
      SVec words;
      split (words, value, ",");
      for (SVecConstIter wordIter = words.begin();
           words.end() != wordIter;
           ++wordIter)
      {
         doubleVecIter->second.push_back( atof( wordIter->c_str() ) );
      }
      // we don't want to mark this as modified because we can add
      // many values to this
      // ns_variableModifiedMap[varname] = true;
      return true;
   }
   // StringVec
   SSVecMapIter stringVecIter = ns_stringVecMap.find(varname);
   if (ns_stringVecMap.end() != stringVecIter)
   {
      // we found it
      SVec words;
      split (words, value, ",");
      for (SVecConstIter wordIter = words.begin();
           words.end() != wordIter;
           ++wordIter)
      {
         stringVecIter->second.push_back( *wordIter );
      }
      // we don't want to mark this as modified because we can add
      // many values to this
      // ns_variableModifiedMap[varname] = true;
      return true;
   }
   // We didn't find your variable.  And we really shouldn't be here
   // because we should have know that we didn't find your variable.
   cerr << "optutl::SetVeriableFromString() Error: "
        << "Unknown variable and internal fault.  Aborting." << endl;
   assert (0);
   return false;
}

bool
optutl::setVariablesFromFile (const string &filename)
{
   ifstream source (filename.c_str(), ios::in);
   if (! source)
   {
      cerr << "file " << filename << "could not be opened" << endl;
      return false;
   }
   string line;
   while (getline (source, line))
   {
      // find the first nonspace
      string::size_type where = line.find_first_not_of(kSpaces);
      if (string::npos == where)
      {
         // no non-spaces
         continue;
      } 
      char first = line.at (where);
      if ('-' != first)
      {
         continue;
      }
      where = line.find_first_not_of(kSpaces, where + 1);
      if (string::npos == where)
      {
         // no non-spaces
         continue;
      }
      // Get string starting at first nonspace after '-'.  Copy it to
      // another string without copying any spaces and stopping at the
      // first '#'.
      string withspaces = line.substr (where);
      string nospaces;
      for (int position = 0; 
           position < (int) withspaces.length(); 
           ++position)
      {
         char ch = withspaces[position];
         if ('#' == ch)
         {
            // start of a comment
            break;
         } else if (' ' == ch || '\t' == ch)
         {
            continue;
         }
         nospaces += ch;
      } // for position
      if (! setVariableFromString (nospaces, true) )
      {
         cerr << "Don't understand line" << endl << line << endl
              << "in options file '" << filename << "'.  Aborting."
              << endl;
         exit(0);
      } // if setting variable failed
   } // while getline
   return true;
}

bool
optutl::runVariableCommandFromString (const string &arg)
{
   SVec equalWords;
   split (equalWords, arg, "=");
   if (2 != equalWords.size())
   {
      return false;
   }
   SVec commandWords;
   split (commandWords, equalWords.at(0), "_");
   if (2 != commandWords.size())
   {
      return false;
   }
   string &command = commandWords.at(1);
   lowercaseString (command);   
   if (command != "load" && command != "clear")
   {
      return false;
   }
   const string &key = commandWords.at(0);
   OptionType type = hasOption (key);
   if (type < kIntegerVector || type > kStringVector)
   {
      cerr << "Command '" << command << "' only works on vectors." << endl;
      return false;
   }

   ///////////
   // Clear //
   ///////////
   if ("clear" == command)
   {
      if (kIntegerVector == type)
      {
         integerVector(key).clear();         
      } else if (kDoubleVector == type)
      {
         doubleVector(key).clear();
      } else if (kStringVector == type)
      {
         stringVector(key).clear();
      } else {
         // If we're here, then I made a coding mistake and want to
         // know about it.
         assert (0);
      }
      return true;
   }

   //////////
   // Load //
   //////////
   const string &filename = equalWords.at(1);
   ifstream source (filename.c_str(), ios::in);
   if (! source)
   {
      cerr << "file " << filename << "could not be opened" << endl;
      return false;
   }
   string line;
   while (getline (source, line))
   {
      // find the first nonspace
      string::size_type where = line.find_first_not_of (kSpaces);
      if (string::npos == where)
      {
         // no non-spaces
         continue;
      }
      // get rid of leading spaces
      line = line.substr (where);
      // get rid of trailing spaces
      where = line.find_last_not_of (kSpaces);
      if (line.length() - 1 != where)
      {
         line = line.substr (0, where - 1);
      }
      if ('#' == line.at(0))
      {
         // this is a comment line, ignore it
         continue;
      }
      if (kIntegerVector == type)
      {
         integerVector(key).push_back( (int) atof( line.c_str() ) );
      } else if (kDoubleVector == type)
      {
         doubleVector(key).push_back( atof( line.c_str() ) );
      } else if (kStringVector == type)
      {
         stringVector(key).push_back( line );
      } else {
         // If we're here, then I made a coding mistake and want to
         // know about it.
         assert (0);
      }
   } // while getline
   return true;
   
}

void
optutl::getSectionFiles (const SVec &inputList, SVec &outputList,
                              int section, int totalSections)
{
   // Make sure the segment numbers make sense
   assert (section > 0 && section <= totalSections);

   // The Perl code:
   // my $entries    = @list;
   // my $perSection = int ($entries / $totalSections);
   // my $extra      = $entries % $totalSections;
   // --$section; # we want 0..n-1 not 1..n
   // my $start = $perSection * $section;
   // my $num   = $perSection - 1;
   // if ($section < $extra) {
   //    $start += $section;
   //    ++$num;
   // } else {
   //    $start += $extra;
   // };
   // my $end = $start + $num;
   int entries = (int) inputList.size();
   int perSection = entries / totalSections;
   int extra = entries % totalSections;
   --section; // we want 0..n-1, not 1..n.
   int current = perSection * section;
   int num = perSection - 1;
   if (section < extra)
   {
      current += section;
      ++num;
   } else 
   {
      current += extra;
   }
   outputList.clear();
   // we want to go from 0 to num inclusive, so make sure we have '<='
   // and not '='
   for (int loop = current; loop <= current + num; ++loop)
   {
      outputList.push_back( inputList.at( loop ) );
   } // for loop
}

void
optutl::setupDefaultOptions()
{
   // Integer options
   addOption ("totalSections", kInteger,
              "Total number of sections", 
              0);
   addOption ("section",       kInteger,
              "This section (from 1..totalSections inclusive)", 
              0);
   addOption ("maxEvents",     kInteger,
              "Maximum number of events to run over (0 for whole file)", 
              0);
   addOption ("jobID",         kInteger,
              "jobID given by CRAB,etc. (-1 means append nothing)", 
              -1);
   addOption ("outputEvery",   kInteger,
              "Output something once every N events (0 for never)", 
              0);
   // String options
   addOption ("outputFile",    kString,
              "Output filename", 
              "output.root");
   addOption ("storePrepend",  kString,
              "Prepend location on files starting with '/store/'");
   addOption ("tag",           kString,
              "A 'tag' to append to output file (e.g., 'v2', etc.");
   // Bool options
   addOption ("logName",       kBool,
              "Print log name and exit");
   // String vector options
   addOption ("inputFiles",    kStringVector,
              "List of input files");
}

void 
optutl::finishDefaultOptions (std::string tag)
{
   ////////////////////////
   // Deal with sections //
   ////////////////////////
   if ( integerValue ("totalSections") )
   {
      // we have a request to break this into sections.  First, note
      // this in the output file
      tag += Form ("_sec%03d", integerValue ("section"));
      SVec tempVec;
      getSectionFiles ( stringVector ("inputFiles"), 
                        tempVec,
                        integerValue ("section"),
                        integerValue ("totalSections") );
      stringVector ("inputFiles") = tempVec;  
   } // if section requested

   ///////////////////////
   // /Store lfn to pfn //
   ///////////////////////
   const string &kStorePrepend = stringValue ("storePrepend");
   if (kStorePrepend.length())
   {
      string match = "/store/";
      int matchLen = match.length();
      SVec tempVec;
      SVec &currentFiles = stringVector ("inputFiles");
      for (SVecConstIter iter = currentFiles.begin();
           currentFiles.end() != iter;
           ++iter)
      {
         const string &filename = *iter;
         if ((int) filename.length() > matchLen &&
             filename.substr(0, matchLen) == match)
         {
            tempVec.push_back( kStorePrepend + filename );
         } else {
            tempVec.push_back( filename);
         }
      }
      currentFiles = tempVec;
   } // if storePrepend.

   //////////////////////////////////
   // //////////////////////////// //
   // // Modify output filename // //
   // //////////////////////////// //
   //////////////////////////////////
   string outputFile = stringValue ("outputFile");
   outputFile = removeEnding (outputFile, ".root");
   outputFile += tag;
   if ( integerValue ("maxEvents") )
   {
      outputFile += Form ("_maxevt%03d", integerValue ("maxEvents"));
   }
   if ( integerValue ("jobID") >= 0)
   {
      outputFile += Form ("_jobID%03d", integerValue ("jobID"));
   }
   if ( stringValue ("tag").length() )
   {
      outputFile += "_" + stringValue ("tag");
   }

   /////////////////////////////////
   // Log File Name, if requested //
   /////////////////////////////////
   if ( boolValue ("logName") )
   {
      cout << outputFile << ".log" << endl;
      exit(0);
   }
   outputFile += ".root";
   stringValue ("outputFile") = outputFile;

   // finally, if they asked us to print all options, let's do so
   // after we've modified all variables appropriately
   if (ns_printOptions)
   {
      printOptionValues();
   } // if printOptions
}
