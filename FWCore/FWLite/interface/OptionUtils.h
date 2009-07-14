// -*- C++ -*-

#if !defined(OptionUtils_H)
#define OptionUtils_H

#include <map>
#include <vector>
#include <string>

namespace optutl
{
   /////////////////////
   // /////////////// //
   // // Constants // //
   // /////////////// //
   /////////////////////

   // typedefs
   typedef std::vector< int >                   IVec;
   typedef std::vector< double >                DVec;
   typedef std::vector< std::string >           SVec;
   typedef std::map< std::string, int >         SIMap;
   typedef std::map< std::string, double >      SDMap;
   typedef std::map< std::string, bool >        SBMap;
   typedef std::map< std::string, std::string > SSMap;
   typedef std::map< std::string, IVec >        SIVecMap;
   typedef std::map< std::string, DVec >        SDVecMap;
   typedef std::map< std::string, SVec >        SSVecMap;
   // Iterators
   typedef IVec::iterator            IVecIter;
   typedef DVec::iterator            DVecIter;
   typedef SVec::iterator            SVecIter;
   typedef SIMap::iterator           SIMapIter;
   typedef SDMap::iterator           SDMapIter;
   typedef SBMap::iterator           SBMapIter;
   typedef SSMap::iterator           SSMapIter;
   typedef SIVecMap::iterator        SIVecMapIter;
   typedef SDVecMap::iterator        SDVecMapIter;
   typedef SSVecMap::iterator        SSVecMapIter;
   // constant iterators
   typedef IVec::const_iterator      IVecConstIter;
   typedef DVec::const_iterator      DVecConstIter;
   typedef SVec::const_iterator      SVecConstIter;
   typedef SIMap::const_iterator     SIMapConstIter;
   typedef SDMap::const_iterator     SDMapConstIter;
   typedef SBMap::const_iterator     SBMapConstIter;
   typedef SSMap::const_iterator     SSMapConstIter;
   typedef SIVecMap::const_iterator  SIVecMapConstIter;
   typedef SDVecMap::const_iterator  SDVecMapConstIter;
   typedef SSVecMap::const_iterator  SSVecMapConstIter;
   

   // constants
   const std::string kSpaces         = " \t";
   const int         kDefaultInteger = 0;
   const double      kDefaultDouble  = 0.;
   const std::string kDefaultString  = "";
   const bool        kDefaultBool    = false;
   const IVec        kEmptyIVec;
   const DVec        kEmptyDVec;
   const SVec        kEmptySVec;

   enum OptionType
   {
      kNone = 0,
      kInteger,
      kDouble,
      kString,
      kBool,
      kIntegerVector,
      kDoubleVector,
      kStringVector,
      kNumOptionTypes
   };

   ///////////////
   // Functions //
   ///////////////

   // parse the command line arguments.  If 'returnArgs' is true, then
   // any non-assignments and non-options will be returned.
   SVec parseArguments (int argc, char** argv, bool returnArgs = false);

   // set a usage string for '--help' option
   void setUsageString (const std::string &usage);

   // prints out '--help' screen, then exits.
   void help();

   // returns OptionType (or kNone (0)) of a given option.  
   OptionType hasOption (std::string key);

   // remove an ending (e.g., '.root') from a string
   std::string removeEnding (const std::string &input, 
                             const std::string &ending);   

   // splits a line into words
   void split (SVec &retval, std::string line, std::string match = " \t",
               bool ignoreComments = true);

   // removes '# ....' comment
   void removeComment (std::string &line);

   // removes leading and trailing spaces
   void removeLeadingAndTrailingSpaces (std::string &line);

   // given a line, finds first non-space word and rest of line
   void findCommand (const std::string &line,
                     std::string &command,
                     std::string &rest);

   // print out all of the variables hooked up
   void printOptionValues();

   // converts a string to lower case characters
   void lowercaseString(std::string &arg); 

   // converts a single character to lower case
   char toLower (char &ch);

   // Add variable to option maps.  'key' is passed in by copy because
   // it is modified in place.
   void addOption (std::string key, OptionType type,
                   const std::string &description = "");
   void addOption (std::string key, OptionType type,
                   const std::string &description, 
                   int defaultValue);
   void addOption (std::string key, OptionType type,
                   const std::string &description, 
                   double defaultValue);
   void addOption (std::string key, OptionType type,
                   const std::string &description, 
                   const std::string &defaultValue);
   void addOption (std::string key, OptionType type,
                   const std::string &description, 
                   const char *defaultValue);
   void addOption (std::string key, OptionType type,
                   const std::string &description, 
                   bool defaultValue);

   // some of the guts of above
   void _checkKey (std::string &key, const std::string &description);

   int         &integerValue  (std::string key);
   double      &doubleValue   (std::string key);
   std::string &stringValue   (std::string key);
   bool        &boolValue     (std::string key);
   IVec        &integerVector (std::string key);
   DVec        &doubleVector  (std::string key);
   SVec        &stringVector  (std::string key);

   // returns true if a variable has been modified from the command
   // line.
   bool valueHasBeenModified (const std::string &key);

   // Sets a variable 'varname' to a 'value' from a string
   // 'varname=value'.  If 'dontOverrideChange' is set true, then the
   // function will NOT set a variable that has been already set.
   // This will allow you to read in a file to set most variables and
   // still be allowed to make changes from the command line.  If you
   // want to ignore the first 'n' characters, simply set 'offset=n'.
   bool setVariableFromString (const std::string &arg,
                               bool dontOverrideChange = false,
                               int offset = 0);

   // sets variable options from file where lines are formatted as
   // - var=value
   bool setVariablesFromFile (const std::string &filename);

   // runs command embedded in arg
   bool runVariableCommandFromString (const std::string &arg);


   // given a section number (1..N) and totalsection (N), fills output
   // list with correct files.
   void getSectionFiles (const SVec &inputList, SVec &outputList,
                         int section, int totalSection);

   // sets up default optioons
   void setupDefaultOptions();

   // finish evaluating default options.  Pass in 'tag' if you want to
   // modify the output name based on options the user has passed in.
   // 'tag' is not passed by const reference because it may be
   // modified.
   void finishDefaultOptions (std::string tag = "");

   /////////////////////
   // /////////////// //
   // // Variables // //
   // /////////////// //
   /////////////////////

   extern SIMap     ns_integerMap;
   extern SDMap     ns_doubleMap;
   extern SSMap     ns_stringMap;
   extern SBMap     ns_boolMap;
   extern SIVecMap  ns_integerVecMap;
   extern SDVecMap  ns_doubleVecMap;
   extern SSVecMap  ns_stringVecMap;

   extern SBMap        ns_variableModifiedMap;
   extern SSMap        ns_variableDescriptionMap;
   extern std::string  ns_usageString;  
   extern std::string  ns_argv0;
   extern bool         ns_printOptions;

}

#endif // OptionUtils_H
