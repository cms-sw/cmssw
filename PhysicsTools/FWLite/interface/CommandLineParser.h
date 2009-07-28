// -*- C++ -*-

#if !defined(CommandLineParser_H)
#define CommandLineParser_H

#include "PhysicsTools/FWLite/interface/VariableMapCont.h"

namespace optutl
{

class CommandLineParser : public VariableMapCont
{
   public:
      //////////////////////
      // Public Constants //
      //////////////////////

      static const std::string kSpaces;

      enum
      {
         kEventContOpt = 1 << 0
      };

      /////////////
      // friends //
      /////////////
      // tells particle data how to print itself out
      friend std::ostream& operator<< (std::ostream& o_stream, 
                                       const CommandLineParser &rhs);

      //////////////////////////
      //            _         //
      // |\/|      |_         //
      // |  |EMBER | UNCTIONS //
      //                      //
      //////////////////////////

      /////////////////////////////////
      // Constructors and Destructor //
      /////////////////////////////////
      CommandLineParser (const std::string &usage, 
                         unsigned int optionsType = kEventContOpt);

      ////////////////
      // One Liners //
      ////////////////

      // turn on (true) or off (false) printing of options by default
      void setPrintOptoins (bool print) { m_printOptions = print; }

      // return vector calling arguments
      const SVec argVec() const { return m_fullArgVec; }

      //////////////////////////////
      // Regular Member Functions //
      //////////////////////////////

      // parse the command line arguments.  If 'returnArgs' is true, then
      // any non-assignments and non-options will be returned.
      void parseArguments (int argc, char** argv, bool allowArgs = false);

      // prints out '--help' screen, then exits.
      void help();

      // print out all of the variables hooked up
      void printOptionValues();

      // Not called by users anymore.  Finish evaluating default options.
      // Pass in 'tag' if you want to modify the output name based on
      // options the user has passed in.  'tag' is not passed by const
      // reference because it may be modified.
      void _finishDefaultOptions (std::string tag = "");

      /////////////////////////////
      // Static Member Functions //
      /////////////////////////////

      // remove an ending (e.g., '.root') from a string
      static std::string removeEnding (const std::string &input, 
                                       const std::string &ending);   

      // splits a line into words
      static void split (SVec &retval, std::string line, 
                         std::string match = " \t",
                         bool ignoreComments = true);

      // removes '# ....' comment
      static void removeComment (std::string &line);

      // removes leading and trailing spaces
      static void removeLeadingAndTrailingSpaces (std::string &line);

      // given a line, finds first non-space word and rest of line
      static void findCommand (const std::string &line,
                               std::string &command,
                               std::string &rest);


  private:
      //////////////////////////////
      // Private Member Functions //
      //////////////////////////////

      // Sets a variable 'varname' to a 'value' from a string
      // 'varname=value'.  If 'dontOverrideChange' is set true, then the
      // function will NOT set a variable that has been already set.
      // This will allow you to read in a file to set most variables and
      // still be allowed to make changes from the command line.  If you
      // want to ignore the first 'n' characters, simply set 'offset=n'.
      bool _setVariableFromString (const std::string &arg,
                                   bool dontOverrideChange = false,
                                   int offset = 0);

      // sets variable options from file where lines are formatted as
      // - var=value
      bool _setVariablesFromFile (const std::string &filename);
      
      // runs command embedded in arg
      bool _runVariableCommandFromString (const std::string &arg);


      // given a section number (1..N) and totalsection (N), fills output
      // list with correct files.
      void _getSectionFiles (const SVec &inputList, SVec &outputList,
                             int section, int totalSection);

      /////////////////////////
      // Private Member Data //
      /////////////////////////
      SVec         m_fullArgVec;
      std::string  m_argv0;
      std::string  m_usageString;  
      bool         m_printOptions;
      unsigned int m_optionsType;

};

}

#endif // CommandLineParser_H
