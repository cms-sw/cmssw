#include "FWCore/ParameterSet/src/ConfigurationPreprocessor.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/parse.h"
#include <sstream>
#include "boost/regex.hpp"

#include <iostream>


namespace edm 
{
  namespace pset 
  {
   
    // If 'input' is an include line, return true and put the name of the
    // included file in filename. Otherwise, return false and leave
    // filename untouched.

    bool ConfigurationPreprocessor::is_include_line(std::string const& input,
                         std::string& filename)
    {
      // possible whitespace at start of line,
      // followed by the string 'include'
      // followed by some whitespace
      //    followed by double-quote filename double-quote (pattern 1)
      //    followed by single-quote filename single-quote (pattern 2)
      // followed by anything
      //  * capture group 1 is the filename, unquoted
      //  * capture group 2 is the following 'anything'
      const boost::regex pattern_dq("^[ \t]*include[ \t]+\"([^\"]+)\"(.*)");
      const boost::regex pattern_sq("^[ \t]*include[ \t]+'([^']+)'(.*)");

      // Check patterns for a match.
      boost::smatch m;
      bool foundMatch =
        boost::regex_search(input, m, pattern_dq) ||
        boost::regex_search(input, m, pattern_sq);

      if (!foundMatch)
        {
          return false;
        }
      // If we got here, we've found a match.

      // Make sure the following 'anything' is either whitespace, or a
      // comment. Comments are:
      //   optional whitespace (space or tab)
      //
      const::boost::regex trailing_whitespace("[ \t]*");
      const::boost::regex comment_pattern_1("[ \t]*#");
      const::boost::regex comment_pattern_2("[ \t]*//");

      std::string maybe_filename = m[1];
      std::string trailing_text  = m[2];

      bool good =  trailing_text.empty()
        ||         boost::regex_search(trailing_text, m, trailing_whitespace)
        ||         boost::regex_search(trailing_text, m, comment_pattern_1)
        ||         boost::regex_search(trailing_text, m, comment_pattern_2);

      if (! good) // trailing text is illegal
        {
          throw edm::Exception(edm::errors::Configuration, "BadInclude")
            <<  "Found trailing non-comment text in line: "
            << input << '\n';
        }
      filename = maybe_filename;
      return true;
    }


    void ConfigurationPreprocessor::process(const std::string & input, std::string & output)
    {
      std::istringstream in(input);
      std::string line;

      // For each line in the input...
      while (std::getline(in, line))
      {
        std::string filename;
        // If we're told to include a file...
        if (this->is_include_line(line, filename))
        {
          // Translate the file name. FileInPath makes sure the
          // file exists. But read_whole_file checks again
          // anyway. This may no be necessary; the chance of the
          // file disappearing betwen this two calls is small.
          FileInPath realfilename(filename);

          // Make sure we don't have a circular inclusion.
          if (std::find(openFiles_.begin(),
                         openFiles_.end(),
                         realfilename.fullPath())
               != openFiles_.end())
          {
            throw edm::Exception(edm::errors::Configuration, "CircularInclude")
              << "The configuration file (or configuration fragment) file: "
              << realfilename.fullPath()
              << " circularly includes itself";
          }

          // also make sure it hasn't been included more than once
          if (std::find(includedFiles_.begin(),
                         includedFiles_.end(),
                         realfilename.fullPath())
               == includedFiles_.end())
          {
            openFiles_.push_back(realfilename.fullPath());
            includedFiles_.push_back(realfilename.fullPath());
            // ... process the file we're to include.
            std::string filecontents;
  
            if (!read_whole_file(realfilename.fullPath(),
                             filecontents))
            {
              throw edm::Exception(edm::errors::Configuration, "MissingFile")
                << "Could not find configuration include file:"
                << filename;
            }
            // recursive call!
            process(filecontents, output);
            openFiles_.pop_back();
          } // not multiply included
         
else {
  std::cout << "DOUBL EINCLUDE IGNORE" << std::endl;
}
        }
        else // not including a file
        {
          // just append the text to the output. Remember that
          // getline has stripped a newline, so we have to
          // re-insert it.
          output += line;
          output += '\n';
        }
      }
    } // process()

  } // pset namespace
}

