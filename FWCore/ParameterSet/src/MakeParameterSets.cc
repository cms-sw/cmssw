#include <fstream>
#include <istream>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

#include "boost/regex.hpp"
#include "boost/shared_ptr.hpp"

#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/ParameterSet/interface/MakeParameterSets.h"
#include "FWCore/ParameterSet/interface/ProcessPSetBuilder.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

using boost::shared_ptr;
using std::string;
using std::stringstream;
using std::vector;

namespace edm
{

  namespace pset
  {

    bool read_whole_file(std::string const& filename,
			 std::string& output)
    {
      std::ifstream input(filename.c_str());
      if (!input) return false;
      std::string buffer;
      while ( std::getline(input, buffer) )  
	{
	  // getline strips newlines; we have to put them back by hand.
	  output += buffer;
	  output += '\n';
	}
      return true;
    }

    // If 'input' is an include line, return true and put the name of the
    // included file in filename. Otherwise, return false and leave
    // filename untouched.

    bool is_include_line(std::string const& input,
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
    
      if ( !foundMatch )
	{
	  std::cerr << "No match found: '" << input << "'\n";
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

      if ( ! good ) // trailing text is illegal
	{
	  std::cerr << "Found trailing garbage in line: " << input << '\n';
	  throw 3;
	}
  
      std::cout << "is_include_line found a filename: '" 
		<< maybe_filename 
		<< "'  and the rest of the line is: '"
		<< trailing_text
		<< "'"
		<< std::endl;
  
      filename = maybe_filename;
      return true;
    }


    // Preprocess the given configuration text, modifying the given
    // input.  Currently, preprocessing consists only of handling
    // 'include' statements.

    // 'include' statements must appear as the first non-whitespace on
    // the line. They have the form:
    //    include "some/file/name" <additional stuff may follow>

    // Read the input string, and write to the output string.
    void preprocessConfigString(std::string const& input,
				std::string& output,
				std::vector<std::string>& openFiles)
    {
      std::istringstream in(input);
      std::string line;

      // For each line in the input...
      while ( std::getline(in, line) )
	{
	  std::string filename;
	  // If we're told to include a file...
	  if ( is_include_line(line, filename) )
	    {
	      // Translate the file name. FileInPath makes sure the
	      // file exists. But read_whole_file checks again
	      // anyway. This may no be necessary; the chance of the
	      // file disappearing betwen this two calls is small.
	      FileInPath realfilename(filename);

	      // Make sure we don't have a circular inclusion.
	      if ( std::find(openFiles.begin(), 
			     openFiles.end(), 
			     realfilename.fullPath())
		   != openFiles.end() )
		{
		  throw edm::Exception(edm::errors::Configuration, "CircularInclude")
		    << "The configuration file (or configuration fragment) file: " 
		    << realfilename.fullPath()
		    << " circularly includes itself";		    
		}
	      openFiles.push_back(realfilename.fullPath());

	      // ... process the file we're to include.
	      std::string filecontents;
	      
	      if (!read_whole_file(realfilename.fullPath(),
				   filecontents))
		{
		  throw edm::Exception(edm::errors::Configuration, "MissingFile")
		    << "Could not find configuration include file:"
		    << filename;				       
		}
	      preprocessConfigString(filecontents, output, openFiles);
	      openFiles.pop_back();
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
    } //preprocessConfigString

  } // namespace pset

  void
  makeParameterSets(string const& configtext,
		    shared_ptr<ParameterSet>& main,
		    shared_ptr<vector<ParameterSet> >& serviceparams)
  {
    // Handle 'include' statements in a pre-processing step. This this
    // becomes complicated, we may need a class to do this. Right now,
    // it is simple enough that we need no such class.

    string finalConfigDoc;
    vector<string> openFileStack;
    pset::preprocessConfigString(configtext, finalConfigDoc, openFileStack);
    edm::ProcessPSetBuilder builder(finalConfigDoc);

    main = builder.getProcessPSet();
    serviceparams = builder.getServicesPSets();

    // NOTE: FIX WHEN POOL BUG IS FIXED.
    // For now, we have to always make use of the "LoadAllDictionaries" service.
    serviceparams->push_back(ParameterSet());
    serviceparams->back().addParameter<std::string>("@service_type", "LoadAllDictionaries");
  }
} // namespace edm
