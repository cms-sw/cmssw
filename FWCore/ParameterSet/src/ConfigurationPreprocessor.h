#ifndef ParameterSet_ConfigurationPreprocessor_h
#define ParameterSet_ConfigurationPreprocessor_h

/** Applies preprocessing to a configuration string
    For now, just expands include statements
  */

#include <vector>
#include <string>

namespace edm {
  namespace pset {

    class ConfigurationPreprocessor 
    {
    public:
      ConfigurationPreprocessor() {}

      /** If 'input' is an include line, return true and put the name of the
          included file in filename. Otherwise, return false and leave
         filename untouched.
       */
      static bool is_include_line(std::string const& input,
                                  std::string& filename);

      /** Preprocess the given configuration text, modifying the given
        input.  Currently, preprocessing consists only of handling
        'include' statements.
       
        'include' statements must appear as the first non-whitespace on
        the line. They have the form:
        include "some/file/name" <additional stuff may follow>
       
        Read the input string, and write to the output string.
      */

      void process(const std::string & input, std::string & output);

      void clear();

    private:
      std::vector<std::string> openFiles_;
      std::vector<std::string> includedFiles_;
    };
  }
}

#endif

