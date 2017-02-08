#include <cxxabi.h>
#include <cctype>
#include <string>
#include <regex>
#include "FWCore/Utilities/interface/Exception.h"

/********************************************************************
        TypeDemangler is used to demangle the mangled name provided by type_info into the type name.

        The conventions used (e.g. const qualifiers before identifiers, no spaces after commas)
        are chosen to match the type names that can be used by the plug-in manager to load data dictionaries.
        It also strips comparators from (multi) maps or sets, and always strips allocators.

        This demangler is platform dependent.  This version works for gcc.

        Known limitations:

        0) It is platform dependent. See above.

        1) It does not demangle function names, only type names.

        2) If an enum value is used as a non-type template parameter, the demangled name cannot
        be used successfully to load the dictionary.  This is because the enumerator value name
        is not available in the mangled name (on this platform).

********************************************************************/
namespace {
  void
  reformatter(std::string& input, char const* exp, char const* format) {
    std::regex regexp(exp, std::regex::egrep);
    while(std::regex_match(input, regexp)) {
      std::string newstring = std::regex_replace(input, regexp, format);
      input.swap(newstring);
    }
  }

  void
  removeParameter(std::string& demangledName, std::string const& toRemove) {
    std::string::size_type const asize = toRemove.size();
    char const* const delimiters = "<>";
    std::string::size_type index = std::string::npos;
    while ((index = demangledName.find(toRemove)) != std::string::npos) {
      int depth = 1;
      std::string::size_type inx = index + asize;
      while ((inx = demangledName.find_first_of(delimiters, inx)) != std::string::npos) {
        if (demangledName[inx] == '<') {
          ++depth;
        } else {
          --depth;
          if (depth == 0) {
            demangledName.erase(index, inx + 1 - index);
            if (demangledName[index] == ' ' && (index == 0 || demangledName[index - 1] != '>')) {
              demangledName.erase(index, 1);
            }
            break;
          }
        }
        ++inx;
      }
    } 
  }

  void
  constBeforeIdentifier(std::string& demangledName) {
    std::string const toBeMoved(" const");
    std::string::size_type const asize = toBeMoved.size();
    std::string::size_type index = std::string::npos;
    while ((index = demangledName.find(toBeMoved)) != std::string::npos) {
      demangledName.erase(index, asize);
      int depth = 0;
      for (std::string::size_type inx = index - 1; inx > 0; --inx) {
        char const c = demangledName[inx];
        if (c == '>') {
          ++depth;
        } else if (depth > 0) {
          if (c == '<') --depth;
        } else if (c == '<' || c == ',') {
          demangledName.insert(inx + 1, "const ");
          break;
        }
      }
    }
  }
}

namespace edm {
  void
  replaceString(std::string& demangledName, std::string const& from, std::string const& to) {
    // from must not be a substring of to.
    std::string::size_type length = from.size(); 
    std::string::size_type pos = 0;
    while((pos = demangledName.find(from, pos)) != std::string::npos) {
       demangledName.replace(pos, length, to); 
    }
  }

  std::string
  typeDemangle(char const* mangledName) {
    int status = 0;
    size_t* const nullSize = 0;
    char* const null = 0;
    
    // The demangled C style string is allocated with malloc, so it must be deleted with free().
    char* demangled = abi::__cxa_demangle(mangledName, null, nullSize, &status);
    if (status != 0) {
      throw cms::Exception("Demangling error") << " '" << mangledName << "'\n";
    } 
    std::string demangledName(demangled);
    free(demangled);
    // We must use the same conventions previously used by REFLEX.
    // The order of these is important.
    // No space after comma
    replaceString(demangledName, ", ", ",");
    // No space before opening square bracket
    replaceString(demangledName, " [", "[");
    // Strip default allocator
    std::string const allocator(",std::allocator<");
    removeParameter(demangledName, allocator);
    // Strip default comparator
    std::string const comparator(",std::less<");
    removeParameter(demangledName, comparator);
    // Put const qualifier before identifier.
    constBeforeIdentifier(demangledName);
    // No two consecutive '>' 
    replaceString(demangledName, ">>", "> >");
    // No u or l qualifiers for integers.
    reformatter(demangledName, "(.*[<,][0-9]+)[ul]l*([,>].*)", "$1$2");
    // For ROOT 6 and beyond, replace 'unsigned long long' with 'ULong64_t'
    replaceString(demangledName, "unsigned long long", "ULong64_t");
    // For ROOT 6 and beyond, replace 'long long' with 'Long64_t'
    replaceString(demangledName, "long long", "Long64_t");
    return demangledName;
  }
}
