#include "FWCore/Utilities/interface/TypeDemangler.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <algorithm>
#include <map>
#include <string>
#include <vector>
#include <iostream>
#include <cassert>
#include <ctype.h>
#include <stdlib.h>

/********************************************************************
	TypeDemangler is used to demangle the mangled name provided by type_info into the type name.

	The conventions used (e.g. const qualifiers before identifiers, no spaces after commas)
	are chosen to match the type names that can be used by the plug-in manager to load data dictionaries.
	It also strips comparators from (multi) maps or sets, and always strips allocators.

	This demangler is platform dependent.  This version works for gcc3.X.X and gcc4.X.X.

	There has been no attempt as of yet to factor out any parts of this code that may be platform independent.

	Known limitations:

	0) It is platform dependent. See above.

	1) It does not demangle function names, only type names.

	2) Because we must put const qualifiers before the identifier, some complex types
	involving both const qualifiers and pointers are not handled propery.
	The only such types that are handled properly are "Type const*", "Type const**", etc.
	Types not handled properly include "Type * const*", Type const * const. 

	3) If an enum value is used as a non-type template parameter, the demangled name cannot
	be used successfully to load the dictionary.  This is because the enumerator value name
	(used by Reflex) is not available in the mangled name (on this platform).

********************************************************************/
namespace edm {
  namespace {
    //  These are compile time constants, so there are no thread safety issues.
    char const typeCodes[] = { 'b', 'c', 'd', 'e', 'f', 'h', 'i', 'j', 'l', 'm', 's', 't', 'x', 'y'};
    char const* typeCodesEnd = typeCodes + sizeof(typeCodes)/sizeof(typeCodes[0]);
    char const* typeNames[] = {"bool", "char", "double", "long double", "float", "unsigned char", "int", "unsigned int",
     "long", "unsigned long", "short", "unsigned short", "long long", "unsigned long long"};
    char const* suffixes[] = {"", "", "", "", "", "", "", "u", "l", "ul", "", "", "ll", "ull"};

    bool isBuiltIn(char const c) {
	char const *p = std::lower_bound(typeCodes, typeCodesEnd, c);
	return (p != typeCodesEnd && *p == c);
    }

    char const* builtInTypeName(char const c) {
	char const *p = std::lower_bound(typeCodes, typeCodesEnd, c);
	if (p == typeCodesEnd || *p != c) return 0;
	return *(typeNames + (p - typeCodes));
    }

    char const* suffixString(char const c) {
	char const *p = std::lower_bound(typeCodes, typeCodesEnd, c);
	if (p == typeCodesEnd || *p != c) return 0;
	return *(suffixes + (p - typeCodes));
    }

    bool isBool(char c) {
      return (c == 'b');
    }

    enum TokenType {
      BuiltIn,        // Codes for a built-in type
      Digit,          // A decimal digit. Used most commonly for char count of an identifier
      Namespace,      // Code for a scope, such as a namespace or containing class.
      Template,	    // Code for the start of template parameters.
      Allocator,	    // Code for std::allocator
      String,	    // Code for std::basic_string<char>
      Std,	    // Code for a class or template in std::, other than string or allocator.
      End,	    // Code to terminate scopes or template parametes.
      Substitute,	    // Code to reuse a previously saved string.
      IndexTerminator,	    // Code to terminate the identification of the index for Substitute
      BuiltInValue,   // Code to indicate the value of a non-type template parameter.
      Pointer,	    // Code Pointer qualifier
      Reference,	    // Code Reference qualifier
      Const,	    // Code for const qualifier
      Volatile,	    // Code for volatile qualifier
      Unknown,	    // Unknown.  Should not get these.
      NoMore	    // No more chars in mangled name
    };
  
    struct Qualifiers {
      Qualifiers() : pointerDepth(0), isReference(false), cvIndex(0U), isConst(false), isVolatile(false) {}
      int pointerDepth;
      bool isReference;
      size_t cvIndex;
      bool isConst;
      bool isVolatile;
    };
  
    class TypeDemangler {
    public:
       TypeDemangler();
       void demangle(char const* name, std::string& demangledName);
    
    private:
      TokenType getNext();
      size_t getCount();
      size_t getSubstituteIndex();
      void doNamespace(bool silent);
      bool doIdentifier(bool silent);
      void copyIdentifier(size_t n, bool silent);
      void doStd(bool silent);
      bool doSubstitute(bool silent);
      void doQualifiers(Qualifiers& qualifiers, bool silent);
      void doBuiltIn(bool silent);
      void doBuiltInValue(bool silent);
      void doTemplate(bool silent, int maxOut = -1);
      void doType(bool silent);
      void doTypes(bool silent, int maxOut);
      void doAllocator();
      std::string result;
      size_t currentIndex;
      std::vector<std::string> substitutionStrings;
      std::vector<size_t> templateIndexes;
      size_t templateDepth;
      char const* mangled;
      char const* iterator;
      char const* itEnd;
    };
  
    TypeDemangler::TypeDemangler() :
	result(), currentIndex(0U),
	substitutionStrings(), templateIndexes(), templateDepth(0),
	mangled(0), iterator(0), itEnd(0) {
      substitutionStrings.reserve(16);
    }
  
    void
    TypeDemangler::demangle(char const* name, std::string& demangledName) {
      mangled = name;
      size_t size = strlen(name);
      iterator = name;
      itEnd = mangled + size;
      result.clear();
      result.reserve(size);
      doType(false);
      demangledName.swap(result);
      TokenType tok = getNext();
      if (tok != NoMore) {
	throw cms::Exception("Demangling error") << "demangle: Extra characters '" << iterator
	<< "' at end of mangled type name '" << mangled << "'\n";
      }
    }
  
    // General function to demangle one type name.
    // Calls the appropriate function.
    void
    TypeDemangler::doType(bool silent) {
      Qualifiers qualifiers;
      doQualifiers(qualifiers, silent);
      TokenType tok = getNext();
      if (tok == Digit) {
	doIdentifier(silent);
      } else if (tok == Substitute) {
	doSubstitute(silent);
      } else if (tok == Namespace) {
	doNamespace(silent);
      } else if (tok == BuiltIn) {
	doBuiltIn(silent);
      } else if (tok == BuiltInValue) {
	doBuiltInValue(silent);
      } else if (tok == Std) {
	doStd(silent);
      } else if (tok == Allocator) {
	doAllocator();
      } else if (tok == String) {
	iterator += 2;
	if (!silent) result += "std::basic_string<char>";
      } else {
	throw cms::Exception("Demangling error") << "doType: Unexpected character at '" << iterator
	<< "' in mangled type name '" << mangled << "'\n";
      }
      if (qualifiers.isConst || qualifiers.isVolatile) {
	substitutionStrings.push_back(result.substr(qualifiers.cvIndex));
      }
      if (!silent) {
	while (qualifiers.pointerDepth > 0) {
	  result += '*';
	  --qualifiers.pointerDepth;
	}
	if (qualifiers.isReference) {
	result += '&';
	}
      }
    }
  
    // Demangle a built in type (not enums, which are handled like class types).
    void
    TypeDemangler::doBuiltIn(bool silent) {
      TokenType tok = getNext();
      assert(tok == BuiltIn);
      if (!silent) result += builtInTypeName(*iterator);
      ++iterator;
    }
  
    // Demangle a non-type template parameter.
    void
    TypeDemangler::doBuiltInValue(bool silent) {
      ++iterator;
      TokenType tok = getNext();
      if (tok == Digit || tok == Substitute) {
	// The non-type template parameter is an enum value.
	if (!silent) result += '(';
	bool terminated = (tok == Digit ? doIdentifier(silent) : doSubstitute(silent));
	if (terminated) {
	  throw cms::Exception("Demangling error") << "doBuiltInValue: Unexpected characters at '" << iterator
	    << "' in mangled type name '" << mangled << "'\n";
	}
	if (!silent) result += ')';
	while(iterator < itEnd) {
	  tok = getNext();
	  if (tok == End) {
	   break;
	}
	if (!silent) result += *iterator;
	++iterator;
	}
	if (tok != End) {
	  throw cms::Exception("Demangling error") << "doBuiltInValue: Expected 'E' at '" << iterator
	  << "' in mangled type name '" << mangled << "'\n";
	}
	++iterator;
	return;
      } // End code for non-type template parameter is an enum value.
  
      // The non type prameter is a
      // (possibly cv, pointer, or reference qualified) simple built-in type
      Qualifiers qualifiers;
      doQualifiers(qualifiers, true);
      tok = getNext();
      if (tok != BuiltIn) {
	  throw cms::Exception("Demangling error") << "doBuiltInValue: Expected built in type code at '" << iterator
	  << "' in mangled type name '" << mangled << "'\n";
      }
      if (!silent && qualifiers.pointerDepth > 0) {
	// Add pointer qualifiers.  We throw away CV and reference qualifiers,
	result += '(';
	result += builtInTypeName(*iterator);
	while (qualifiers.pointerDepth > 0) {
	  result += '*';
	  --qualifiers.pointerDepth;
	}
	result += ')';
      }
      char const ctype = *iterator;
      ++iterator;
      if (iterator >= itEnd) {
	throw cms::Exception("Demangling error") << "doBuiltInValue: Ran off end of mangled type name '" << mangled << "'\n";
      }
      if (isBool(ctype)) {
	// Use "false" and "true" for bool.
	char const c = *iterator;
	if (c == '0') {
	  if (!silent) result += "false";
	} else {
	  if (!silent) result += "true";
	}
	++iterator;
	tok = getNext();
      } else {
	// For other built-ins, just copy the unmangled string until an 'End' is found.
	char const c = *iterator;
	if (!(isdigit(c) || c == '+' || c == '-')) {
	  throw cms::Exception("Demangling error") << "doBuiltInValue: Expected digit or sign at '" << iterator
	    << "' in mangled type name '" << mangled << "'\n";
	}
	while(iterator < itEnd) {
	  tok = getNext();
	  if (tok == End) {
	    break;
	  }
	  if (!silent) result += *iterator;
	  ++iterator;
	}
	if (!silent) result += suffixString(ctype);
      }
      if (tok != End) {
	throw cms::Exception("Demangling error") << "doBuiltInValue: Expected 'E' at '" << iterator
	  << "' in mangled type name '" << mangled << "'\n";
      }
      ++iterator;
    }
  
    // Demangle an identifier (e.g type name, namespace name).
    // If it is followed by template parameters, demangle those too.
    // Returns true if there were tamplate parameters, false otherwise.
    bool
    TypeDemangler::doIdentifier(bool silent) {
      size_t count = getCount();
      if (!count) {
	throw cms::Exception("Demangling error") << "doIdentifier: Expected non-zero count at '" << iterator
	  << "' in mangled type name '" << mangled << "'\n";
      }
      copyIdentifier(count, silent);
      TokenType tok = getNext();
      if (tok == Template) {
	doTemplate(silent);
	return true;
      }
      return false;
    }
  
    // Demangle a type in std::, other than strings or allocators, including any templare parameters.
    void
    TypeDemangler::doStd(bool silent) {
      iterator += 2;
      if (iterator > itEnd) {
	throw cms::Exception("Demangling error") << "doStd: Ran off end of mangled type name '" << mangled << "'\n";
      }
      size_t count = getCount();
      if (!count) {
	throw cms::Exception("Demangling error") << "doStd: Expected non-zero count at '" << iterator
	  << "' in mangled type name '" << mangled << "'\n";
      }
      if (!silent) result += "std::";
      size_t begin = result.size();
      copyIdentifier(count, silent);
      // Allocators are not output, but they are self identifying.
      // For (multi) maps and sets we need to avoid outputting the comparator.
      int maxParamsOutput = -1; //unlimited
      if (!silent) {
	char c = (result)[begin];
	if (c == 'm' || c == 's') { // Avoid string ops if no possibility of match
	  std::string id = result.substr(begin);
	  if (id == std::string("map") || id == std::string("multimap")) {
	    maxParamsOutput = 2;
	  } else if (id == std::string("set") || id == std::string("multiset")) {
	    maxParamsOutput = 1;
	  }
	}
      }
      // Now demangle its template parameters, if any.
      TokenType tok = getNext();
      if (tok == Template) {
	doTemplate(silent, maxParamsOutput);
      }
    }
  
    // Demangle std::Allocator and its template parameter.
    void
    TypeDemangler::doAllocator() {
      iterator += 2;
      if (iterator > itEnd) {
	throw cms::Exception("Demangling error") << "doAllocator: Ran off end of mangled type name '" << mangled << "'\n";
      }
      TokenType tok = getNext();
      assert(tok == Template);
      doTemplate(true);
    }
  
    // Demangle a substitution string (identified by a token representing a previously saved demangled string).
    // If it is followed by template parameters, demangle those too.
    // Returns true if there were tamplate parameters, false otherwise.
    bool
    TypeDemangler::doSubstitute(bool silent) {
      size_t ix = getSubstituteIndex();
      if (!silent) result += substitutionStrings.at(ix);
      TokenType tok = getNext();
      if (tok == Template) {
	doTemplate(silent);
	return true;
      }
      return false;
    }
  
    // Demangle CV and pointer qualifiers.
    // Due to the fact that we output CV qualifiers before the type name, we support only one level of CV.
    void
    TypeDemangler::doQualifiers(Qualifiers& qualifiers, bool silent) {
      TokenType tok = getNext();
      while (tok == Pointer || tok == Reference || tok == Const || tok == Volatile) {
	if (tok == Pointer) {
	  ++iterator;
	  ++qualifiers.pointerDepth;
	} else if (tok == Reference) {
	  ++iterator;
	  qualifiers.isReference = true;
	} else if (tok == Const) {
	  ++iterator;
	  if (!silent) result += "const ";
	  qualifiers.isConst = true;
	} else if (tok == Volatile) {
	  ++iterator;
	  if (!silent) result += "volatile ";
	  qualifiers.isVolatile = true;
	}
	if (iterator >= itEnd) {
	  throw cms::Exception("Demangling error") << "doQualifiers: Ran off end of mangled type name '" << mangled << "'\n";
	}
	tok = getNext();
      }
      if (qualifiers.isConst || qualifiers.isVolatile) {
	qualifiers.cvIndex = currentIndex;
	currentIndex = result.size();
      }
    }
 
    // Demangle a scoped (by namespaces and/or other classes) identifier
    // including all identifiers that are part of it, and any following template parameters.
    void
    TypeDemangler::doNamespace(bool silent) {
      ++iterator;
      bool hasTerminated = false;
      TokenType tok = getNext();
      if (tok == Digit) {
	doIdentifier(silent);
      } else if (tok == Substitute) {
	doSubstitute(silent);
      } else {
	throw cms::Exception("Demangling error") << "doNamespace: Unexpected character at '" << iterator
	  << "' in mangled type name '" << mangled << "'\n";
      }
      while (iterator < itEnd) {
	tok = getNext();
	if (tok == End) {
	  ++iterator;
	  return;
	}
	if (hasTerminated) {
	  throw cms::Exception("Demangling error") << "doNamespace: Unexpected character at '" << iterator
	    << "' in mangled type name '" << mangled << "'\n";
	}
	if (tok == Digit) {
	  if (!silent) result += "::";
	  hasTerminated = doIdentifier(silent);
	} else if (tok == Substitute) {
	  if (!silent) result += "::";
	  hasTerminated = doSubstitute(silent);
	} else {
	  throw cms::Exception("Demangling error") << "doNamespace: Unexpected character at '" << iterator
	    << "' in mangled type name '" << mangled << "'\n";
	}
      }
    }
  
    // Retrieves a decimal number.
    // This can be the number of chars in an identifier, a coding for a substitution index, or possibly other things.
    size_t
    TypeDemangler::getCount() {
      TokenType tok = getNext();
      assert(tok == Digit);
      char *endptr = 0;
      size_t count = strtol(iterator, &endptr, 0);
      iterator = endptr;
      return count;
    }
  
    // For a substitution, determines the index of the previously saved demangled string to substitute.
    size_t
    TypeDemangler::getSubstituteIndex() {
      TokenType tok = getNext();
      assert(tok == Substitute);
      ++iterator;
      tok = getNext();
      if (tok == IndexTerminator) {
	++iterator;
	return 0;
      }
      size_t ix = getCount() + 1;
      tok = getNext();
      assert(tok == IndexTerminator);
      ++iterator;
      return ix;
    }
  
    // Copies an identifier.
    void
    TypeDemangler::copyIdentifier(size_t count, bool silent) {
      if (iterator + count > itEnd) {
	throw cms::Exception("Demangling error") << "copyIdentifier: Ran off end of mangled type name '" << mangled << "'\n";
      }
      if (!silent) {
	char const* anonymousNamespace = "_GLOBAL__N_";
	int n = strlen(anonymousNamespace);
	if (strncmp(anonymousNamespace, iterator, n) == 0) {
	  // special case for an anonymous namespace.
	  result += "(anonymous namespace)";
	} else {
	  result.append(iterator, count);
	}
      }
      substitutionStrings.push_back(result.substr(currentIndex));
      iterator += count;
    }
  
    // Determines type of next token in mangled name.
    TokenType
    TypeDemangler::getNext() {
      if (iterator >= itEnd) {
	 return NoMore;
      }
      char const c = *iterator;
      if (c == 'E') {
	return End;
      }
      if (c == 'N') {
	return Namespace;
      }
      if (c == 'I') {
	return Template;
      }
      if (c == '_') {
	return IndexTerminator;
      }
      if (c == 'L') {
	return BuiltInValue;
      }
      if (c == 'P') {
	return Pointer;
      }
      if (c == 'R') {
	return Reference;
      }
      if (c == 'K') {
	return Const;
      }
      if (c == 'V') {
	return Volatile;
      }
      if (isdigit(c)) {
	return Digit;
      }
      if (isBuiltIn(c)) {
	return BuiltIn;
      }
      if (c == 'S') {
	++iterator;
	if (iterator >= itEnd) {
	  throw cms::Exception("Demangling error") << "getNext: Ran off end of mangled type name '" << mangled << "'\n";
	}
	char const c2 = *iterator;
	if (c2 == 't') {
	  --iterator;
	  return Std;
	}
	if (c2 == 's') {
	  --iterator;
	  return String;
	}
	if (c2 == 'a') {
	  --iterator;
	  return Allocator;
	}
	if (c2 == '_' || isdigit(c2)) {
	  --iterator;
	  return Substitute;
	}
	assert(0);;
      }
      return Unknown;
    }
  
    // General function to demangle multiple consecutive type names (i.e. template parameters).
    // maxOut limits the number of parameters output.  (-1 for no limit).
    void
    TypeDemangler::doTypes(bool silent, int maxOut) {
      doType(silent);
      if (--maxOut == 0) silent = true;
      while (iterator < itEnd) {
	TokenType tok = getNext();
	if (tok == End) {
	  ++iterator;
	  return;
	}
	if (tok == Allocator) silent = true;
	if (!silent && *(result).rbegin() != ' ') result += ',';
	currentIndex = result.size();
	doType(silent);
	if (--maxOut == 0) silent = true;
      }
    }
  
    // Demangles template parameters.
    // Calls doTypes for the parameters themselves.
    void
    TypeDemangler::doTemplate(bool silent, int maxOut) {
      assert(templateDepth <= templateIndexes.size());
      if (templateDepth == templateIndexes.size()) {
	templateIndexes.push_back(currentIndex);
      } else {
	templateIndexes[templateDepth] = currentIndex;
      }
      ++templateDepth;
      if (!silent) result += '<';
      currentIndex = result.size();
      ++iterator;
      doTypes(silent, maxOut);
      if (!silent) {
	if (*(result).rbegin() == '>') {
	  result += ' ';
	}
	result += '>';
      }
      --templateDepth;
      substitutionStrings.push_back(result.substr(templateIndexes[templateDepth]));
    }
  }
  
  void
  typeDemangle(char const* name, std::string& demangledName) {
    TypeDemangler t;
    t.demangle(name, demangledName);
  }
}
