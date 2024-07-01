/*
 *  friendlyName.cpp
 *  CMSSW
 *
 *  Created by Chris Jones on 2/24/06.
 *
 */
#include <string>
#include <regex>
#include <iostream>
#include <cassert>
#include "oneapi/tbb/concurrent_unordered_map.h"

//NOTE:  This should probably be rewritten so that we break the class name into a tree where the template arguments are the node.  On the way down the tree
// we look for '<' or ',' and on the way up (caused by finding a '>') we can apply the transformation to the output string based on the class name for the
// templated class.  Up front we'd register a class name to a transformation function (which would probably take a std::vector<std::string> which holds
// the results of the node transformations)

namespace {
  constexpr bool debug = false;
  std::string prefix;  // used only if debug == true
}  // namespace

namespace edm {
  namespace friendlyname {
    static std::regex const reBeginSpace("^ +");
    static std::regex const reEndSpace(" +$");
    static std::regex const reAllSpaces(" +");
    static std::regex const reColons("::");
    static std::regex const reComma(",");
    static std::regex const reTemplateArgs("[^<]*<(.*)>$");
    static std::regex const rePointer("\\*");
    static std::regex const reArray("\\[\\]");
    static std::regex const reUniquePtrDeleter("^std::unique_ptr< *(.*), *std::default_delete<\\1> *>");
    static std::regex const reUniquePtr("^std::unique_ptr");
    static std::string const emptyString("");

    std::string handleNamespaces(std::string const& iIn) { return std::regex_replace(iIn, reColons, emptyString); }

    std::string removeExtraSpaces(std::string const& iIn) {
      return std::regex_replace(std::regex_replace(iIn, reBeginSpace, emptyString), reEndSpace, emptyString);
    }

    std::string removeAllSpaces(std::string const& iIn) { return std::regex_replace(iIn, reAllSpaces, emptyString); }
    static std::regex const reWrapper("edm::Wrapper<(.*)>");
    static std::regex const reString("std::basic_string<char>");
    static std::regex const reString2("std::string");
    static std::regex const reString3("std::basic_string<char,std::char_traits<char> >");
    //The c++11 abi for gcc internally uses a different namespace for standard classes
    static std::regex const reCXX11("std::__cxx11::");
    static std::regex const reSorted("edm::SortedCollection<(.*), *edm::StrictWeakOrdering<\\1 *> >");
    static std::regex const reclangabi("std::__1::");
    static std::regex const reULongLong("ULong64_t");
    static std::regex const reLongLong("Long64_t");
    static std::regex const reUnsigned("unsigned ");
    static std::regex const reLong("long ");
    static std::regex const reVector("std::vector");
    static std::regex const reUnorderedSetHashKeyEqual(
        "std::unordered_set< *(.*), *std::hash<\\1> *, *std::equal_to<\\1> *>");
    static std::regex const reUnorderedSetCustomHashKeyEqual(
        "std::unordered_set< *(.*), *(.*) *, *std::equal_to<\\1> *>");
    static std::regex const reUnorderedSetHash("std::unordered_set< *(.*), *std::hash<\\1> *>");
    static std::regex const reUnorderedSet("std::unordered_set");
    static std::regex const reUnorderedMapHashKeyEqual(
        "std::unordered_map< *(.*), *(.*), *std::hash<\\1> *, *std::equal_to<\\1> *>");
    static std::regex const reUnorderedMapCustomHashKeyEqual(
        "std::unordered_map< *(.*), *(.*), *(.*) *, *std::equal_to<\\1> *>");
    static std::regex const reUnorderedMapHash("std::unordered_map< *(.*), *(.*), *std::hash<\\1> *>");
    static std::regex const reUnorderedMap("std::unordered_map");
    static std::regex const reSharedPtr("std::shared_ptr");
    static std::regex const reAIKR(
        ", *edm::helper::AssociationIdenticalKeyReference");  //this is a default so can replaced with empty
    //force first argument to also be the argument to edm::ClonePolicy so that if OwnVector is within
    // a template it will not eat all the remaining '>'s
    static std::regex const reOwnVector("edm::OwnVector<(.*), *edm::ClonePolicy<\\1 *> >");

    //NOTE: the '?' means make the smallest match. This may lead to problems where the template arguments themselves have commas
    // but we are using it in the cases where edm::AssociationMap appears multiple times in template arguments
    static std::regex const reOneToOne("edm::AssociationMap< *edm::OneToOne<(.*?),(.*?), *u[a-z]*> >");
    static std::regex const reOneToMany("edm::AssociationMap< *edm::OneToMany<(.*?),(.*?), *u[a-z]*> >");
    static std::regex const reOneToValue("edm::AssociationMap< *edm::OneToValue<(.*?),(.*?), *u[a-z]*> >");
    static std::regex const reOneToManyWithQuality(
        "edm::AssociationMap<edm::OneToManyWithQuality<(.*?), *(.*?), *(.*?), *u[a-z]*> >");
    static std::regex const reToVector("edm::AssociationVector<(.*), *(.*), *edm::Ref.*,.*>");
    //NOTE: if the item within a clone policy is a template, this substitution will probably fail
    static std::regex const reToRangeMap("edm::RangeMap< *(.*), *(.*), *edm::(Clone|Copy)Policy<([^>]*)> >");
    //NOTE: If container is a template with one argument which is its 'type' then can simplify name
    static std::regex const reToRefs1(
        "edm::RefVector< *(.*)< *(.*) *>, *\\2 *, *edm::refhelper::FindUsingAdvance< *\\1< *\\2 *> *, *\\2 *> *>");
    static std::regex const reToRefs2(
        "edm::RefVector< *(.*) *, *(.*) *, *edm::refhelper::FindUsingAdvance< *\\1, *\\2 *> *>");
    static std::regex const reToRefsAssoc(
        "edm::RefVector< *Association(.*) *, *edm::helper(.*), *Association(.*)::Find>");

    // type aliases for Alpaka internals
    static std::regex const reAlpakaDevCpu("alpaka::DevCpu");                                     // alpakaDevCpu
    static std::regex const reAlpakaDevCudaRt("alpaka::DevUniformCudaHipRt<alpaka::ApiCudaRt>");  // alpakaDevCudaRt
    static std::regex const reAlpakaDevHipRt("alpaka::DevUniformCudaHipRt<alpaka::ApiHipRt>");    // alpakaDevHipRt
    static std::regex const reAlpakaQueueCpuBlocking(
        "alpaka::QueueGenericThreadsBlocking<alpaka::DevCpu>");  // alpakaQueueCpuBlocking
    static std::regex const reAlpakaQueueCpuNonBlocking(
        "alpaka::QueueGenericThreadsNonBlocking<alpaka::DevCpu>");  // alpakaQueueCpuNonBlocking
    static std::regex const reAlpakaQueueCudaRtBlocking(
        "alpaka::uniform_cuda_hip::detail::QueueUniformCudaHipRt<alpaka::ApiCudaRt,true>");  // alpakaQueueCudaRtBlocking
    static std::regex const reAlpakaQueueCudaRtNonBlocking(
        "alpaka::uniform_cuda_hip::detail::QueueUniformCudaHipRt<alpaka::ApiCudaRt,false>");  // alpakaQueueCudaRtNonBlocking
    static std::regex const reAlpakaQueueHipRtBlocking(
        "alpaka::uniform_cuda_hip::detail::QueueUniformCudaHipRt<alpaka::ApiHipRt,true>");  // alpakaQueueHipRtBlocking
    static std::regex const reAlpakaQueueHipRtNonBlocking(
        "alpaka::uniform_cuda_hip::detail::QueueUniformCudaHipRt<alpaka::ApiHipRt,false>");  // alpakaQueueHipRtNonBlocking

    std::string standardRenames(std::string const& iIn) {
      using std::regex;
      using std::regex_replace;
      std::string name = regex_replace(iIn, reWrapper, "$1");
      name = regex_replace(name, rePointer, "ptr");
      name = regex_replace(name, reArray, "As");
      name = regex_replace(name, reAIKR, "");
      name = regex_replace(name, reclangabi, "std::");
      name = regex_replace(name, reCXX11, "std::");
      name = regex_replace(name, reString, "String");
      name = regex_replace(name, reString2, "String");
      name = regex_replace(name, reString3, "String");
      name = regex_replace(name, reSorted, "sSorted<$1>");
      name = regex_replace(name, reULongLong, "ull");
      name = regex_replace(name, reLongLong, "ll");
      name = regex_replace(name, reUnsigned, "u");
      name = regex_replace(name, reLong, "l");
      name = regex_replace(name, reVector, "s");
      name = regex_replace(name, reSharedPtr, "SharedPtr");
      name = regex_replace(name, reOwnVector, "sOwned<$1>");
      name = regex_replace(name, reToVector, "AssociationVector<$1,To,$2>");
      name = regex_replace(name, reOneToOne, "Association<$1,ToOne,$2>");
      name = regex_replace(name, reOneToMany, "Association<$1,ToMany,$2>");
      name = regex_replace(name, reOneToValue, "Association<$1,ToValue,$2>");
      name = regex_replace(name, reOneToManyWithQuality, "Association<$1,ToMany,$2,WithQuantity,$3>");
      name = regex_replace(name, reToRangeMap, "RangeMap<$1,$2>");
      name = regex_replace(name, reToRefs1, "Refs<$1<$2>>");
      name = regex_replace(name, reToRefs2, "Refs<$1,$2>");
      name = regex_replace(name, reToRefsAssoc, "Refs<Association$1>");

      // Alpaka types
      name = regex_replace(name, reAlpakaQueueCpuBlocking, "alpakaQueueCpuBlocking");
      name = regex_replace(name, reAlpakaQueueCpuNonBlocking, "alpakaQueueCpuNonBlocking");
      name = regex_replace(name, reAlpakaQueueCudaRtBlocking, "alpakaQueueCudaRtBlocking");
      name = regex_replace(name, reAlpakaQueueCudaRtNonBlocking, "alpakaQueueCudaRtNonBlocking");
      name = regex_replace(name, reAlpakaQueueHipRtBlocking, "alpakaQueueHipRtBlocking");
      name = regex_replace(name, reAlpakaQueueHipRtNonBlocking, "alpakaQueueHipRtNonBlocking");
      // devices should be last, as they can appear as template arguments in other types
      name = regex_replace(name, reAlpakaDevCpu, "alpakaDevCpu");
      name = regex_replace(name, reAlpakaDevCudaRt, "alpakaDevCudaRt");
      name = regex_replace(name, reAlpakaDevHipRt, "alpakaDevHipRt");

      if constexpr (debug) {
        std::cout << prefix << "standardRenames iIn " << iIn << " result " << name << std::endl;
      }
      return name;
    }

    std::string handleTemplateArguments(std::string const&);
    std::string subFriendlyName(std::string const& iFullName) {
      using namespace std;
      std::string result = removeExtraSpaces(iFullName);

      // temporarily remove leading const
      std::string leadingConst;
      if (std::string_view{result}.substr(0, 5) == "const") {
        leadingConst = "const";
        result = removeExtraSpaces(result.substr(5));
      }

      if constexpr (debug) {
        std::cout << prefix << "subFriendlyName iFullName " << iFullName << " result " << result << std::endl;
      }
      // Handle unique_ptr, which may contain the deleter (but handle only std::default_delete)
      {
        auto result2 =
            regex_replace(result, reUniquePtrDeleter, "UniquePtr<$1>", std::regex_constants::format_first_only);
        if (result2 == result) {
          result2 = regex_replace(result, reUniquePtr, "UniquePtr", std::regex_constants::format_first_only);
        }
        result = std::move(result2);
      }
      // insert the leading const back if it was there
      result = leadingConst + result;
      // Handle unordered_set, which may contain a hash and an an equal for the key
      {
        auto result2 =
            regex_replace(result, reUnorderedSetHashKeyEqual, "stduset<$1>", std::regex_constants::format_first_only);
        if (result2 == result) {
          result2 = regex_replace(
              result, reUnorderedSetCustomHashKeyEqual, "stduset<$1, $2>", std::regex_constants::format_first_only);
        }
        if (result2 == result) {
          result2 = regex_replace(result, reUnorderedSetHash, "stduset<$1>", std::regex_constants::format_first_only);
        }
        if (result2 == result) {
          result2 = regex_replace(result, reUnorderedSet, "stduset", std::regex_constants::format_first_only);
        }
        result = std::move(result2);
      }
      // Handle unordered_map, which may contain a hash and an an equal for the key
      {
        auto result2 = regex_replace(
            result, reUnorderedMapHashKeyEqual, "stdumap<$1, $2>", std::regex_constants::format_first_only);
        if (result2 == result) {
          result2 = regex_replace(
              result, reUnorderedMapCustomHashKeyEqual, "stdumap<$1, $2, $3>", std::regex_constants::format_first_only);
        }
        if (result2 == result) {
          result2 =
              regex_replace(result, reUnorderedMapHash, "stdumap<$1, $2>", std::regex_constants::format_first_only);
        }
        if (result2 == result) {
          result2 = regex_replace(result, reUnorderedMap, "stdumap", std::regex_constants::format_first_only);
        }
        result = std::move(result2);
      }
      if (smatch theMatch; regex_match(result, theMatch, reTemplateArgs)) {
        //std::cout <<"found match \""<<theMatch.str(1) <<"\"" <<std::endl;
        //static regex const templateClosing(">$");
        //std::string aMatch = regex_replace(theMatch.str(1),templateClosing,"");
        std::string aMatch = theMatch.str(1);
        if constexpr (debug) {
          prefix += "  ";
        }
        std::string theSub = handleTemplateArguments(aMatch);
        if constexpr (debug) {
          prefix.pop_back();
          prefix.pop_back();
          std::cout << prefix << " aMatch " << aMatch << " theSub " << theSub << std::endl;
        }
        regex const eMatch(std::string("(^[^<]*)<") + aMatch + ">");
        result = regex_replace(result, eMatch, theSub + "$1");
      }
      return removeAllSpaces(result);
    }

    std::string handleTemplateArguments(std::string const& iIn) {
      using namespace std;
      std::string result = removeExtraSpaces(iIn);
      if constexpr (debug) {
        std::cout << prefix << "handleTemplateArguments " << iIn << " removeExtraSpaces " << result << std::endl;
      }

      // Trick to have every full class name to end with comma to
      // avoid treating the end as a special case
      result += ",";

      std::string result2;
      result2.reserve(iIn.size());
      unsigned int openTemplate = 0;
      bool hadTemplate = false;
      size_t begin = 0;
      for (size_t i = 0, size = result.size(); i < size; ++i) {
        if (result[i] == '<') {
          ++openTemplate;
          hadTemplate = true;
          continue;
        } else if (result[i] == '>') {
          --openTemplate;
        }
        // If we are not within the template arguments of a class template
        // - encountering comma means that we are within a template
        //   argument of some other class template, and we've reached
        //   a point when we should translate the argument class name
        // - encountering colon, but only if the class name so far
        //   itself was a template, we've reached a point when we
        //   should translate the class name
        if (const bool hasComma = result[i] == ',', hasColon = hadTemplate and result[i] == ':';
            openTemplate == 0 and (hasComma or hasColon)) {
          std::string templateClass = result.substr(begin, i - begin);
          if constexpr (debug) {
            std::cout << prefix << " templateClass " << templateClass << std::endl;
          }
          if (hadTemplate) {
            if constexpr (debug) {
              prefix += "  ";
            }
            std::string friendlierName = subFriendlyName(templateClass);
            if constexpr (debug) {
              prefix.pop_back();
              prefix.pop_back();
              std::cout << prefix << " friendlierName " << friendlierName << std::endl;
            }
            result2 += friendlierName;
          } else {
            result2 += templateClass;
          }
          if constexpr (debug) {
            std::cout << prefix << " result2 " << result2 << std::endl;
          }
          // reset counters
          hadTemplate = false;
          begin = i + 1;
          // With colon we need to eat the second colon as well
          if (hasColon) {
            assert(result[begin] == ':');
            ++begin;
          }
        }
      }

      result = regex_replace(result2, reComma, "");
      if constexpr (debug) {
        std::cout << prefix << " reComma " << result << std::endl;
      }
      return result;
    }
    std::string friendlyName(std::string const& iFullName) {
      if constexpr (debug) {
        std::cout << "\nfriendlyName for " << iFullName << std::endl;
        prefix = " ";
      }
      typedef oneapi::tbb::concurrent_unordered_map<std::string, std::string> Map;
      static Map s_fillToFriendlyName;
      auto itFound = s_fillToFriendlyName.find(iFullName);
      if (s_fillToFriendlyName.end() == itFound) {
        itFound = s_fillToFriendlyName
                      .insert(Map::value_type(iFullName, handleNamespaces(subFriendlyName(standardRenames(iFullName)))))
                      .first;
      }
      if constexpr (debug) {
        std::cout << "result " << itFound->second << std::endl;
      }
      return itFound->second;
    }
  }  // namespace friendlyname
}  // namespace edm
