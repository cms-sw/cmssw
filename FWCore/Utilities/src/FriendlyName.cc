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
#include "tbb/concurrent_unordered_map.h"

//NOTE:  This should probably be rewritten so that we break the class name into a tree where the template arguments are the node.  On the way down the tree
// we look for '<' or ',' and on the way up (caused by finding a '>') we can apply the transformation to the output string based on the class name for the
// templated class.  Up front we'd register a class name to a transformation function (which would probably take a std::vector<std::string> which holds
// the results of the node transformations)

namespace edm {
  namespace friendlyname {
    static std::regex const reBeginSpace("^ +");
    static std::regex const reEndSpace(" +$");
    static std::regex const reAllSpaces(" +");
    static std::regex const reColons("::");
    static std::regex const reComma(",");
    static std::regex const reTemplateArgs("[^<]*<(.*)>$");
    static std::regex const reTemplateClass("([^<>,]+<[^<>]*>)");
    static std::regex const rePointer("\\*");
    static std::string const emptyString("");

    std::string handleNamespaces(std::string const& iIn) {
       return std::regex_replace(iIn,reColons,emptyString);

    }

    std::string removeExtraSpaces(std::string const& iIn) {
       return std::regex_replace(std::regex_replace(iIn,reBeginSpace,emptyString),
                                    reEndSpace, emptyString);
    }

    std::string removeAllSpaces(std::string const& iIn) {
      return std::regex_replace(iIn, reAllSpaces,emptyString);
    }
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
    static std::regex const reSharedPtr("std::shared_ptr");
    static std::regex const reUniquePtr("std::unique_ptr");
    static std::regex const reAIKR(", *edm::helper::AssociationIdenticalKeyReference"); //this is a default so can replaced with empty
    //force first argument to also be the argument to edm::ClonePolicy so that if OwnVector is within
    // a template it will not eat all the remaining '>'s
    static std::regex const reOwnVector("edm::OwnVector<(.*), *edm::ClonePolicy<\\1 *> >");

    //NOTE: the '?' means make the smallest match. This may lead to problems where the template arguments themselves have commas
    // but we are using it in the cases where edm::AssociationMap appears multiple times in template arguments
    static std::regex const reOneToOne("edm::AssociationMap< *edm::OneToOne<(.*?),(.*?), *u[a-z]*> >");
    static std::regex const reOneToMany("edm::AssociationMap< *edm::OneToMany<(.*?),(.*?), *u[a-z]*> >");
    static std::regex const reOneToValue("edm::AssociationMap< *edm::OneToValue<(.*?),(.*?), *u[a-z]*> >");
    static std::regex const reOneToManyWithQuality("edm::AssociationMap<edm::OneToManyWithQuality<(.*?), *(.*?), *(.*?), *u[a-z]*> >");
    static std::regex const reToVector("edm::AssociationVector<(.*), *(.*), *edm::Ref.*,.*>");
    //NOTE: if the item within a clone policy is a template, this substitution will probably fail
    static std::regex const reToRangeMap("edm::RangeMap< *(.*), *(.*), *edm::ClonePolicy<([^>]*)> >");
    //NOTE: If container is a template with one argument which is its 'type' then can simplify name
    static std::regex const reToRefs1("edm::RefVector< *(.*)< *(.*) *>, *\\2 *, *edm::refhelper::FindUsingAdvance< *\\1< *\\2 *> *, *\\2 *> *>");
    static std::regex const reToRefs2("edm::RefVector< *(.*) *, *(.*) *, *edm::refhelper::FindUsingAdvance< *\\1, *\\2 *> *>");
    static std::regex const reToRefsAssoc("edm::RefVector< *Association(.*) *, *edm::helper(.*), *Association(.*)::Find>");
    
    
    std::string standardRenames(std::string const& iIn) {
       using std::regex_replace;
       using std::regex;
       std::string name = regex_replace(iIn, reWrapper, "$1");
       name = regex_replace(name,rePointer,"ptr");
       name = regex_replace(name,reAIKR,"");
       name = regex_replace(name,reclangabi,"std::");
       name = regex_replace(name,reCXX11,"std::");
       name = regex_replace(name,reString,"String");
       name = regex_replace(name,reString2,"String");
       name = regex_replace(name,reString3,"String");
       name = regex_replace(name,reSorted,"sSorted<$1>");
       name = regex_replace(name,reULongLong,"ull");
       name = regex_replace(name,reLongLong,"ll");
       name = regex_replace(name,reUnsigned,"u");
       name = regex_replace(name,reLong,"l");
       name = regex_replace(name,reVector,"s");
       name = regex_replace(name,reSharedPtr,"SharedPtr");
       name = regex_replace(name,reUniquePtr,"UniquePtr");
       name = regex_replace(name,reOwnVector,"sOwned<$1>");
       name = regex_replace(name,reToVector,"AssociationVector<$1,To,$2>");
       name = regex_replace(name,reOneToOne,"Association<$1,ToOne,$2>");
       name = regex_replace(name,reOneToMany,"Association<$1,ToMany,$2>");
       name = regex_replace(name,reOneToValue,"Association<$1,ToValue,$2>");
       name = regex_replace(name,reOneToManyWithQuality,"Association<$1,ToMany,$2,WithQuantity,$3>");
       name = regex_replace(name,reToRangeMap,"RangeMap<$1,$2>");
       name = regex_replace(name,reToRefs1,"Refs<$1<$2>>");
       name = regex_replace(name,reToRefs2,"Refs<$1,$2>");
       name = regex_replace(name,reToRefsAssoc,"Refs<Association$1>");
       //std::cout <<"standardRenames '"<<name<<"'"<<std::endl;
       return name;
    }

    std::string handleTemplateArguments(std::string const&);
    std::string subFriendlyName(std::string const& iFullName) {
       using namespace std;
       std::string result = removeExtraSpaces(iFullName);

       smatch theMatch;
       if(regex_match(result,theMatch,reTemplateArgs)) {
          //std::cout <<"found match \""<<theMatch.str(1) <<"\"" <<std::endl;
          //static regex const templateClosing(">$");
          //std::string aMatch = regex_replace(theMatch.str(1),templateClosing,"");
          std::string aMatch = theMatch.str(1);
          std::string theSub = handleTemplateArguments(aMatch);
          regex const eMatch(std::string("(^[^<]*)<")+aMatch+">");
          result = regex_replace(result,eMatch,theSub+"$1");
       }
       return result;
    }

    std::string handleTemplateArguments(std::string const& iIn) {
       using namespace std;
       std::string result = removeExtraSpaces(iIn);
       bool shouldStop = false;
       while(!shouldStop) {
          if(std::string::npos != result.find_first_of("<")) {
             smatch theMatch;
             if(regex_search(result,theMatch,reTemplateClass)) {
                std::string templateClass = theMatch.str(1);
                std::string friendlierName = removeAllSpaces(subFriendlyName(templateClass));
               
                //std::cout <<" t: "<<templateClass <<" f:"<<friendlierName<<std::endl;
                result = regex_replace(result, regex(templateClass),friendlierName);
             } else {
                //static regex const eComma(",");
                //result = regex_replace(result,eComma,"");
                std::cout <<" no template match for \""<<result<<"\""<<std::endl;
                assert(0 =="failed to find a match for template class");
             }
          } else {
             shouldStop=true;
          }
       }
       result = regex_replace(result,reComma,"");
       return result;
    }
    std::string friendlyName(std::string const& iFullName) {
       typedef tbb::concurrent_unordered_map<std::string, std::string> Map;
       static Map s_fillToFriendlyName;
       auto itFound = s_fillToFriendlyName.find(iFullName);
       if(s_fillToFriendlyName.end()==itFound) {
          itFound = s_fillToFriendlyName.insert(Map::value_type(iFullName, handleNamespaces(subFriendlyName(standardRenames(iFullName))))).first;
       }
       return itFound->second;
    }
  }
} // namespace edm

