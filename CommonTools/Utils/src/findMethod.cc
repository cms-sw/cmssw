#include "CommonTools/Utils/src/findMethod.h"

#include "CommonTools/Utils/src/ErrorCodes.h"
#include "CommonTools/Utils/interface/Exception.h"
#include "FWCore/Utilities/interface/BaseWithDict.h"
#include "FWCore/Utilities/interface/TypeWithDict.h"
#include "FWCore/Utilities/interface/TypeID.h"

#include <cassert>

using AnyMethodArgument=reco::parser::AnyMethodArgument;

/// Checks for errors which show we got the correct function
/// but we cannot use it.
static
bool
fatalErrorCondition(const int err)
{
  return (err == reco::parser::kIsNotPublic) ||
         (err == reco::parser::kIsStatic) ||
         (err == reco::parser::kIsFunctionAddedByROOT) ||
         (err == reco::parser::kIsConstructor) ||
         (err == reco::parser::kIsDestructor) ||
         (err == reco::parser::kIsOperator);
}

namespace reco {

int
checkMethod(const edm::FunctionWithDict& mem,
            const edm::TypeWithDict& type,
            const std::vector<AnyMethodArgument>& args,
            std::vector<AnyMethodArgument>& fixuppedArgs)
{
  int casts = 0;
  if (mem.isConstructor()) {
    return -1 * parser::kIsConstructor;
  }
  if (mem.isDestructor()) {
    return -1 * parser::kIsDestructor;
  }
  // Some operators are allowed, e.g. operator[].
  //if (mem.isOperator()) {
  //  return -1 * parser::kIsOperator;
  //}
  if (!mem.isPublic()) {
    return -1 * parser::kIsNotPublic;
  }
  if (mem.isStatic()) {
    return -1 * parser::kIsStatic;
  }
  if (!mem.isConst()) {
    return -1 * parser::kIsNotConst;
  }
  if (mem.name().substr(0, 2) == "__") {
    return -1 * parser::kIsFunctionAddedByROOT;
  }
  // Functions from a base class are allowed.
  //if (mem.declaringType() != type) {
    //std::cerr <<
    //  "\nMETHOD OVERLOAD " <<
    //  mem.name() <<
    //  " by " <<
    //  type.qualifiedName() <<
    //  " from " <<
    //  mem.declaringType().qualifiedName() <<
    //  std::endl;
    //return -1 * parser::kOverloaded;
  //}

  size_t minArgs = mem.functionParameterSize(true);
  size_t maxArgs = mem.functionParameterSize(false);
  if ((args.size() < minArgs) || (args.size() > maxArgs)) {
    return -1 * parser::kWrongNumberOfArguments;
  }
  //std::cerr <<
  //  "\nMETHOD " <<
  //  mem.name() <<
  //  " of " <<
  //  mem.declaringType().name() <<
  //  ", min #args = " <<
  //  minArgs <<
  //  ", max #args = " <<
  //  maxArgs <<
  //  ", args = " <<
  //  args.size() <<
  //  std::endl;
  if (!args.empty()) {
    std::vector<AnyMethodArgument> tmpFixups;
    size_t i = 0;
    for (auto const& param : mem) {
      edm::TypeWithDict parameter(param);
      std::pair<AnyMethodArgument, int> fixup =
        boost::apply_visitor(reco::parser::AnyMethodArgumentFixup(parameter),
                             args[i]);
      //std::cerr <<
      //  "\t ARG " <<
      //  i <<
      //  " type is " <<
      //  parameter.name() <<
      //  " conversion = " <<
      //  fixup.second <<
      //  std::endl;
      if (fixup.second >= 0) {
        tmpFixups.push_back(fixup.first);
        casts += fixup.second;
      }
      else {
        return -1 * parser::kWrongArgumentType;
      }
      if (++i == args.size()) {
        break;
      }
    }
    fixuppedArgs.swap(tmpFixups);
  }
  //std::cerr <<
  //  "\nMETHOD " <<
  //  mem.name() <<
  //  " of " <<
  //  mem.declaringType().name() <<
  //  ", min #args = " <<
  //  minArgs <<
  //  ", max #args = " <<
  //  maxArgs <<
  //  ", args = " <<
  //  args.size() <<
  //  " fixupped args = " <<
  //  fixuppedArgs.size() <<
  //  "(" << casts <<
  //  " implicit casts)" <<
  //  std::endl;
  return casts;
}

typedef std::pair<int, edm::FunctionWithDict> OK;

bool
nCasts(const OK& a, const OK& b)
{
  return a.first < b.first;
}

std::pair<edm::FunctionWithDict, bool>
findMethod(const edm::TypeWithDict& t, /*class=in*/
           const std::string& name, /*function member name=in*/
           const std::vector<AnyMethodArgument>& args, /*args=in*/
           std::vector<AnyMethodArgument>& fixuppedArgs, /*args=out*/
           const char* iIterator, /*???=out*/
           int& oError) /*err code=out*/
{
  oError = parser::kNameDoesNotExist;
  edm::TypeWithDict type = t;
  if (!bool(type)) {
    throw parser::Exception(iIterator) << "No dictionary for class \"" <<
          type.name() << "\".";
  }
  while (type.isPointer() || type.isReference()) {
    type = type.toType();
  }
  while (type.isTypedef()) {
    edm::TypeWithDict theType = type.finalType();
    if(theType == type) {
      break;
    }
    type = theType;
  }
  // strip const, volatile, c++ ref, ..
  type = type.stripConstRef();
  // Create our return value.
  std::pair<edm::FunctionWithDict, bool> mem;
  //FIXME: We must initialize mem.first!
  mem.second = false;
  // suitable members and number of integer->real casts required to get them
  std::vector<std::pair<int, edm::FunctionWithDict> > oks;
  std::string theArgs;
  for(auto const& item : args) {
    if(!theArgs.empty()) {
      theArgs += ',';
    }
    theArgs += edm::TypeID(item.type()).className();
  }
  edm::FunctionWithDict f = type.functionMemberByName(name, theArgs, true);
  if(bool(f)) {
    int casts = checkMethod(f, type, args, fixuppedArgs);
    if (casts > -1) {
      oks.push_back(std::make_pair(casts, f));
    } else {
      oError = -1 * casts;
      //is this a show stopper error?
      if (fatalErrorCondition(oError)) {
        return mem;
      }
    }
  } else {
    edm::TypeFunctionMembers functions(type);
    for (auto const& F : functions) {
      edm::FunctionWithDict f(F);
      if (f.name() != name) {
        continue;
      }
      int casts = checkMethod(f, type, args, fixuppedArgs);
      if (casts > -1) {
        oks.push_back(std::make_pair(casts, f));
      } else {
        oError = -1 * casts;
        //is this a show stopper error?
        if (fatalErrorCondition(oError)) {
          return mem;
        }
      }
    }
  }
  //std::cout << "At base scope (type " << (type.name()) << ") found " <<
  //  oks.size() << " methods." << std::endl;
  // found at least one method
  if (!oks.empty()) {
    if (oks.size() > 1) {
      // sort by number of conversions needed
      sort(oks.begin(), oks.end(), nCasts);
      if (oks[0].first == oks[1].first) { // two methods with same ambiguity
        throw parser::Exception(iIterator) << "Can't resolve method \"" <<
              name << "\" for class \"" << type.name() <<
              "\", the two candidates " << oks[0].second.name() << " and " <<
              oks[1].second.name() <<
              " require the same number of integer->real conversions (" <<
              oks[0].first << ").";
      }
      // We must fixup the args again, as both good methods
      // have pushed them on fixuppedArgs.
      fixuppedArgs.clear();
      checkMethod(oks.front().second, type, args, fixuppedArgs);
    }
    mem.first = oks.front().second;
  }
  // if nothing was found, look in parent scopes without
  // checking for cross-scope overloading, as it is not
  // allowed
  int baseError = parser::kNameDoesNotExist;
  if (!bool(mem.first)) {
    edm::TypeBases bases(type);
    for (auto const& B : bases) {
      mem = findMethod(edm::BaseWithDict(B).typeOf(), name, args,
                       fixuppedArgs, iIterator, baseError);
      if (bool(mem.first)) {
        break;
      }
      if (fatalErrorCondition(baseError)) {
        oError = baseError;
        return mem;
      }
    }
  }
  // otherwise see if this object is just a Ref or Ptr and we should pop it out
  if (!bool(mem.first)) {
    // check for edm::Ref or edm::RefToBase or edm::Ptr
    if (type.isTemplateInstance()) {
      std::string name = type.templateName();
      if (!name.compare("edm::Ref") || !name.compare("edm::RefToBase") ||
          !name.compare("edm::Ptr")) {
        // in this case  i think 'get' should be taken with no arguments!
        std::vector<AnyMethodArgument> empty;
        std::vector<AnyMethodArgument> empty2;
        int error = 0;
        mem = findMethod(type, "get", empty, empty2, iIterator, error);
        if (!bool(mem.first)) {
          throw parser::Exception(iIterator) <<
                "No member \"get\" in reference of type \"" <<
                type.name() << "\".";
        }
        mem.second = true;
      }
    }
  }
  //if (!bool(mem.first)) {
  //  throw edm::Exception(edm::errors::Configuration) << "member \""" <<
  //        name << "\"" not found in class \""  << type.name() << "\"";
  //}
  if (bool(mem.first)) {
    oError = parser::kNoError;
  }
  else {
    // use error from base check if we never found function in primary class
    if (oError == parser::kNameDoesNotExist) {
      oError = baseError;
    }
  }
  return mem;
}

} // namespace reco

