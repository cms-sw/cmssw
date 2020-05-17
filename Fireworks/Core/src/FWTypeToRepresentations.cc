// -*- C++ -*-
//
// Package:     Core
// Class  :     FWTypeToRepresentations
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Tue Nov 11 14:09:01 EST 2008
//

// system include files

// user include files
#include "Fireworks/Core/interface/FWTypeToRepresentations.h"
#include "Fireworks/Core/interface/FWRepresentationCheckerBase.h"

//
// constants, enums and typedefs
//
typedef std::map<std::string, std::vector<FWRepresentationInfo> > TypeToReps;

//
// static data member definitions
//

//
// constructors and destructor
//
FWTypeToRepresentations::FWTypeToRepresentations() {}

// FWTypeToRepresentations::FWTypeToRepresentations(const FWTypeToRepresentations& rhs)
// {
//    // do actual copying here;
// }

FWTypeToRepresentations::~FWTypeToRepresentations() {}

//
// assignment operators
//
// const FWTypeToRepresentations& FWTypeToRepresentations::operator=(const FWTypeToRepresentations& rhs)
// {
//   //An exception safe implementation is
//   FWTypeToRepresentations temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void FWTypeToRepresentations::add(const std::shared_ptr<FWRepresentationCheckerBase>& iChecker) {
  m_checkers.push_back(iChecker);
  if (!m_typeToReps.empty()) {
    //see if this works with types we already know about
    for (auto& m_typeToRep : m_typeToReps) {
      FWRepresentationInfo info = iChecker->infoFor(m_typeToRep.first);
      if (info.isValid()) {
        //NOTE TO SELF: should probably sort by proximity
        m_typeToRep.second.push_back(info);
      }
    }
  }
}
void FWTypeToRepresentations::insert(const FWTypeToRepresentations& iOther) {
  m_typeToReps.clear();
  for (const auto& m_checker : iOther.m_checkers) {
    m_checkers.push_back(m_checker);
  }
}

//
// const member functions
//
const std::vector<FWRepresentationInfo>& FWTypeToRepresentations::representationsForType(
    const std::string& iTypeName) const {
  auto itFound = m_typeToReps.find(iTypeName);
  if (itFound == m_typeToReps.end()) {
    std::vector<FWRepresentationInfo> reps;
    //check all reps
    for (const auto& m_checker : m_checkers) {
      FWRepresentationInfo info = m_checker->infoFor(iTypeName);
      if (info.isValid())
        reps.push_back(info);
    }

    m_typeToReps.insert(std::make_pair(iTypeName, reps));
    itFound = m_typeToReps.find(iTypeName);
  }

  return itFound->second;
}

//
// static member functions
//
