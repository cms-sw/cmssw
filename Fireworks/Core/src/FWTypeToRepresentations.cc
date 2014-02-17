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
// $Id: FWTypeToRepresentations.cc,v 1.2 2009/01/23 21:35:44 amraktad Exp $
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
FWTypeToRepresentations::FWTypeToRepresentations()
{
}

// FWTypeToRepresentations::FWTypeToRepresentations(const FWTypeToRepresentations& rhs)
// {
//    // do actual copying here;
// }

FWTypeToRepresentations::~FWTypeToRepresentations()
{
}

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
void
FWTypeToRepresentations::add( boost::shared_ptr<FWRepresentationCheckerBase> iChecker)
{
   m_checkers.push_back(iChecker);
   if(m_typeToReps.size()) {
      //see if this works with types we already know about
      for(TypeToReps::iterator it = m_typeToReps.begin(), itEnd = m_typeToReps.end();
          it != itEnd;
          ++it) {
         FWRepresentationInfo info = iChecker->infoFor(it->first);
         if(info.isValid()) {
            //NOTE TO SELF: should probably sort by proximity
            it->second.push_back(info);
         }
      }
   }
}
void
FWTypeToRepresentations::insert( const FWTypeToRepresentations& iOther)
{
   m_typeToReps.clear();
   for(std::vector<boost::shared_ptr<FWRepresentationCheckerBase> >::const_iterator it =iOther.m_checkers.begin(),
                                                                                    itEnd = iOther.m_checkers.end();
       it != itEnd;
       ++it) {
      m_checkers.push_back(*it);
   }
}

//
// const member functions
//
const std::vector<FWRepresentationInfo>&
FWTypeToRepresentations::representationsForType(const std::string& iTypeName) const
{
   TypeToReps::const_iterator itFound = m_typeToReps.find(iTypeName);
   if(itFound == m_typeToReps.end()) {
      std::vector<FWRepresentationInfo> reps;
      //check all reps
      for(std::vector<boost::shared_ptr<FWRepresentationCheckerBase> >::const_iterator it = m_checkers.begin(),
                                                                                       itEnd = m_checkers.end();
          it != itEnd;
          ++it) {
         FWRepresentationInfo info = (*it)->infoFor(iTypeName);
         if(info.isValid()) {
            reps.push_back(info);
         }
      }
      m_typeToReps.insert(std::make_pair(iTypeName,reps));
      itFound = m_typeToReps.find(iTypeName);
   }

   return itFound->second;
}

//
// static member functions
//
