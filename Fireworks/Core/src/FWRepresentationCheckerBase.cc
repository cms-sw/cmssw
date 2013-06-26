// -*- C++ -*-
//
// Package:     Core
// Class  :     FWRepresentationCheckerBase
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Tue Nov 11 14:08:50 EST 2008
// $Id: FWRepresentationCheckerBase.cc,v 1.3 2010/06/02 22:36:36 chrjones Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FWRepresentationCheckerBase.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWRepresentationCheckerBase::FWRepresentationCheckerBase(const std::string& iPurpose, 
                                                         unsigned int iBitPackedViews,
                                                         bool iRepresentsSubPart) :
   m_purpose(iPurpose), m_bitPackedViews(iBitPackedViews), m_representsSubPart(iRepresentsSubPart)
{
}

// FWRepresentationCheckerBase::FWRepresentationCheckerBase(const FWRepresentationCheckerBase& rhs)
// {
//    // do actual copying here;
// }

FWRepresentationCheckerBase::~FWRepresentationCheckerBase()
{
}

//
// assignment operators
//
// const FWRepresentationCheckerBase& FWRepresentationCheckerBase::operator=(const FWRepresentationCheckerBase& rhs)
// {
//   //An exception safe implementation is
//   FWRepresentationCheckerBase temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//

//
// const member functions
//
const std::string&
FWRepresentationCheckerBase::purpose() const
{
   return m_purpose;
}

unsigned int 
FWRepresentationCheckerBase::bitPackedViews() const
{
   return m_bitPackedViews;
}

bool FWRepresentationCheckerBase::representsSubPart() const
{
   return m_representsSubPart;
}
//
// static member functions
//
