// -*- C++ -*-
//
// Package:     Core
// Class  :     FWListViewObject
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Mon Mar 10 09:02:44 CDT 2008
// $Id: FWListViewObject.cc,v 1.2 2008/03/11 23:30:04 chrjones Exp $
//

// system include files

// user include files
#include "Fireworks/Core/src/FWListViewObject.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWListViewObject::FWListViewObject(const char* iName,
                                   FWViewBase* iView):
TNamed(iName,""),
m_view(iView)
{
}

// FWListViewObject::FWListViewObject(const FWListViewObject& rhs)
// {
//    // do actual copying here;
// }

/*
FWListViewObject::~FWListViewObject()
{
}
*/

//
// assignment operators
//
// const FWListViewObject& FWListViewObject::operator=(const FWListViewObject& rhs)
// {
//   //An exception safe implementation is
//   FWListViewObject temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
bool
FWListViewObject::doSelection(bool iToggleSelection)
{
   return true;
}

//
// const member functions
//
Bool_t
FWListViewObject::CanEditMainColor() const
{
   return kFALSE;
}

//
// static member functions
//

ClassImp(FWListViewObject)
