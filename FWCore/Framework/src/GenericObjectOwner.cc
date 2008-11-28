// -*- C++ -*-
//
// Package:     Framework
// Class  :     GenericObjectOwner
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Thu Feb  7 17:21:22 EST 2008
// $Id: GenericObjectOwner.cc,v 1.1 2008/02/12 21:48:33 chrjones Exp $
//

// system include files

// user include files
#include "FWCore/Framework/interface/GenericObjectOwner.h"

namespace edm {
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
//GenericObjectOwner::GenericObjectOwner()
//{
//}

// GenericObjectOwner::GenericObjectOwner(const GenericObjectOwner& rhs)
// {
//    // do actual copying here;
// }

GenericObjectOwner::~GenericObjectOwner()
{
   if(m_ownData) {
      m_object.Destruct();
   }
}

//
// assignment operators
//
// const GenericObjectOwner& GenericObjectOwner::operator=(const GenericObjectOwner& rhs)
// {
//   //An exception safe implementation is
//   GenericObjectOwner temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void 
GenericObjectOwner::swap(GenericObjectOwner& iOther)
{
   Reflex::Object old(m_object);
   m_object = iOther.m_object;
   iOther.m_object = m_object;
}

//
// const member functions
//
Reflex::Object 
GenericObjectOwner::object() const
{
   return m_object;
}

//
// static member functions
//
}
