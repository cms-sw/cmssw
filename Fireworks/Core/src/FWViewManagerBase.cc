// -*- C++ -*-
//
// Package:     Core
// Class  :     FWViewManagerBase
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Sat Jan  5 10:56:17 EST 2008
// $Id: FWViewManagerBase.cc,v 1.16 2011/07/08 04:40:00 amraktad Exp $
//

// system include files
#include "TClass.h"
#include "TROOT.h"
#include <assert.h>
#include <iostream>
#include <boost/bind.hpp>

// user include files
#include "Fireworks/Core/interface/FWViewManagerBase.h"
#include "Fireworks/Core/interface/FWModelChangeManager.h"
#include "Fireworks/Core/interface/FWColorManager.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWViewManagerBase::FWViewManagerBase() :
   m_context(0),
   m_changeManager(0),
   m_colorManager(0)
{
}

// FWViewManagerBase::FWViewManagerBase(const FWViewManagerBase& rhs)
// {
//    // do actual copying here;
// }

FWViewManagerBase::~FWViewManagerBase()
{
}

//
// assignment operators
//
// const FWViewManagerBase& FWViewManagerBase::operator=(const FWViewManagerBase& rhs)
// {
//   //An exception safe implementation is
//   FWViewManagerBase temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void*
FWViewManagerBase::createInstanceOf(const TClass* iBaseClass,
                                    const char* iNameOfClass)
{
   //create proxy builders
   Int_t error;
   assert(iBaseClass !=0);

   //does the class already exist?
   TClass *c = TClass::GetClass( iNameOfClass );
   if(0==c) {
      //try to load a macro of that name

      //How can I tell if this succeeds or failes? error and value are always 0!
      // I could not make the non-compiled mechanism to work without seg-faults
      // Int_t value =
      gROOT->LoadMacro( (std::string(iNameOfClass)+".C+").c_str(), &error );
      c = TClass::GetClass( iNameOfClass );
      if(0==c ) {
         std::cerr <<"failed to find "<< iNameOfClass << std::endl;
         return 0;
      }
   }
   void* inst = c->New();
   void* baseClassInst = c->DynamicCast(iBaseClass,inst);
   if(0==baseClassInst) {
      std::cerr<<"conversion to "<<iBaseClass->ClassName() << " for class " << iNameOfClass << " failed"<<std::endl;
      return 0;
   }
   return baseClassInst;
}

void
FWViewManagerBase::modelChangesComingSlot()
{
   //forward call to the virtual function
   this->modelChangesComing();
}
void
FWViewManagerBase::modelChangesDoneSlot()
{
   //forward call to the virtual function
   this->modelChangesDone();
}
void
FWViewManagerBase::colorsChangedSlot()
{
   this->colorsChanged();
}


void
FWViewManagerBase::setChangeManager(FWModelChangeManager* iCM)
{
   assert(0!=iCM);
   m_changeManager = iCM;
   m_changeManager->changeSignalsAreComing_.connect(boost::bind(&FWViewManagerBase::modelChangesComing,this));
   m_changeManager->changeSignalsAreDone_.connect(boost::bind(&FWViewManagerBase::modelChangesDone,this));
}

void
FWViewManagerBase::setColorManager(FWColorManager* iCM)
{
   assert(0!= iCM);
   m_colorManager = iCM;
   m_colorManager->colorsHaveChanged_.connect(boost::bind(&FWViewManagerBase::colorsChanged,this));
   //make sure to pickup any changes that occurred earlier
   colorsChanged();
}

//
// const member functions
//

FWModelChangeManager&
FWViewManagerBase::changeManager() const
{
   assert(m_changeManager != 0);
   return *m_changeManager;
}

FWColorManager&
FWViewManagerBase::colorManager() const
{
   assert(m_colorManager !=0);
   return *m_colorManager;
}

