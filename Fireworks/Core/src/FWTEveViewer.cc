// -*- C++ -*-
//
// Package:     Subsystem/Package
// Class  :     FWTEveViewer
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  
//         Created:  Tue, 03 Feb 2015 21:46:04 GMT
//

// system include files

// user include files

#include "Fireworks/Core/interface/FWTEveViewer.h"
#include "Fireworks/Core/interface/FWTGLViewer.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWTEveViewer::FWTEveViewer(const char* n, const char* t) :
   TEveViewer(n, t)
{
}

// FWTEveViewer::FWTEveViewer(const FWTEveViewer& rhs)
// {
//    // do actual copying here;
// }

FWTEveViewer::~FWTEveViewer()
{
}

//
// assignment operators
//
// const FWTEveViewer& FWTEveViewer::operator=(const FWTEveViewer& rhs)
// {
//   //An exception safe implementation is
//   FWTEveViewer temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//

FWTGLViewer* FWTEveViewer::SpawnFWTGLViewer()
{
   TGCompositeFrame* cf = GetGUICompositeFrame();

   FWTGLViewer* v = new FWTGLViewer(cf);
   SetGLViewer(v, v->GetFrame());

   cf->AddFrame(fGLViewerFrame, new TGLayoutHints(kLHintsNormal | kLHintsExpandX | kLHintsExpandY));

   fGLViewerFrame->MapWindow();

   if (fEveFrame == 0)
      PreUndock();

   return v;
}

//
// const member functions
//

//
// static member functions
//
