// -*- C++ -*-
//
// Package:     Subsystem/Package
// Class  :     FWTGLViewer
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  
//         Created:  Tue, 03 Feb 2015 21:45:22 GMT
//

// system include files

// user include files
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
FWTGLViewer::FWTGLViewer(const TGWindow *parent) :
   TGLEmbeddedViewer(parent, 0, 0, 0)
{
}

// FWTGLViewer::FWTGLViewer(const FWTGLViewer& rhs)
// {
//    // do actual copying here;
// }

FWTGLViewer::~FWTGLViewer()
{
}

//
// assignment operators
//
// const FWTGLViewer& FWTGLViewer::operator=(const FWTGLViewer& rhs)
// {
//   //An exception safe implementation is
//   FWTGLViewer temp(rhs);
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

//
// static member functions
//
