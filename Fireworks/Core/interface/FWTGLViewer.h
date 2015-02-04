#ifndef Subsystem_Package_FWTGLViewer_h
#define Subsystem_Package_FWTGLViewer_h
// -*- C++ -*-
//
// Package:     Subsystem/Package
// Class  :     FWTGLViewer
// 
/**\class FWTGLViewer FWTGLViewer.h "FWTGLViewer.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  
//         Created:  Tue, 03 Feb 2015 21:45:22 GMT
//

// system include files

// user include files

#include "TGLEmbeddedViewer.h"

// forward declarations

class TGWindow;


class FWTGLViewer : public TGLEmbeddedViewer
{

public:
   FWTGLViewer(const TGWindow *parent);
   virtual ~FWTGLViewer();

   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------

private:
   FWTGLViewer(const FWTGLViewer&); // stop default

   const FWTGLViewer& operator=(const FWTGLViewer&); // stop default

   // ---------- member data --------------------------------

};


#endif
