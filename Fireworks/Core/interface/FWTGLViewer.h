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
class TGLFBO;

class FWTGLViewer : public TGLEmbeddedViewer
{

public:
   FWTGLViewer(const TGWindow *parent);
   virtual ~FWTGLViewer();

   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------

   void    DrawHiLod(Bool_t swap_buffers);
   void    JustSwap();

   TGLFBO* MakeFbo();
   TGLFBO* MakeFboWidth (Int_t width,   Bool_t pixel_object_scale=kTRUE);
   TGLFBO* MakeFboHeight(Int_t height,  Bool_t pixel_object_scale=kTRUE);
   TGLFBO* MakeFboScale (Float_t scale, Bool_t pixel_object_scale=kTRUE);

   TGLFBO* GenerateFbo(Int_t w, Int_t h, Float_t pixel_object_scale);

private:
   FWTGLViewer(const FWTGLViewer&); // stop default

   const FWTGLViewer& operator=(const FWTGLViewer&); // stop default

   // ---------- member data --------------------------------

   TGLFBO *m_fbo;
   int     m_fbo_w, m_fbo_h;
};


#endif
