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

#include <stdexcept>

// user include files

#include "TMath.h"

#include "TGLIncludes.h"
#include "TGLFBO.h"
#include "TGLWidget.h"

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
   TGLEmbeddedViewer(parent, 0, 0, 0),
   m_fbo(0),
   m_fbo_w(-1), m_fbo_h(-1)
{
}

// FWTGLViewer::FWTGLViewer(const FWTGLViewer& rhs)
// {
//    // do actual copying here;
// }

FWTGLViewer::~FWTGLViewer()
{
   delete m_fbo;
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

//------------------------------------------------------------------------------
// Draw functions
//------------------------------------------------------------------------------

void FWTGLViewer::DrawHiLod(Bool_t swap_buffers)
{
   fRedrawTimer->Stop();

   // Ignore request if GL window or context not yet availible or shown.
   if ((!fGLWidget && fGLDevice == -1) || (fGLWidget && !fGLWidget->IsMapped()))
   {
      return;
   }

   // Take scene draw lock - to be revisited
   if ( ! TakeLock(kDrawLock))
   {
      // If taking drawlock fails the previous draw is still in progress
      // set timer to do this one later
      Error("FWTGLViewer::DrawHiLodNoSwap", "viewer locked - skipping this draw.");
      fRedrawTimer->RequestDraw(100, TGLRnrCtx::kLODHigh);
      return;
   }

   fLOD = TGLRnrCtx::kLODHigh;

   DoDraw(swap_buffers);
}

void FWTGLViewer::JustSwap()
{
   fGLWidget->SwapBuffers();
}

//------------------------------------------------------------------------------
// FBO functions
//------------------------------------------------------------------------------

//______________________________________________________________________________
TGLFBO* FWTGLViewer::MakeFbo()
{
   // Generate FBO with same dimensions as the viewport.

   return GenerateFbo(fViewport.Width(), fViewport.Height(), kFALSE);
}

//______________________________________________________________________________
TGLFBO* FWTGLViewer::MakeFboWidth(Int_t width, Bool_t pixel_object_scale)
{
   // Generate FBO with given width (height scaled proportinally).
   // If pixel_object_scale is true (default), the corresponding
   // scaling gets calculated from the current window size.

   Float_t scale  = Float_t(width) / fViewport.Width();
   Int_t   height = TMath::Nint(scale*fViewport.Height());

   return GenerateFbo(width, height, pixel_object_scale ? scale : 0);
}

//______________________________________________________________________________
TGLFBO* FWTGLViewer::MakeFboHeight(Int_t height, Bool_t pixel_object_scale)
{
   // Generate FBO with given height (width scaled proportinally).
   // If pixel_object_scale is true (default), the corresponding
   // scaling gets calculated from the current window size.

   Float_t scale = Float_t(height) / fViewport.Height();
   Int_t   width = TMath::Nint(scale*fViewport.Width());

   return GenerateFbo(width, height, pixel_object_scale ? scale : 0);
}

//______________________________________________________________________________
TGLFBO* FWTGLViewer::MakeFboScale (Float_t scale, Bool_t pixel_object_scale)
{
   // Generate FBO with given scale to current window size.
   // If pixel_object_scale is true (default), the same scaling is
   // used.

   Int_t w = TMath::Nint(scale*fViewport.Width());
   Int_t h = TMath::Nint(scale*fViewport.Height());

   return GenerateFbo(w, h, pixel_object_scale ? scale : 0);
}

//______________________________________________________________________________
TGLFBO* FWTGLViewer::GenerateFbo(Int_t w, Int_t h, Float_t pixel_object_scale)
{
   // Generate FBO -- function that does the actual work.

   static const TString eh("FWTGLViewer::SavePictureUsingFBO");

   if ( ! GLEW_EXT_framebuffer_object)
   {
      ::Warning(eh, "Missing FBO support.");
   }

   if ( ! TakeLock(kDrawLock)) {
      ::Error(eh, "viewer locked - try later.");
      return 0;
   }

   TUnlocker ulck(this);

   MakeCurrent();

   if (m_fbo == 0)
   {
      m_fbo = new TGLFBO();
   }
   if (m_fbo_w != w || m_fbo_h != h)
   {
      try
      {
         m_fbo->Init(w, h, fGLWidget->GetPixelFormat()->GetSamples());
      }
      catch (std::runtime_error& exc)
      {
         m_fbo_w = m_fbo_h = -1;

         ::Error(eh, "%s",exc.what());
         return 0;
      }

      m_fbo_w = w; m_fbo_h = h;
   }

   TGLRect old_vp(fViewport);
   SetViewport(0, 0, w, h);

   Float_t old_scale = 1;
   if (pixel_object_scale != 0)
   {
      old_scale = fRnrCtx->GetRenderScale();
      fRnrCtx->SetRenderScale(old_scale * pixel_object_scale);
   }

   m_fbo->Bind();

   fLOD = TGLRnrCtx::kLODHigh;
   fRnrCtx->SetGrabImage(kTRUE);

   DoDraw(kFALSE);

   fRnrCtx->SetGrabImage(kFALSE);

   m_fbo->Unbind();

   if (pixel_object_scale != 0)
   {
      fRnrCtx->SetRenderScale(old_scale);
   }

   SetViewport(old_vp);

   return m_fbo;
}

//
// const member functions
//

//
// static member functions
//
