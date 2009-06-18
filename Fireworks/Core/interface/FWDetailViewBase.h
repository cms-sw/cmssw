#ifndef Fireworks_Core_FWDetailViewBase_h
#define Fireworks_Core_FWDetailViewBase_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWDetailViewBase
//
/**\class FWDetailViewBase FWDetailViewBase.h Fireworks/Core/interface/FWDetailViewBase.h

   Description: Base class for detailed views

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Fri Jan  9 13:35:52 EST 2009
// $Id: FWDetailViewBase.h,v 1.4 2009/06/05 20:00:32 amraktad Exp $
//

// system include files
#include "Rtypes.h"

// user include files
#include "Fireworks/Core/interface/FWSimpleProxyHelper.h"

// forward declarations
class TEveElement;
class TLatex;
class TGLViewer;
class FWModelId;
class TCanvas;

class FWDetailViewBase
{
public:
   virtual ~FWDetailViewBase ();

   ///the calling code takes ownership of the returned object
   TEveElement* build (const FWModelId &);
   virtual void  clearOverlayElements() {}

   void         setLatex (TLatex *v) {
      m_latex = v;
   }
   void         setViewer (TGLViewer *v) {
      m_viewer = v;
   }

   void         setCanvas (TCanvas *c) {
      m_canvas = c;
   }


   void  setUseGL (Bool_t x) {
      m_useGL = x;
   }

   Bool_t useGL() const {
      return m_useGL;
   }

protected:
   FWDetailViewBase(const std::type_info&);

   TLatex*      latex() const {
      return m_latex;
   }

   TGLViewer*       viewer () const {
      return m_viewer;
   }

   TCanvas*       canvas () const {
      return m_canvas;
   }

   const Double_t*  rotationCenter() const {
      return m_rotationCenter;
   }
   Double_t*        rotationCenter() {
      return m_rotationCenter;
   }

   void getCenter( Double_t* vars )
   {
      vars[0] = rotationCenter()[0];
      vars[1] = rotationCenter()[1];
      vars[2] = rotationCenter()[2];
   }

   void resetCenter() {
      rotationCenter()[0] = 0;
      rotationCenter()[1] = 0;
      rotationCenter()[2] = 0;
   }

private:
   FWDetailViewBase(const FWDetailViewBase&); // stop default

   const FWDetailViewBase& operator=(const FWDetailViewBase&); // stop default

   virtual TEveElement* build(const FWModelId&, const void*) = 0;

   Bool_t           m_useGL;

   TGLViewer       *m_viewer;
   TCanvas         *m_canvas;
   TLatex          *m_latex;
   Double_t         m_rotationCenter[3];


   FWSimpleProxyHelper m_helper;

   // ---------- member data --------------------------------

};


#endif
