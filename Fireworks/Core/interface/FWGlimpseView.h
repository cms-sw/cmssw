#ifndef Fireworks_Core_FWGlimpseView_h
#define Fireworks_Core_FWGlimpseView_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWGlimpseView
//
/**\class FWGlimpseView FWGlimpseView.h Fireworks/Core/interface/FWGlimpseView.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Thu Feb 21 11:22:37 EST 2008
// $Id: FWGlimpseView.h,v 1.10 2009/01/23 21:35:41 amraktad Exp $
//

// system include files
#include "Rtypes.h"

// user include files
#include "Fireworks/Core/interface/FWViewBase.h"
#include "Fireworks/Core/interface/FWLongParameter.h"
#include "Fireworks/Core/interface/FWDoubleParameter.h"
#include "Fireworks/Core/interface/FWBoolParameter.h"

// forward declarations
class TGFrame;
class TGLEmbeddedViewer;
class TEvePad;
class TEveViewer;
class TEveScene;
class TEveElementList;
class TEveGeoShape;
class TGLMatrix;
class FWGlimpseViewManager;
class FWEveValueScaler;
class TEveWindowSlot;

class FWGlimpseView : public FWViewBase
{

public:
   FWGlimpseView(TEveWindowSlot*, TEveElementList*, FWEveValueScaler*);
   virtual ~FWGlimpseView();

   // ---------- const member functions ---------------------
   TGFrame* frame() const;
   const std::string& typeName() const;
   virtual void addTo(FWConfiguration&) const;

   virtual void saveImageTo(const std::string& iName) const;

   // ---------- static member functions --------------------
   static const std::string& staticTypeName();

   // ---------- member functions ---------------------------
   virtual void setFrom(const FWConfiguration&);

private:
   FWGlimpseView(const FWGlimpseView&);    // stop default

   const FWGlimpseView& operator=(const FWGlimpseView&);    // stop default

   void updateScale( double scale );
   void showAxes( );
   void showCylinder( );

   // ---------- member data --------------------------------
   TEvePad* m_pad;
   TEveViewer* m_viewer;
   TGLEmbeddedViewer* m_embeddedViewer;
   TEveScene* m_scene;
   TEveGeoShape*  m_cylinder;

   TGLMatrix* m_cameraMatrix;
   TGLMatrix* m_cameraMatrixBase;
   Double_t*  m_cameraFOV;

   // FWDoubleParameter m_scaleParam;
   FWBoolParameter m_showAxes;
   FWBoolParameter m_showCylinder;
   FWEveValueScaler* m_scaler;
};


#endif
