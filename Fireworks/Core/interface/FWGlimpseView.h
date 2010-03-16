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
// $Id: FWGlimpseView.h,v 1.13 2009/11/03 16:56:38 amraktad Exp $
//

// system include files
#include "Rtypes.h"

// user include files
#include "Fireworks/Core/interface/FWEveView.h"
#include "Fireworks/Core/interface/FWBoolParameter.h"

// forward declarations
class TEveViewer;
class TEveScene;
class TEveElementList;
class TGLMatrix;
class FWEveValueScaler;
class TEveWindowSlot;
class TEveGeoShape;

class FWGlimpseView : public FWEveView
{
public:
   FWGlimpseView(TEveWindowSlot*, TEveElementList*, FWEveValueScaler*);
   virtual ~FWGlimpseView();

   // ---------- const member functions ---------------------
   const std::string& typeName() const;

   virtual void addTo(FWConfiguration&) const;
   virtual void setFrom(const FWConfiguration&);

   // ---------- static member functions --------------------
   static const std::string& staticTypeName();

private:
   FWGlimpseView(const FWGlimpseView&);    // stop default

   const FWGlimpseView& operator=(const FWGlimpseView&);    // stop default

   void updateScale( double scale );
   void showAxes( );
   void showCylinder( );

   // ---------- member data --------------------------------
   TEveGeoShape*  m_cylinder;

   TGLMatrix* m_cameraMatrix;
   TGLMatrix* m_cameraMatrixBase;
   Double_t*  m_cameraFOV;

   // FWDoubleParameter m_scaleParam;
   FWBoolParameter   m_showAxes;
   FWBoolParameter   m_showCylinder;
   FWEveValueScaler* m_scaler;
};


#endif
