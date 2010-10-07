#ifndef Fireworks_Core_FWRhoPhiZView_h
#define Fireworks_Core_FWRhoPhiZView_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWRhoPhiZView
//
/**\class FWRhoPhiZView FWRhoPhiZView.h Fireworks/Core/interface/FWRhoPhiZView.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Tue Feb 19 10:33:21 EST 2008
// $Id: FWRhoPhiZView.h,v 1.29 2010/03/16 11:51:53 amraktad Exp $
//

// system include files
#include <string>
#include "TEveProjections.h"

// user include files
#include "Fireworks/Core/interface/FWEveView.h"
#include "Fireworks/Core/interface/FWDoubleParameter.h"
#include "Fireworks/Core/interface/FWBoolParameter.h"
#include "Fireworks/Core/interface/FWEvePtr.h"

// forward declarations
class TEveProjectionManager;
class TGLMatrix;
class TEveCalo2D;
class TEveProjectionAxes;
class TEveWindowSlot;
class FWRhoPhiZViewManager;


class FWRhoPhiZView : public FWEveView
{

public:
   FWRhoPhiZView(TEveWindowSlot* iParent,
                 const std::string& iTypeName,
                 const TEveProjection::EPType_e& iProjType);
   virtual ~FWRhoPhiZView();

   // ---------- const member functions ---------------------
   const std::string& typeName() const;

   virtual void addTo(FWConfiguration&) const;
   virtual void setFrom(const FWConfiguration&);

   // ---------- member functions ---------------------------
   void resetCamera();
   void destroyElements();
   void replicateGeomElement(TEveElement*);
   void showProjectionAxes( );
   void eventEnd();

   //returns the new element created from this import
   TEveElement* importElements(TEveElement* iProjectableChild, float iLayer, TEveElement* iProjectedParent=0);

private:
   void doDistortion();
   void doCompression(bool);
   void doZoom(double);
   void updateCaloParameters();
   void updateScaleParameters();
   void updateCalo(TEveElement*, bool dataChanged = false);
   void updateCaloLines(TEveElement*);

   FWRhoPhiZView(const FWRhoPhiZView&);    // stop default

   const FWRhoPhiZView& operator=(const FWRhoPhiZView&);    // stop default

   // ---------- member data --------------------------------
   FWEvePtr<TEveProjectionManager> m_projMgr;
   TEveProjection::EPType_e m_projType;
   std::vector<TEveElement*> m_geom;
   std::string m_typeName;
   double m_caloScale;
   FWEvePtr<TEveProjectionAxes> m_axes;

   // parameters
   FWLongParameter    m_overlayEventInfoLevel;
   FWBoolParameter    m_drawCMSLogo;
   FWDoubleParameter  m_caloDistortion;
   FWDoubleParameter  m_muonDistortion;
   FWBoolParameter    m_showProjectionAxes;
   FWBoolParameter    m_compressMuon;
   FWDoubleParameter  m_caloFixedScale;
   FWBoolParameter    m_caloAutoScale;
   FWBoolParameter*   m_showHF;
   FWBoolParameter*   m_showEndcaps;

   // camera parameters
   double*    m_cameraZoom;
   TGLMatrix* m_cameraMatrix;
};


#endif
