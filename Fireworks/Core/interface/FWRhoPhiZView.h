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
// $Id: FWRhoPhiZView.h,v 1.30 2010/03/16 14:52:46 amraktad Exp $
//

// system include files
#include <string>

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
class FWColorManager;

class FWRhoPhiZView : public FWEveView
{
public:
   FWRhoPhiZView(TEveWindowSlot* iParent, FWViewType::EType);
   virtual ~FWRhoPhiZView();

   // ---------- const member functions ---------------------

   virtual void addTo(FWConfiguration&) const;
   virtual void setFrom(const FWConfiguration&);

   // ---------- member functions ---------------------------
   virtual void setGeometry( const DetIdToMatrix* geom, FWColorManager&);

   //returns the new element created from this import
   
   void eventEnd();
   void importElements(TEveElement* iProjectableChild, float iLayer, TEveElement* iProjectedParent=0);

private:
   FWRhoPhiZView(const FWRhoPhiZView&);    // stop default
   const FWRhoPhiZView& operator=(const FWRhoPhiZView&);    // stop default 

   void doDistortion();
   void doCompression(bool);
   // void doZoom(double);

   void updateCaloParameters();
   void updateScaleParameters();
   void updateCalo(TEveElement*, bool dataChanged = false);
   void updateCaloLines(TEveElement*);

   void showProjectionAxes( );
   // ---------- member data --------------------------------
   FWEvePtr<TEveProjectionManager> m_projMgr;

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

};


#endif
