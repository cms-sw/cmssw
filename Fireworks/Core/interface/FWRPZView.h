#ifndef Fireworks_Core_FWRPZView_h
#define Fireworks_Core_FWRPZView_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWRPZView
//
/**\class FWRPZView FWRPZView.h Fireworks/Core/interface/FWRPZView.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Tue Feb 19 10:33:21 EST 2008
//

// system include files
#include <string>

// user include files
#include "Fireworks/Core/interface/FWEveView.h"
#include "Fireworks/Core/interface/FWDoubleParameter.h"
#include "Fireworks/Core/interface/FWBoolParameter.h"
#include "Fireworks/Core/interface/FWEvePtr.h"
#include "TEveVector.h"

// forward declarations
class TEveProjectionManager;
class TGLMatrix;
class TEveCalo2D;
class TEveProjectionAxes;
class TEveWindowSlot;
class FWColorManager;
class FWRPZViewGeometry;

class FWRPZView : public FWEveView
{
public:
   FWRPZView(TEveWindowSlot* iParent, FWViewType::EType);
   virtual ~FWRPZView();

   // ---------- const member functions ---------------------

   virtual void addTo(FWConfiguration&) const;
   virtual void populateController(ViewerParameterGUI&) const;
   virtual TEveCaloViz* getEveCalo() const;

   // ---------- member functions ---------------------------
   virtual void setContext(const fireworks::Context&);
   virtual void setFrom(const FWConfiguration&);
   virtual void voteCaloMaxVal();

   virtual void eventBegin();

   //returns the new element created from this import
   void importElements(TEveElement* iProjectableChild, float layer, TEveElement* iProjectedParent=0);
 
   void shiftOrigin(TEveVector& center);
   void resetOrigin();
private:
   FWRPZView(const FWRPZView&);    // stop default
   const FWRPZView& operator=(const FWRPZView&);    // stop default 

   void doPreScaleDistortion();
   void doFishEyeDistortion();
   void doCompression(bool);
   void doShiftOriginToBeamSpot();


   void setEtaRng();
   
   void showProjectionAxes( );
   void projectionAxesLabelSize( );

   // ---------- member data --------------------------------
   const  static float s_distortF;
   const  static float s_distortFInv;

   FWRPZViewGeometry* m_geometryList;
   TEveProjectionManager* m_projMgr;
   TEveProjectionAxes*    m_axes;
   TEveCalo2D*            m_calo;


   // parameters

   FWBoolParameter m_showPixelBarrel;
   FWBoolParameter m_showPixelEndcap;
   FWBoolParameter m_showTrackerBarrel;
   FWBoolParameter m_showTrackerEndcap;
   FWBoolParameter m_showRpcEndcap;
   FWBoolParameter m_showGEM;
   FWBoolParameter m_showME0;

   FWBoolParameter    m_shiftOrigin;
   FWDoubleParameter  m_fishEyeDistortion;
   FWDoubleParameter  m_fishEyeR;

   FWDoubleParameter  m_caloDistortion;
   FWDoubleParameter  m_muonDistortion;
   FWBoolParameter    m_showProjectionAxes;
   FWDoubleParameter  m_projectionAxesLabelSize;
   FWBoolParameter    m_compressMuon;

   FWBoolParameter*   m_showHF;
   FWBoolParameter*   m_showEndcaps;

};


#endif
