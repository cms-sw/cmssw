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
// $Id: FWRPZView.h,v 1.15 2010/11/04 22:38:54 amraktad Exp $
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

   //returns the new element created from this import
   void importElements(TEveElement* iProjectableChild, float layer, TEveElement* iProjectedParent=0);
private:
   FWRPZView(const FWRPZView&);    // stop default
   const FWRPZView& operator=(const FWRPZView&);    // stop default 

   void doDistortion();
   void doCompression(bool);
   
   void setEtaRng();

   void showProjectionAxes( );

   // ---------- member data --------------------------------
  static FWRPZViewGeometry* s_geometryList;

   TEveProjectionManager* m_projMgr;
   TEveProjectionAxes*    m_axes;
   TEveCalo2D*            m_calo;

   // parameters
   FWDoubleParameter  m_caloDistortion;
   FWDoubleParameter  m_muonDistortion;
   FWBoolParameter    m_showProjectionAxes;
   FWBoolParameter    m_compressMuon;
   FWBoolParameter*   m_showHF;
   FWBoolParameter*   m_showEndcaps;

};


#endif
