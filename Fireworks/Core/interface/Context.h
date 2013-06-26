#ifndef Fireworks_Core_Context_h
#define Fireworks_Core_Context_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     Context
//
/**\class Context Context.h Fireworks/Core/interface/Context.h

   Description: Central collection of all framework managers

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Tue Sep 30 14:21:45 EDT 2008
// $Id: Context.h,v 1.22 2011/03/07 18:23:28 matevz Exp $
//

// system include files

// user include files

// forward declarations
class TEveTrackPropagator;
class TEveCaloDataHist;
class TEveCaloDataVec;

class FWModelChangeManager;
class FWSelectionManager;
class FWEventItemsManager;
class FWColorManager;
class FWJobMetadataManager;
class FWMagField;
class FWGeometry;
class FWBeamSpot;
class CmsShowCommon;

namespace fireworks {
class Context {

public:
   Context(FWModelChangeManager* iCM,
           FWSelectionManager*   iSM,
           FWEventItemsManager*  iEM,
           FWColorManager*       iColorM,
           FWJobMetadataManager* iJMDM);
   virtual ~Context();

   void  setGeom(const FWGeometry* x) { m_geom = x; }

   // ---------- const member functions ---------------------
   FWModelChangeManager* modelChangeManager() const {
      return m_changeManager;
   }
   FWSelectionManager* selectionManager() const {
      return m_selectionManager;
   }

   const FWEventItemsManager* eventItemsManager() const {
      return m_eventItemsManager;
   }
      
   FWColorManager* colorManager() const {
      return m_colorManager;
   }

   FWJobMetadataManager* metadataManager() const {
      return m_metadataManager;
   }

   TEveTrackPropagator* getTrackPropagator()        const { return m_propagator;        }
   TEveTrackPropagator* getTrackerTrackPropagator() const { return m_trackerPropagator; }
   TEveTrackPropagator* getMuonTrackPropagator()    const { return m_muonPropagator;    }

   FWMagField*          getField()             const { return m_magField; }
   FWBeamSpot*          getBeamSpot()          const { return m_beamSpot; }

   TEveCaloDataHist*    getCaloData()   const { return m_caloData; }
   TEveCaloDataVec*     getCaloDataHF() const { return m_caloDataHF; }

   const  FWGeometry* getGeom()  const { return m_geom; }   

   CmsShowCommon* commonPrefs() const;

   float getMaxEnergyInEvent(bool isEt) const;
   void  voteMaxEtAndEnergy(float Et, float energy) const;
   void  resetMaxEtAndEnergy() const;

   // ---------- member functions ---------------------------
  
   void initEveElements();
   void deleteEveElements();

   // ---------- static member  ---------------------------

   static float  caloR1(bool offset = true);
   static float  caloR2(bool offset = true);
   static float  caloZ1(bool offset = true);
   static float  caloZ2(bool offset = true);

   static float  caloTransEta();
   static float  caloTransAngle();
   static double caloMaxEta();

private:
   Context(const Context&); // stop default
   const Context& operator=(const Context&); // stop default

   // ---------- member data --------------------------------
   FWModelChangeManager *m_changeManager;
   FWSelectionManager   *m_selectionManager;
   FWEventItemsManager  *m_eventItemsManager;
   FWColorManager       *m_colorManager;
   FWJobMetadataManager *m_metadataManager;

   const FWGeometry     *m_geom;

   TEveTrackPropagator  *m_propagator;
   TEveTrackPropagator  *m_trackerPropagator;
   TEveTrackPropagator  *m_muonPropagator;

   FWMagField           *m_magField;
   FWBeamSpot           *m_beamSpot;

   CmsShowCommon        *m_commonPrefs;

   mutable float                 m_maxEt;
   mutable float                 m_maxEnergy;

   TEveCaloDataHist     *m_caloData;
   TEveCaloDataVec      *m_caloDataHF;

   // calo data
   static const float s_caloTransEta;
   static const float s_caloTransAngle;
   // simplified 
   static const float s_caloR; 
   static const float s_caloZ;

   // proxy-builder offsets
   static const float s_caloOffR;
   static const float s_caloOffZ;
};
}

#endif
