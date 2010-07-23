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
// $Id: Context.h,v 1.14 2010/06/07 17:54:01 amraktad Exp $
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
class DetIdToMatrix;

namespace fireworks {
class Context {

public:
   Context(FWModelChangeManager* iCM,
           FWSelectionManager*   iSM,
           FWEventItemsManager*  iEM,
           FWColorManager*       iColorM,
           FWJobMetadataManager* iJMDM);
   virtual ~Context();

   void  setGeom(const DetIdToMatrix* x) { m_geom = x; }
 
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

   TEveCaloDataHist*    getCaloData()   const { return m_caloData; }
   TEveCaloDataVec*     getCaloDataHF() const { return m_caloDataHF; }

  const  DetIdToMatrix* getGeom()  const { return m_geom; }   

   // ---------- member functions ---------------------------

   // ---------- static member  ---------------------------

   static const float s_ecalR;
   static const float s_ecalZ;
   static const float s_transitionAngle;

  void initEveElements();
  void deleteEveElements();

private:
   Context(const Context&); // stop default
   const Context& operator=(const Context&); // stop default

   // ---------- member data --------------------------------
   FWModelChangeManager *m_changeManager;
   FWSelectionManager   *m_selectionManager;
   FWEventItemsManager  *m_eventItemsManager;
   FWColorManager       *m_colorManager;
   FWJobMetadataManager *m_metadataManager;

   const DetIdToMatrix  *m_geom;

   TEveTrackPropagator  *m_propagator;
   TEveTrackPropagator  *m_trackerPropagator;
   TEveTrackPropagator  *m_muonPropagator;

   FWMagField           *m_magField;

   TEveCaloDataHist     *m_caloData;
   TEveCaloDataVec      *m_caloDataHF;
};
}

#endif
