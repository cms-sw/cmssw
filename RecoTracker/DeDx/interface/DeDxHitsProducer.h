#ifndef DeDxHitsProducer_H
#define DeDxHitsProducer_H
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/TrackReco/interface/Track.h"
//
// class decleration
//
#include <map>
class TrackerGeometry;

class DeDxHitsProducer : public edm::EDProducer {
   public:
      explicit DeDxHitsProducer(const edm::ParameterSet&);
      ~DeDxHitsProducer();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      /**cached access to normalization map */ 
      double normalization(DetId id);
     
      /**cached access to thickness map */ 
      double thickness(DetId id);

      /**cached access to distance map */
      double distance(DetId id);

      bool compatibleTracks(const reco::Track &,const reco::Track &);
      
      /** 
       * At the moment the following function is doing nothing special but
       * in principle can be used to take into account non linear effects
       * e.g. for the pixel detector
       */
       
      double normalize(DetId id, double charge) { 
        return normalization(id)*charge;
      }

     
      // ----------member data ---------------------------
     edm::InputTag m_tracksTag;
     edm::InputTag m_refittedTracksTag;
    // edm::InputTag m_trajectoriesTag;
     edm::InputTag m_trajTrackAssociationTag;
     
     std::map<DetId,double> m_normalizationMap;
     std::map<DetId,double> m_distanceMap;
     std::map<DetId,double> m_thicknessMap;

     const TrackerGeometry * m_tracker;
 
     unsigned long long m_geometryCacheId;
     unsigned long long m_calibrationCacheId;
     
};


#endif

