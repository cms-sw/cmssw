#include "RecoHI/HiEgammaAlgos/interface/TxyCalculator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/Candidate/interface/Candidate.h"

using namespace edm;
using namespace reco;

TxyCalculator::TxyCalculator(const edm::Event &iEvent, const edm::EventSetup &iSetup, edm::InputTag trackLabel)
{ 
   // Get reconstructed tracks
   iEvent.getByLabel(trackLabel, recCollection); // !!
} 

double TxyCalculator::getTxy(const reco::SuperClusterRef p, double x, double y)
{
   using namespace edm;
   using namespace reco;

   /*
   if(!recCollection)
   {
      LogError("TxyCalculator") << "Error! The track container is not found.";
      return -100;
   }
   */
   

   double eta1 = p->eta();
   double phi1 = p->phi();
   
   float txy = 0;

   for(reco::TrackCollection::const_iterator
   	  recTrack = recCollection->begin(); recTrack!= recCollection->end(); recTrack++)
   {
      double pt = recTrack->pt();
      double eta2 = recTrack->eta();
      double phi2 = recTrack->phi();
      
      if(dRDistance(eta1,phi1,eta2,phi2) >= 0.1 * x)
         continue;

      if(pt > y * 0.4)
         txy = txy + 1;
   }

   return txy;
}

