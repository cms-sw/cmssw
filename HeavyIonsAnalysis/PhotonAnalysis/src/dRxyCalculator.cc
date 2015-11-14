#include "HeavyIonsAnalysis/PhotonAnalysis/plugins/dRxyCalculator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/PatCandidates/interface/Photon.h"

using namespace edm;
using namespace reco;

dRxyCalculator::dRxyCalculator(const edm::Event &iEvent, const edm::EventSetup &iSetup, edm::InputTag trackLabel,std::string trackQuality)
{
   // Get reconstructed tracks
   iEvent.getByLabel(trackLabel, recCollection); // !!
   trackQuality_=trackQuality;
} 

double dRxyCalculator::getDRxy(const reco::Photon p, double x, double y)
{
   using namespace edm;
   using namespace reco;

   double eta1 = p.eta();
   double phi1 = p.phi();

   std::vector<double> Rxy;

   for(reco::TrackCollection::const_iterator
   	  recTrack = recCollection->begin(); recTrack!= recCollection->end(); recTrack++)
   {
      bool goodtrack = recTrack->quality(reco::TrackBase::qualityByName(trackQuality_));
	    if(!goodtrack) continue;

      double pt = recTrack->pt();
      double eta2 = recTrack->eta();
      double phi2 = recTrack->phi();
      
      if(pt < x * 0.4)
         continue;

      double dR = dRDistance(eta1,phi1,eta2,phi2);

      if(Rxy.size() < y+1)
      {  
         Rxy.push_back(dR);
         sort(Rxy.begin(), Rxy.end());
         continue;
      }

      if(dR < Rxy[Rxy.size()-1])
      {  
         Rxy[Rxy.size()-1] = dR;
         sort(Rxy.begin(), Rxy.end());
      }
   }
   
   double rxy = 4;
   if (Rxy.size()<y) rxy=4; else rxy = Rxy[(int)y-1];
   Rxy.clear();
   return rxy;
}

