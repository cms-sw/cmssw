#include "HeavyIonsAnalysis/PhotonAnalysis/plugins/TxyCalculator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/Candidate/interface/Candidate.h"

using namespace edm;
using namespace reco;

TxyCalculator::TxyCalculator(const edm::Event &iEvent, const edm::EventSetup &iSetup, edm::InputTag trackLabel,std::string trackQuality)
{ 
   // Get reconstructed tracks
   iEvent.getByLabel(trackLabel, recCollection); // !!
   trackQuality_=trackQuality;
} 


int TxyCalculator::getNumAllTracks(double ptCut)
{
  using namespace edm;
  using namespace reco;

  int nTracks = 0;
  
  for(reco::TrackCollection::const_iterator
	recTrack = recCollection->begin(); recTrack!= recCollection->end(); recTrack++)
    {
    
      bool goodtrack = recTrack->quality(reco::TrackBase::qualityByName(trackQuality_));
	    if(!goodtrack) continue;

      double pt = recTrack->pt();
      if ( pt > ptCut)  
	nTracks = nTracks +1;
    }
  return nTracks;
}


int TxyCalculator::getNumLocalTracks(const reco::Photon p, double detaCut, double ptCut)
{
  using namespace edm;
  using namespace reco;

  int nTracks = 0;

  double eta1 = p.eta();
  double phi1 = p.phi();

  for(reco::TrackCollection::const_iterator
        recTrack = recCollection->begin(); recTrack!= recCollection->end(); recTrack++)
    {
      double pt = recTrack->pt();
      if ( (pt > ptCut) && ( fabs(eta1 - recTrack->eta()) < detaCut) && ( fabs(calcDphi(recTrack->phi(),phi1)) < 3.141592/2. ) )
        nTracks= nTracks +1;
    }
  return nTracks;
}

double TxyCalculator::getTxy(const reco::Photon p, double x, double y)
{
   using namespace edm;
   using namespace reco;

   //   if(!recCollection)
   //    {
   //	 LogError("TxyCalculator") << "Error! The track container is not found.";
   //	 return -100;
   //   }
   

   double eta1 = p.eta();
   double phi1 = p.phi();
   
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

double TxyCalculator::getHollSxy(const reco::Photon p, double thePtCut, double outerR, double innerR)
{
   using namespace edm;
   using namespace reco;
   
   double eta1 = p.eta();
   double phi1 = p.phi();

   double ptSum = 0;

   for(reco::TrackCollection::const_iterator
          recTrack = recCollection->begin(); recTrack!= recCollection->end(); recTrack++)
      {
	 double pt = recTrack->pt();
	 double eta2 = recTrack->eta();
	 double phi2 = recTrack->phi();
	 if (dRDistance(eta1,phi1,eta2,phi2) >= outerR)
	    continue;
	 if (dRDistance(eta1,phi1,eta2,phi2) <= innerR)
	    continue;
	 if(pt > thePtCut)
	    ptSum = ptSum + pt;
      }
   
   return ptSum;
}
