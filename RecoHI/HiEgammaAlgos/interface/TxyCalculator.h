#ifndef TxyCalculator_h
#define TxyCalculator_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "DataFormats/EgammaCandidates/interface/Photon.h"

#define PI 3.141592653

class TxyCalculator
{
public:  
   TxyCalculator(const edm::Event &iEvent, const edm::EventSetup &iSetup, const edm::InputTag& trackLabel);
   double getTxy(const reco::Photon& p, double x, double y);
   double getHollSxy(const reco::Photon& p, double thePtCut, double outerR, double innerR);
   int getNumAllTracks(double ptCut);
   int getNumLocalTracks(const reco::Photon& p, double detaCut, double ptCut);

   
private:

   double dRDistance(double eta1,double phi1,double eta2,double phi2)
   {
      double deta = eta1 - eta2;
      double dphi = (calcDphi(phi1, phi2));
      
      return sqrt(deta * deta + dphi * dphi);
   }

   double calcDphi(double phi1_,double phi2_)
   {
       double dphi=phi1_-phi2_;

      if (dphi>0){
         while (dphi>2*PI) dphi-=2*PI;
         if (dphi>PI) dphi=2*PI-dphi; 
      } else {
         while (dphi<-2*PI) dphi+=2*PI;
         if (dphi<-PI) dphi=-2*PI-dphi;
      }
      return dphi;
   }

private:
   edm::Handle<reco::TrackCollection>  recCollection;
};

#endif

