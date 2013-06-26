#ifndef TxCalculator_h
#define TxCalculator_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "DataFormats/EgammaCandidates/interface/Photon.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "CLHEP/Random/RandFlat.h"


#define PI 3.141592653

class TxCalculator
{
  public:
  
   TxCalculator(const edm::Event &iEvent, const edm::EventSetup &iSetup, edm::InputTag trackLabel) ;
  
   double getTx(const reco::Photon clus, double i,double threshold, double innerDR=0,double effRatio=2);
   double getCTx(const reco::Photon clus, double i,double threshold, double innerDR=0, double effRatio=2);
   double getMPT(double ptCut=0, double etaCut=1000);

   double getJurassicArea( double r1, double r2, double width) ;
   double getJt(const reco::Photon cluster, double r1=0.4, double r2=0.04, double jWidth=0.015, double threshold=2);
   double getJct(const reco::Photon cluster, double r1=0.4, double r2=0.04, double jWidth=0.015, double threshold=2);
   
 private:
   
   edm::Handle<reco::TrackCollection>  recCollection;
   CLHEP::RandFlat *theDice;

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
   
};

#endif
