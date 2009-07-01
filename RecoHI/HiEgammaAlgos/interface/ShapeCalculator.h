#ifndef ShapeCalculator_h
#define ShapeCalculator_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "TLorentzVector.h"

class SuperFoxWolfram{
  private:
    double sum[10];
  public:
   SuperFoxWolfram();
   ~SuperFoxWolfram();
   void fill(std::vector<TLorentzVector>&, std::vector<TLorentzVector>&);
   double R(int i);
};

class ShapeCalculator
{
  public:
  
   ShapeCalculator(const edm::Event &iEvent, const edm::EventSetup &iSetup) ;
  
   double getBCMax(const reco::SuperCluster* clus,int i);
   double getCx(const reco::SuperCluster* clus, double i,double threshold);
   double getCxRemoveSC(const reco::SuperCluster* clus, double i,double threshold);
   double getCCx(const reco::SuperCluster* clus, double i,double threshold); // background subtracted Cx
   double getCCxRemoveSC(const reco::SuperCluster* clus, double i,double threshold); // background subtracted Cx
   double getCorrection(const reco::SuperCluster* clus, double i,double j,double threshold); // background subtracted Cx
   bool checkUsed(const reco::SuperCluster* clus, const reco::BasicCluster* clu);

    
   int calculate(const reco::SuperCluster* cluster);

   TLorentzVector thrust();
   TLorentzVector thrust(std::vector<TLorentzVector>& ptl);
   int thrust(int ntrk, double ptrk[][3], double* thr, double tvec[3], int itrk[]);

   double Sper(TLorentzVector& thr);
   double Sper(std::vector<TLorentzVector>& ptl, TLorentzVector& Bthr);
   double Moment();
   double Moment(std::vector<TLorentzVector>& ptl);
   int spherp(int ntrk, double ptrk[][3], double *sper, double jet[3]);
   std::vector<TLorentzVector>& getVector() { return recHitPosCollection_; } 
   std::vector<TLorentzVector>& getVector2() { return recHitPosCollection2_; } 
   

  private:
   
   const reco::BasicClusterCollection *fEBclusters_;
   const reco::BasicClusterCollection *fEEclusters_;
   const EcalRecHitCollection   *fEBHit_;
   const EcalRecHitCollection   *fEEHit_;
   const CaloGeometry                 *geometry_;
   std::vector<TLorentzVector> recHitPosCollection_;
   std::vector<TLorentzVector> recHitPosCollection2_;

};



#endif
