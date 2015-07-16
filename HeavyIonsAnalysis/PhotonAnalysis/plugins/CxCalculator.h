#ifndef CxCalculator_h
#define CxCalculator_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"


class CxCalculator
{
  public:
  
   CxCalculator(const edm::Event &iEvent, const edm::EventSetup &iSetup, edm::InputTag barrelLabel, edm::InputTag endcapLabel) ;
  
   double getJurassicArea( double r1, double r2, double width) ;
   double getBCMax(const reco::SuperClusterRef clus,int i);
   double getCx(const reco::SuperClusterRef clus, double i,double threshold);
   double getCxRemoveSC(const reco::SuperClusterRef clus, double i,double threshold);
   double getCCx(const reco::SuperClusterRef clus, double i,double threshold); // background subtracted Cx
   double getJc (const reco::SuperClusterRef cluster, double r1=0.4, double r2=0.06, double jWidth=0.04, double threshold=0);
   double getJcc(const reco::SuperClusterRef cluster, double r1=0.4, double r2=0.06, double jWidth=0.04, double threshold=0);
   double getCCxRemoveSC(const reco::SuperClusterRef clus, double i,double threshold); // background subtracted Cx
   double getCorrection(const reco::SuperClusterRef clus, double i,double j,double threshold); // background subtracted Cx
   double getAvgBCEt(const reco::SuperClusterRef clus, double eta,double phi1, double phi2,double threshold); // background subtracted Cx
   double getNBC(const reco::SuperClusterRef clus, double eta,double phi1, double phi2,double threshold); // background subtracted Cx
   bool checkUsed(const reco::SuperClusterRef clus, const reco::BasicCluster* clu);

  private:
   
   const reco::BasicClusterCollection *fEBclusters_;
   const reco::BasicClusterCollection *fEEclusters_;
   const CaloGeometry                 *geometry_;
   
};

#endif
