#ifndef HGCALShowerBasedEmIdentification_h
#define HGCALShowerBasedEmIdentification_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"

#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "Geometry/FCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DataFormats/ForwardDetId/interface/HGCEEDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCHEDetId.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

#include "TPrincipal.h"

#include <unordered_map>

class HGCALShowerBasedEmIdentification
{

  public:

  HGCALShowerBasedEmIdentification(bool withPileup) ;
  
  bool isEm(const reco::PFCluster& clu); 
  bool cutStartPosition(const reco::PFCluster& clu);
  bool cutSigmaetaeta(const reco::PFCluster& clu);
  bool cutLengthCompatibility(const reco::PFCluster& clu);

  math::XYZPoint startPosition(const reco::PFCluster& clu);
  double sigmaetaeta(const reco::PFCluster& clu);
  double lengthCompatibility(const reco::PFCluster& clu);
  
  void setShowerPosition(const math::XYZPoint & pos);
  void setShowerDirection(const math::XYZVector & dir);

  void reset() {
    showerPos_ = math::XYZPoint(0.,0.,0.);
    showerDir_ = math::XYZVector(0.,0.,0.);
    showerPosIsSet_ = false;
    showerDirIsSet_ = false;
  }

  ~HGCALShowerBasedEmIdentification();
  
private:
  
  math::XYZPoint showerPos_;
  math::XYZVector showerDir_;
  
  bool showerPosIsSet_;
  bool showerDirIsSet_;
   
  bool withPileup_;

  // parameters and cut values, to be moved as configurable values
  double mip_;
  double minenergy_;
  double rmax_;
  double hovereConesize_;
  double cutStartPosition_;
  double cutSigmaetaeta_;
  double cutHoverem_;
  double cutLengthCompatibility_;
  
  // longitudinal parametrisation
  double criticalEnergy_;
  double radiationLength_;
  double meant0_;
  double meant1_;
  double meanalpha0_;
  double meanalpha1_;
  double sigmalnt0_;
  double sigmalnt1_;
  double sigmalnalpha0_;
  double sigmalnalpha1_;
  double corrlnalphalnt0_;
  double corrlnalphalnt1_;

};
#endif
