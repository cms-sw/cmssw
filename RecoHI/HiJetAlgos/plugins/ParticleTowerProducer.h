#ifndef ParticleTowerProducer_h
#define ParticleTowerProducer_h


// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

#include "TMath.h"
#include "TRandom.h"



class ParticleTowerProducer : public edm::EDProducer {
 public:
  explicit ParticleTowerProducer(const edm::ParameterSet&);
  ~ParticleTowerProducer();
  
 private:
  virtual void beginJob() ;
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  void resetTowers(edm::Event& iEvent,const edm::EventSetup& iSetup);
  DetId getNearestTower(const reco::PFCandidate & in) const;
  DetId getNearestTower(double eta, double phi) const;
  uint32_t denseIndex(int ieta, int iphi, double eta) const;
  int eta2ieta(double eta) const;
  int phi2iphi(double phi, int ieta) const;
  
  // ----------member data ---------------------------
  
  edm::InputTag src_;
  bool useHF_;
  
  std::map<DetId,double> towers_;
  
  
  double PI;
  TRandom* random_;
  
  CaloGeometry const *  geo_;                       // geometry


  static const double etatow[];
  static const double etacent[];
  double etaedge[42];
  
  
 
  
};
#endif
