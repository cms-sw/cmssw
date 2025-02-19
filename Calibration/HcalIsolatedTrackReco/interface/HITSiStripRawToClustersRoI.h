#ifndef Calibration_HcalIsolatedTrackReco_HITSiStripRawToClustersRoI_H
#define Calibration_HcalIsolatedTrackReco_HITSiStripRawToClustersRoI_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/RefGetter.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "CalibFormats/SiStripObjects/interface/SiStripRegionCabling.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include <string>
#include <memory>
#include "boost/bind.hpp"

/**
   @file EventFilter/SiStripRawToDigi/test/plugins/SiStripRawToClustersRoI.h
   @class SiStripRawToClustersRoI
   @author M.Wingham
*/

class HITSiStripRawToClustersRoI : public edm::EDProducer {
  
 public:

  typedef edm::LazyGetter<SiStripCluster> LazyGetter;
  typedef edm::RefGetter<SiStripCluster> RefGetter;
  typedef SiStripRegionCabling::Position Position;
  typedef SiStripRegionCabling::SubDet SubDet;

  HITSiStripRawToClustersRoI( const edm::ParameterSet& );
  ~HITSiStripRawToClustersRoI();
  
  virtual void beginJob( );
  virtual void endJob();
  virtual void produce( edm::Event&, const edm::EventSetup& );
  
 private: 

  /** Defines allowed physical layer numbers */
  bool physicalLayer(SubDet&, uint32_t&) const;

  /** Defines regions of interest randomly */
  void random(RefGetter&, edm::Handle<LazyGetter>&) const;
  
  /** Defines regions of interest by taujets */
  void taujets(const l1extra::L1JetParticleCollection&, RefGetter&, edm::Handle<LazyGetter>&) const;

  /** seeding by isolated pixel tracks*/
  void ptracks(const trigger::TriggerFilterObjectWithRefs&, RefGetter&, edm::Handle<LazyGetter>&) const;
  
  /** Cabling */
  edm::ESHandle<SiStripRegionCabling> cabling_;

  /** Record of all region numbers */
  std::vector<uint32_t> allregions_;

  /** Layers of SST to unpack (from innermost) */
  int nlayers_;

  /** Booleans to define objects of interest */
  bool global_;
  bool random_;
  bool taujets_;
  bool ptrack_;

  /** reco module labels to define regions of interest */
  edm::InputTag siStripLazyGetter_;
  edm::InputTag taujetL1_;
  edm::InputTag ptrackLabel_;
  

  /** deta/dphi to define regions of interest around physics objects */
  double taujetdeta_;
  double taujetdphi_;
  double ptrackEta_;
  double ptrackPhi_;
};

#endif 

