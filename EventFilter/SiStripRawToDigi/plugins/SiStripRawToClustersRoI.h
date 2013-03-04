#ifndef EventFilter_SiStripRawToDigi_SiStripRawToClustersRoI_H
#define EventFilter_SiStripRawToDigi_SiStripRawToClustersRoI_H

#include "CalibFormats/SiStripObjects/interface/SiStripRegionCabling.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/RefGetter.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <string>
#include <memory>
#include "boost/bind.hpp"

/**
   @file EventFilter/SiStripRawToDigi/test/plugins/SiStripRawToClustersRoI.h
   @class SiStripRawToClustersRoI
   @author M.Wingham
*/

class SiStripRawToClustersRoI : public edm::EDProducer {
  
 public:

  typedef edm::LazyGetter<SiStripCluster> LazyGetter;
  typedef edm::RefGetter<SiStripCluster> RefGetter;
  typedef SiStripRegionCabling::Position Position;
  typedef SiStripRegionCabling::SubDet SubDet;

  SiStripRawToClustersRoI( const edm::ParameterSet& );
  ~SiStripRawToClustersRoI();
  
  virtual void beginRun( const edm::Run&, const edm::EventSetup& ) override;
  virtual void produce( edm::Event&, const edm::EventSetup& ) override;
  
 private: 

  void updateCabling( const edm::EventSetup& setup );

  /** Defines allowed physical layer numbers */
  bool physicalLayer( SubDet&, uint32_t& ) const;

  /** Defines regions of interest randomly */
  void random(RefGetter&, edm::Handle<LazyGetter>&) const;
  
  /** Defines regions of interest by superclusters */
  void electrons(const reco::SuperClusterCollection&, RefGetter&, edm::Handle<LazyGetter>&) const;

  /** Defines regions of interest by muons */
  void muons(const reco::TrackCollection&, RefGetter&, edm::Handle<LazyGetter>&) const;

  /** Defines regions of interest by taujets */
  void taujets(const reco::CaloJetCollection&, RefGetter&, edm::Handle<LazyGetter>&) const;

  /** Defines regions of interest by bjets */
  void bjets(const reco::CaloJetCollection&, RefGetter&, edm::Handle<LazyGetter>&) const;

  /** Cabling */
  const SiStripRegionCabling* cabling_;
  
  uint32_t cacheId_;
  
  /** Record of all region numbers */
  std::vector<uint32_t> allregions_;

  /** Layers of SST to unpack (from innermost) */
  int nlayers_;

  /** Booleans to define objects of interest */
  bool global_;
  bool random_;
  bool electrons_;
  bool muons_;
  bool taujets_;
  bool bjets_;

  /** reco module labels to define regions of interest */
  edm::InputTag siStripLazyGetter_;
  edm::InputTag electronBarrelL2_;
  edm::InputTag electronEndcapL2_;
  edm::InputTag muonL2_;
  edm::InputTag taujetL2_;
  edm::InputTag bjetL2_;

  /** deta/dphi to define regions of interest around physics objects */
  double electrondeta_;
  double electrondphi_;
  double muondeta_;
  double muondphi_;
  double taujetdeta_;
  double taujetdphi_;
  double bjetdeta_;
  double bjetdphi_;
};

#endif 

