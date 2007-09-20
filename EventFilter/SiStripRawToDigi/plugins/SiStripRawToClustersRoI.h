#ifndef EventFilter_SiStripRawToDigi_SiStripRawToClustersRoI_H
#define EventFilter_SiStripRawToDigi_SiStripRawToClustersRoI_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/SiStripCommon/interface/SiStripRefGetter.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "CalibFormats/SiStripObjects/interface/SiStripRegionCabling.h"
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

  typedef edm::SiStripLazyGetter<SiStripCluster> LazyGetter;
  typedef edm::SiStripRefGetter<SiStripCluster> RefGetter;

  SiStripRawToClustersRoI( const edm::ParameterSet& );
  ~SiStripRawToClustersRoI();
  
  virtual void beginJob( const edm::EventSetup& );
  virtual void endJob();
  virtual void produce( edm::Event&, const edm::EventSetup& );
  
 private: 

  /** Defines allowed physical layer numbers */
  bool physicalLayer(SiStripRegionCabling::SubDet&, uint32_t&) const;

  /** Defines regions of interest randomly */
  void random(RefGetter&, edm::Handle<LazyGetter>&) const;

  /** Defines ALL regions of interest */
  void global(RefGetter&, edm::Handle<LazyGetter>&) const;
  
  /** Defines regions of interest by superclusters */
  void superclusters(const reco::SuperClusterCollection&, RefGetter&, edm::Handle<LazyGetter>&) const;

  /** Defines regions of interest by muons */
  void muons(const reco::TrackCollection&, RefGetter&, edm::Handle<LazyGetter>&) const;

  /** Defines regions of interest by taus */
  void taus(const reco::CaloJetCollection&, RefGetter&, edm::Handle<LazyGetter>&) const;

  /** Defines regions of interest by bjets */
  void bjets(const reco::CaloJetCollection&, RefGetter&, edm::Handle<LazyGetter>&) const;

  /** Cabling */
  edm::ESHandle<SiStripRegionCabling> cabling_;

  /** Input module label of SiStripLazyGetter */
  std::string inputModuleLabel_;

  /** Layers of SST to unpack (from innermost) */
  int nlayers_;

  /** Booleans to define objects of interest */
  bool global_;
  bool random_;
  bool electrons_;
  bool muons_;
  bool taus_;
  bool bjets_;

  /** reco module labels to define regions of interest */
  std::string electronBarrelModule_;
  std::string electronBarrelProduct_;
  std::string electronEndcapModule_;
  std::string electronEndcapProduct_;
  std::string muonModule_;
  std::string muonProduct_;
  std::string tauModule_;
  std::string tauProduct_;
  std::string bjetModule_;
  std::string bjetProduct_;

  /** deta/dphi to define regions of interest around physics objects */
  double electrondeta_;
  double electrondphi_;
  double muondeta_;
  double muondphi_;
  double taudeta_;
  double taudphi_;
  double bjetdeta_;
  double bjetdphi_;
};

#endif //  EventFilter_SiStripRawToDigi_SiStripRawToClustersRoI_H

