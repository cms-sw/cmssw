#ifndef L1PFTAU_PRDC_H
#define L1PFTAU_PRDC_H

#include <iostream>
#include <math.h>
#include <vector>
#include <list>
#include <TLorentzVector.h>

////////////////////
// FRAMEWORK HEADERS
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/Ref.h"

//L1 TPG Legacy
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "CondFormats/L1TObjects/interface/L1CaloHcalScale.h"
#include "CondFormats/DataRecord/interface/L1CaloHcalScaleRcd.h"
#include "DataFormats/HcalDigi/interface/HcalTriggerPrimitiveDigi.h"

#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/EcalAlgo/interface/EcalBarrelGeometry.h"
#include "Geometry/EcalAlgo/interface/EcalEndcapGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"

//Geometry
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/HcalTowerAlgo/interface/HcalTrigTowerGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"

#include "DataFormats/L1Trigger/interface/L1PFTau.h"
#include "DataFormats/Phase2L1ParticleFlow/interface/PFCandidate.h"
#include "L1Trigger/Phase2L1Taus/interface/TauMapper.h"

using namespace l1t;

//#define tracker_eta 3.5
//#define tau_size_eta 0.7
//#define tau_size_phi 0.7

typedef struct{
  float three_prong_seed;
  float three_prong_delta_r;
  float three_prong_max_delta_Z;
  float isolation_delta_r;
  float one_prong_seed;
  float dummy;
  float input_EoH_cut;
  float max_neighbor_strip_dist;
  float min_strip;
  float eg_strip_merge;
} algo_config_t;


typedef struct
{
  float et = 0;
  float eta = 0;
  float phi = 0;
} strip_t;



typedef L1PFTau pftau_t;

using std::vector;
using namespace l1t;

class L1PFTauProducer : public edm::EDProducer {
   public:
  explicit L1PFTauProducer(const edm::ParameterSet&);

  ~L1PFTauProducer();

   private:
  //virtual void produce(edm::Event&, const edm::EventSetup&) override;

  tauMapperCollection tauCandidates;
  
  void createTaus(tauMapperCollection &inputCollection);
  void tau_cand_sort(tauMapperCollection tauCandidates, std::unique_ptr<L1PFTauCollection> &newL1PFTauCollection, unsigned int nCands);

  /// ///////////////// ///
  /// MANDATORY METHODS ///
  virtual void beginRun( const edm::Run& run, const edm::EventSetup& iSetup );
  virtual void endRun( const edm::Run& run, const edm::EventSetup& iSetup );
  virtual void produce( edm::Event& iEvent, const edm::EventSetup& iSetup );
  /// ///////////////// ///

  bool debug;
  int input_EoH_cut_;
  int input_HoE_cut_;
  int input_min_n_stubs_;
  int input_max_chi2_; 
  float three_prong_delta_r_;
  float three_prong_max_delta_Z_;
  float isolation_delta_r_;
  edm::InputTag L1TrackInputTag;
  edm::EDGetTokenT< vector<l1t::PFCandidate> > L1ClustersToken_;
  edm::EDGetTokenT< std::vector< TTTrack< Ref_Phase2TrackerDigi_ > > > ttTrackToken_;
  edm::EDGetTokenT< vector<l1t::PFCandidate> > L1PFToken_;
  edm::EDGetTokenT< vector<l1t::PFCandidate> > L1NeutralToken_;


};


#endif
