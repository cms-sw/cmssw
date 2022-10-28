#ifndef RecoLocalMuon_GEMRecHit_GEMRecHitProducer_h
#define RecoLocalMuon_GEMRecHit_GEMRecHitProducer_h

/** \class GEMRecHitProducer
 *  Module for GEMRecHit production. 
 *  
 *  \author M. Maggim -- INFN Bari
 */

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"
#include "CondFormats/GEMObjects/interface/GEMMaskedStrips.h"
#include "CondFormats/GEMObjects/interface/GEMDeadStrips.h"
#include "CondFormats/DataRecord/interface/GEMMaskedStripsRcd.h"
#include "CondFormats/DataRecord/interface/GEMDeadStripsRcd.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "RecoLocalMuon/GEMRecHit/interface/GEMRecHitBaseAlgo.h"

class GEMRecHitProducer : public edm::stream::EDProducer<> {
public:
  /// Constructor
  GEMRecHitProducer(const edm::ParameterSet& config);

  /// Destructor
  ~GEMRecHitProducer() override;

  // Method that access the EventSetup for each run
  void beginRun(const edm::Run&, const edm::EventSetup&) override;

  /// The method which produces the rechits
  void produce(edm::Event& event, const edm::EventSetup& setup) override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  // The token to be used to retrieve GEM digis from the event
  edm::EDGetTokenT<GEMDigiCollection> theGEMDigiToken;

  // The reconstruction algorithm
  std::unique_ptr<GEMRecHitBaseAlgo> theAlgo;

  // Object with mask-strips-vector for all the GEM Detectors
  std::unique_ptr<GEMMaskedStrips> theGEMMaskedStripsObj;

  // Object with dead-strips-vector for all the GEM Detectors
  std::unique_ptr<GEMDeadStrips> theGEMDeadStripsObj;

  enum class MaskSource { File, EventSetup } maskSource_, deadSource_;

  edm::ESHandle<GEMGeometry> gemGeom_;

  edm::ESGetToken<GEMGeometry, MuonGeometryRecord> gemGeomToken_;
  edm::ESGetToken<GEMMaskedStrips, GEMMaskedStripsRcd> maskedStripsToken_;
  edm::ESGetToken<GEMDeadStrips, GEMDeadStripsRcd> deadStripsToken_;

  // map of mask and dead strips
  std::map<GEMDetId, EtaPartitionMask> gemMask_;

  bool applyMasking_;
  bool ge21Off_;
};
#endif
