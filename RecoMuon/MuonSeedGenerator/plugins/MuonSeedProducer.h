#ifndef RecoMuon_MuonSeedProducer_H
#define RecoMuon_MuonSeedProducer_H

/** \class MuonSeedProducer
 *
 * This EDProducer produces a collection of muon seeds.  
 * To do so, it forms pairs of CSC and/or DT segments and look
 * at the properties of the segment pair (eta, dphi)
 * first to estimate the properties of the muon, and segment direction
 * in case where there is only one segment available.
 *
 * \author Dominique Fortin - UCR
 *
 */

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "RecoMuon/DetLayers/interface/MuonDetLayerGeometry.h"
#include "RecoMuon/Records/interface/MuonRecoGeometryRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

class MuonSeedBuilder;

class MuonSeedProducer : public edm::stream::EDProducer<> {
public:
  /// Constructor
  MuonSeedProducer(const edm::ParameterSet&);

  /// Destructor
  ~MuonSeedProducer() override;

  // Operations

  /// Get event properties to send to builder to fill seed collection
  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  // This Producer private debug flag
  bool debug;

  /// Builder where seeds are formed
  MuonSeedBuilder* muonSeedBuilder_;

  edm::ESGetToken<MuonDetLayerGeometry, MuonRecoGeometryRecord> muonLayersToken_;
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magFieldToken_;
};

#endif
