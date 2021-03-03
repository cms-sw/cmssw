#ifndef HLTScoutingEgammaProducer_h
#define HLTScoutingEgammaProducer_h

// -*- C++ -*-
//
// Package:    HLTrigger/Egamma
// Class:      HLTScoutingEgammaProducer
//
/**\class HLTScoutingEgammaProducer HLTScoutingEgammaProducer.h HLTScoutingEgammaProducer.h

Description: Producer for ScoutingElectron and ScoutingPhoton

*/
//
// Original Author:  David G. Sheffield (Rutgers)
//         Created:  Mon, 20 Jul 2015
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/getRef.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

#include "DataFormats/Scouting/interface/Run3ScoutingElectron.h"
#include "DataFormats/Scouting/interface/Run3ScoutingPhoton.h"

#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalTools.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/Math/interface/libminifloat.h"

class HLTScoutingEgammaProducer : public edm::global::EDProducer<> {
  typedef edm::AssociationMap<edm::OneToValue<std::vector<reco::RecoEcalCandidate>, float, unsigned int> >
      RecoEcalCandMap;

public:
  explicit HLTScoutingEgammaProducer(const edm::ParameterSet&);
  ~HLTScoutingEgammaProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID sid, edm::Event& iEvent, edm::EventSetup const& setup) const final;

  const edm::EDGetTokenT<reco::RecoEcalCandidateCollection> EgammaCandidateCollection_;
  const edm::EDGetTokenT<reco::GsfTrackCollection> EgammaGsfTrackCollection_;
  const edm::EDGetTokenT<RecoEcalCandMap> SigmaIEtaIEtaMap_;
  const edm::EDGetTokenT<RecoEcalCandMap> R9Map_;
  const edm::EDGetTokenT<RecoEcalCandMap> HoverEMap_;
  const edm::EDGetTokenT<RecoEcalCandMap> DetaMap_;
  const edm::EDGetTokenT<RecoEcalCandMap> DphiMap_;
  const edm::EDGetTokenT<RecoEcalCandMap> MissingHitsMap_;
  const edm::EDGetTokenT<RecoEcalCandMap> OneOEMinusOneOPMap_;
  const edm::EDGetTokenT<RecoEcalCandMap> EcalPFClusterIsoMap_;
  const edm::EDGetTokenT<RecoEcalCandMap> EleGsfTrackIsoMap_;
  const edm::EDGetTokenT<RecoEcalCandMap> HcalPFClusterIsoMap_;

  //const EcalClusterLazyTools::ESGetTokens ecalClusterToolsESGetTokens_;

  const double egammaPtCut;
  const double egammaEtaCut;
  const double egammaHoverECut;
  const int mantissaPrecision;
  const bool saveRecHitTiming;
  const int rechitMatrixSize;

  const edm::EDGetTokenT<EcalRecHitCollection> ecalRechitEB_;
  const edm::EDGetTokenT<EcalRecHitCollection> ecalRechitEE_;
  edm::ESGetToken<CaloTopology, CaloTopologyRecord> topologyToken_;
};

#endif
