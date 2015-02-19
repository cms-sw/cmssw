#ifndef RecoMuon_L2MuonIsolationProducer_H
#define RecoMuon_L2MuonIsolationProducer_H

/**  \class L2MuonIsolationProducer
 * 
 *   L2 HLT muon isolation producer
 *
 *   \author  J.Alcaraz
 */

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "PhysicsTools/IsolationAlgos/interface/IsoDepositExtractor.h"
#include "RecoMuon/MuonIsolation/interface/MuIsoBaseIsolator.h"

#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"

class L2MuonIsolationProducer : public edm::stream::EDProducer<> {

 public:

  /// constructor with config
  L2MuonIsolationProducer(const edm::ParameterSet&);
  
  /// destructor
  virtual ~L2MuonIsolationProducer(); 

  /// ParameterSet descriptions
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

  /// Produce isolation maps
  virtual void produce(edm::Event&, const edm::EventSetup&) override;
  // ex virtual void reconstruct();

 private:
  
  // Muon track Collection Label
  edm::InputTag theSACollectionLabel;
  edm::EDGetTokenT<reco::RecoChargedCandidateCollection> theSACollectionToken;

  // Option to write MuIsoDeposits into the event
  bool optOutputDecision;

  // Option to write MuIsoDeposit sum into the event
  bool optOutputIsolatorFloat;

  // MuIsoExtractor
  reco::isodeposit::IsoDepositExtractor* theExtractor;

  // muon isolator 
  muonisolation::MuIsoBaseIsolator * theDepositIsolator;

};

#endif
