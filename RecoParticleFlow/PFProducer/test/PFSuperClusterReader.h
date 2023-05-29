#ifndef RECOPARTICLEFLOW_PFSUPERCLUSTERREADER_H
#define RECOPARTICLEFLOW_PFSUPERCLUSTERREADER_H
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/PreshowerClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include <iostream>
#include <sstream>
#include <string>
#include <map>

class PFSuperClusterReader : public edm::one::EDAnalyzer<> {
public:
  explicit PFSuperClusterReader(const edm::ParameterSet&);
  ~PFSuperClusterReader() override = default;
  void analyze(edm::Event const&, edm::EventSetup const&) override;

private:
  const edm::InputTag inputTagGSFTracks_;
  const edm::InputTag inputTagValueMapSC_;
  const edm::InputTag inputTagValueMapMVA_;
  const edm::InputTag inputTagPFCandidates_;
  const edm::EDGetTokenT<reco::GsfTrackCollection> trackToken_;
  const edm::EDGetTokenT<reco::PFCandidateCollection> pfCandToken_;
  const edm::EDGetTokenT<edm::ValueMap<reco::SuperClusterRef> > pfClusToken_;
  const edm::EDGetTokenT<edm::ValueMap<float> > pfMapToken_;

  const reco::PFCandidate* findPFCandidate(const reco::PFCandidateCollection* coll, const reco::GsfTrackRef& ref);
};
#endif
