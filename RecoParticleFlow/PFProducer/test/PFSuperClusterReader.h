#ifndef RECOPARTICLEFLOW_PFSUPERCLUSTERREADER_H
#define RECOPARTICLEFLOW_PFSUPERCLUSTERREADER_H
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/PreshowerClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include <iostream>
#include <string>
#include <map>


class PFSuperClusterReader : public edm::EDAnalyzer
{
 public:
  explicit PFSuperClusterReader(const edm::ParameterSet&);
  ~PFSuperClusterReader();
  virtual void beginRun(edm::Run const&, edm::EventSetup const& );
  virtual void analyze(const edm::Event & iEvent,const edm::EventSetup & c);

 private:
  edm::InputTag inputTagGSFTracks_;
  edm::InputTag inputTagValueMapSC_;
  edm::InputTag inputTagValueMapMVA_;
  edm::InputTag inputTagPFCandidates_;

  const reco::PFCandidate * findPFCandidate(const reco::PFCandidateCollection * coll,const reco::GsfTrackRef& ref);
};
#endif
