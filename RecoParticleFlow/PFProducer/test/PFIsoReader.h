#ifndef RECOPARTICLEFLOW_PFISOREADER_H
#define RECOPARTICLEFLOW_PFISOREADER_H
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include <iostream>
#include <string>
#include <map>


class PFIsoReader : public edm::EDAnalyzer
{
 public:
  explicit PFIsoReader(const edm::ParameterSet&);
  ~PFIsoReader();
  virtual void beginRun(edm::Run const&, edm::EventSetup const& );
  virtual void analyze(const edm::Event & iEvent,const edm::EventSetup & c);

 private:
  typedef std::vector< edm::Handle< edm::ValueMap<reco::IsoDeposit> > > IsoDepositMaps;
  void printIsoDeposits(const IsoDepositMaps & depmap, const reco::PFCandidatePtr & ptr) const;
  
  edm::InputTag inputTagGsfElectrons_;
  edm::InputTag inputTagPhotons_;
  edm::InputTag inputTagValueMapPhotons_;
  edm::InputTag inputTagValueMapMerged_;
  edm::InputTag inputTagValueMapElectrons_;
  edm::InputTag inputTagPFCandidates_;
  std::vector<edm::InputTag> inputTagElectronIsoDeposits_;
  std::vector<edm::InputTag> inputTagPhotonIsoDeposits_;
  bool useValueMaps_;
};
#endif
