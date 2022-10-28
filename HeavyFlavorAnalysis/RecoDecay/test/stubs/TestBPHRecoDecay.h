#ifndef HeavyFlavorAnalysis_RecoDecay_TestBPHRecoDecay_h
#define HeavyFlavorAnalysis_RecoDecay_TestBPHRecoDecay_h

#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHAnalyzerTokenWrapper.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHTrackReference.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/GenericParticle.h"
#include "DataFormats/PatCandidates/interface/CompositeCandidate.h"

#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"

#include <string>
#include <iostream>
#include <fstream>

class TH1F;
class BPHRecoCandidate;

class TestBPHRecoDecay : public BPHAnalyzerWrapper<BPHModuleWrapper::one_analyzer> {
public:
  explicit TestBPHRecoDecay(const edm::ParameterSet& ps);
  ~TestBPHRecoDecay() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void beginJob() override;
  void analyze(const edm::Event& ev, const edm::EventSetup& es) override;
  void endJob() override;

private:
  std::string patMuonLabel;
  std::string ccCandsLabel;
  std::string pfCandsLabel;
  std::string pcCandsLabel;
  std::string gpCandsLabel;

  // token wrappers to allow running both on "old" and "new" CMSSW versions
  BPHESTokenWrapper<TransientTrackBuilder, TransientTrackRecord> ttBToken;
  BPHTokenWrapper<pat::MuonCollection> patMuonToken;
  BPHTokenWrapper<std::vector<pat::CompositeCandidate> > ccCandsToken;
  BPHTokenWrapper<std::vector<reco::PFCandidate> > pfCandsToken;
  BPHTokenWrapper<std::vector<BPHTrackReference::candidate> > pcCandsToken;
  BPHTokenWrapper<std::vector<pat::GenericParticle> > gpCandsToken;

  bool usePM;
  bool useCC;
  bool usePF;
  bool usePC;
  bool useGP;

  std::string outDump;
  std::string outHist;

  std::ostream* fPtr;

  std::map<std::string, TH1F*> histoMap;

  void dumpRecoCand(const std::string& name, const BPHRecoCandidate* cand);
  void fillHisto(const std::string& name, const BPHRecoCandidate* cand);
  void fillHisto(const std::string& name, float x);

  void createHisto(const std::string& name, int nbin, float hmin, float hmax);
};

#endif
