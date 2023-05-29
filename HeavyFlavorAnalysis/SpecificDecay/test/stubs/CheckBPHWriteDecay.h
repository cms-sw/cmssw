#ifndef HeavyFlavorAnalysis_SpecificDecay_CheckBPHWriteDecay_h
#define HeavyFlavorAnalysis_SpecificDecay_CheckBPHWriteDecay_h

#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHAnalyzerTokenWrapper.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHTrackReference.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/PatCandidates/interface/GenericParticle.h"
#include "DataFormats/PatCandidates/interface/CompositeCandidate.h"

#include <vector>
#include <map>
#include <string>
#include <iostream>
#include <fstream>

class TH1F;
class BPHRecoCandidate;

class CheckBPHWriteDecay : public BPHAnalyzerWrapper<BPHModuleWrapper::one_analyzer> {
public:
  explicit CheckBPHWriteDecay(const edm::ParameterSet& ps);
  ~CheckBPHWriteDecay() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void beginJob() override;
  void analyze(const edm::Event& ev, const edm::EventSetup& es) override;
  void endJob() override;

private:
  std::ostream* osPtr;
  unsigned int runNumber;
  unsigned int evtNumber;
  bool writePtr;

  std::vector<std::string> candsLabel;
  std::vector<BPHTokenWrapper<std::vector<pat::CompositeCandidate> > > candsToken;
  std::map<const pat::CompositeCandidate*, int> idMap;

  typedef edm::Ref<std::vector<reco::Vertex> > vertex_ref;
  typedef edm::Ref<pat::CompositeCandidateCollection> compcc_ref;

  void dump(std::ostream& os, const pat::CompositeCandidate& cand);
  template <class T>
  static void writeCartesian(std::ostream& os, const std::string& s, const T& v, bool endLine = true) {
    os << s << v.x() << " " << v.y() << " " << v.z();
    if (endLine)
      os << std::endl;
    return;
  }
  template <class T>
  static void writeCylindric(std::ostream& os, const std::string& s, const T& v, bool endLine = true) {
    os << s << v.pt() << " " << v.eta() << " " << v.phi();
    if (endLine)
      os << std::endl;
    return;
  }
};

#endif
