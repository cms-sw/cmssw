#ifndef HeavyFlavorAnalysis_SpecificDecay_CheckBPHWriteDecay_h
#define HeavyFlavorAnalysis_SpecificDecay_CheckBPHWriteDecay_h

#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHAnalyzerTokenWrapper.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHTrackReference.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/PatCandidates/interface/GenericParticle.h"
#include "DataFormats/PatCandidates/interface/CompositeCandidate.h"

#include <vector>
#include <string>
#include <iostream>
#include <fstream>

class TH1F;
class BPHRecoCandidate;

class CheckBPHWriteDecay:
      public BPHAnalyzerWrapper<BPHModuleWrapper::one_analyzer> {

 public:

  explicit CheckBPHWriteDecay( const edm::ParameterSet& ps );
  virtual ~CheckBPHWriteDecay();

  static void fillDescriptions( edm::ConfigurationDescriptions& descriptions );

  virtual void beginJob();
  virtual void analyze( const edm::Event& ev, const edm::EventSetup& es );
  virtual void endJob();

 private:

  std::ostream* osPtr;
  unsigned int runNumber;
  unsigned int evtNumber;

  std::vector<std::string> candsLabel;
  std::vector< BPHTokenWrapper< std::vector<pat::CompositeCandidate> > >
                           candsToken;

  typedef edm::Ref< std::vector<reco::Vertex> > vertex_ref;
  typedef edm::Ref< pat::CompositeCandidateCollection > compcc_ref;

  static void dump( std::ostream& os, const pat::CompositeCandidate& cand );
  template<class T>
    static void writePosition( std::ostream& os,
                               const std::string& s, const T& p,
                               bool endLine = true ) {
    os << s << p.x() << " "
            << p.y() << " "
            << p.z();
    if ( endLine ) os << std::endl;
    return;
  }
  template<class T>
  static void writeMomentum( std::ostream& os,
                               const std::string& s, const T& p,
                               bool endLine = true ) {
    os << s << p. pt() << " "
            << p.eta() << " "
            << p.phi();
    if ( endLine ) os << std::endl;
    return;
  }

};

#endif
