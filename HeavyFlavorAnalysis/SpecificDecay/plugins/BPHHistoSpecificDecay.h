#ifndef HeavyFlavorAnalysis_SpecificDecay_BPHHistoSpecificDecay_h
#define HeavyFlavorAnalysis_SpecificDecay_BPHHistoSpecificDecay_h

#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHAnalyzerTokenWrapper.h"
#include "DataFormats/PatCandidates/interface/CompositeCandidate.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Ref.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include <string>

class TH1F;
class TVector3;

namespace reco {
  class Candidate;
  class Vertex;
}

class BPHHistoSpecificDecay:
      public BPHAnalyzerWrapper<BPHModuleWrapper::one_analyzer> {

 public:

  explicit BPHHistoSpecificDecay( const edm::ParameterSet& ps );
  virtual ~BPHHistoSpecificDecay();

  static void fillDescriptions( edm::ConfigurationDescriptions& descriptions );

  virtual void beginJob();
  virtual void analyze( const edm::Event& ev, const edm::EventSetup& es );
  virtual void endJob();

  class CandidateSelect {
   public:
    virtual ~CandidateSelect() {}
    virtual bool accept( const pat::CompositeCandidate& cand,
                         const reco::Vertex* pv = 0 ) const = 0 ;
  };

 private:

  std::string oniaCandsLabel;
  std::string   sdCandsLabel;
  std::string   ssCandsLabel;
  std::string   buCandsLabel;
  std::string   bdCandsLabel;
  std::string   bsCandsLabel;
  BPHTokenWrapper< std::vector<pat::CompositeCandidate> > oniaCandsToken;
  BPHTokenWrapper< std::vector<pat::CompositeCandidate> >   sdCandsToken;
  BPHTokenWrapper< std::vector<pat::CompositeCandidate> >   ssCandsToken;
  BPHTokenWrapper< std::vector<pat::CompositeCandidate> >   buCandsToken;
  BPHTokenWrapper< std::vector<pat::CompositeCandidate> >   bdCandsToken;
  BPHTokenWrapper< std::vector<pat::CompositeCandidate> >   bsCandsToken;
  bool useOnia;
  bool useSd;
  bool useSs;
  bool useBu;
  bool useBd;
  bool useBs;

  edm::Service<TFileService> fs;
  std::map<std::string,TH1F*> histoMap;

  CandidateSelect*  phiBasicSelect;
  CandidateSelect* jPsiBasicSelect;
  CandidateSelect* psi2BasicSelect;
  CandidateSelect*  upsBasicSelect;
  CandidateSelect* oniaVertexSelect;
  CandidateSelect* oniaDaughterSelect;

  CandidateSelect* buJPsiBasicSelect;
  CandidateSelect* buVertexSelect;
  CandidateSelect* buJPsiDaughterSelect;

  CandidateSelect* bdJPsiBasicSelect;
  CandidateSelect* bdKx0BasicSelect;
  CandidateSelect* bdVertexSelect;
  CandidateSelect* bdJPsiDaughterSelect;

  CandidateSelect* bsJPsiBasicSelect;
  CandidateSelect* bsPhiBasicSelect;
  CandidateSelect* bsVertexSelect;
  CandidateSelect* bsJPsiDaughterSelect;

  double buKPtMin;

  void fillHisto   ( const std::string& name,
                     const pat::CompositeCandidate& cand );
  void fillHisto   ( const std::string& name, float x );
  void createHisto ( const std::string& name,
                     int nbin, float hmin, float hmax );

};

#endif
