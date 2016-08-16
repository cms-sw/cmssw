#ifndef TestBaseNtuple_h
#define TestBaseNtuple_h

#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHAnalyzerTokenWrapper.h"
#include "DataFormats/PatCandidates/interface/CompositeCandidate.h"

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Ref.h"

#include <string>

class TH1F;
class TVector3;

namespace reco {
  class Candidate;
  class Vertex;
}

class BPHHistoSpecificDecay: public BPHAnalyzerWrapper<edm::EDAnalyzer> {

 public:

  explicit BPHHistoSpecificDecay( const edm::ParameterSet& ps );
  virtual ~BPHHistoSpecificDecay();

  virtual void beginJob();
  virtual void analyze( const edm::Event& ev, const edm::EventSetup& es );
  virtual void endJob();

  class CandidateSelect {
   public:
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

  std::string outHist;
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

  typedef edm::Ref< std::vector<reco::Vertex> > vertex_ref;
  typedef edm::Ref< pat::CompositeCandidateCollection > compcc_ref;

  static std::string getParameter( const edm::ParameterSet& ps,
                                   const std::string& name );

  void fillHisto   ( const std::string& name,
                     const pat::CompositeCandidate& cand );
  void fillHisto   ( const std::string& name, float x );
  void createHisto ( const std::string& name,
                     int nbin, float hmin, float hmax );

};

#endif
