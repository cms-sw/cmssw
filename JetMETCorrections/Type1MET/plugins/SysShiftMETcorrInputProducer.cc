#include "JetMETCorrections/Type1MET/plugins/SysShiftMETcorrInputProducer.h"

#include "DataFormats/METReco/interface/CorrMETData.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/Common/interface/View.h"

SysShiftMETcorrInputProducer::SysShiftMETcorrInputProducer(const edm::ParameterSet& cfg)
  : moduleLabel_(cfg.getParameter<std::string>("@module_label")),
    extractor_(0)
{
  srcMEt_ = cfg.getParameter<edm::InputTag>("srcMEt");
  srcVertices_ = cfg.getParameter<edm::InputTag>("srcVertices");
  srcJets_ = cfg.getParameter<edm::InputTag>("srcJets");
  jetPtThreshold_ = cfg.getParameter<double>("jetPtThreshold");

  edm::ParameterSet cfg_cloned(cfg);
  cfg_cloned.addParameter<std::string>("name", moduleLabel_);
  extractor_ = new SysShiftMETcorrExtractor(cfg_cloned);
  
  produces<CorrMETData>();
}

SysShiftMETcorrInputProducer::~SysShiftMETcorrInputProducer()
{
  delete extractor_;
}

void SysShiftMETcorrInputProducer::produce(edm::Event& evt, const edm::EventSetup& es)
{
  //std::cout << "<SysShiftMETcorrInputProducer::produce>:" << std::endl;

  typedef edm::View<reco::MET> METView;
  edm::Handle<METView> met;
  evt.getByLabel(srcMEt_, met);
  if ( met->size() != 1 ) 
    throw cms::Exception("SysShiftMETcorrExtractor::operator()") 
      << "Failed to find unique MET object !!\n";
  double sumEt = met->front().sumEt();

  edm::Handle<reco::VertexCollection> vertices;
  evt.getByLabel(srcVertices_, vertices);
  int Nvtx = vertices->size();

  typedef edm::View<reco::Candidate> CandidateView;
  edm::Handle<CandidateView> jets;
  evt.getByLabel(srcJets_, jets);
  int numJets = 0;
  for ( CandidateView::const_iterator jet = jets->begin();
	jet != jets->end(); ++jet ) {
    if ( jet->pt() > jetPtThreshold_ ) ++numJets;
  }

  std::auto_ptr<CorrMETData> metCorr(new CorrMETData((*extractor_)(sumEt, Nvtx, numJets)));
  //std::cout << "--> metCorr: Px = " << metCorr->mex << ", Py = " << metCorr->mey << std::endl;
  
  evt.put(metCorr);
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(SysShiftMETcorrInputProducer);
