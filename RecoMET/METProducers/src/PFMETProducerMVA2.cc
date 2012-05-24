#include "RecoMET/METProducers/interface/PFMETProducerMVA2.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/PFMETCollection.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

#include "DataFormats/Common/interface/ValueMap.h"

#include <TMatrixD.h>

#include <algorithm>

using namespace reco;

namespace
{
  template <typename T>
  std::string format_vT(const std::vector<T>& vT)
  {
    std::ostringstream os;

    os << "{ ";

    unsigned numEntries = vT.size();
    for ( unsigned iEntry = 0; iEntry < numEntries; ++iEntry ) {
      os << vT[iEntry];
      if ( iEntry < (numEntries - 1) ) os << ", ";
    }

    os << " }";

    return os.str();
  }

  std::string format_vInputTag(const std::vector<edm::InputTag>& vit)
  {
    std::vector<std::string> vit_string;
    for ( std::vector<edm::InputTag>::const_iterator vit_i = vit.begin();
	  vit_i != vit.end(); ++vit_i ) {
      vit_string.push_back(vit_i->label());
    }
    return format_vT(vit_string);
  }

}

PFMETProducerMVA2::PFMETProducerMVA2(const edm::ParameterSet& cfg)
  : mvaMEtAlgo_(cfg)
{
  srcMVAData_     = cfg.getParameter<edm::InputTag>("srcMVAData");
  srcPFCandidates_ = cfg.getParameter<edm::InputTag>("srcPFCandidates");
  srcLeptons_      = cfg.getParameter<vInputTag>("srcLeptons");

  verbosity_ = ( cfg.exists("verbosity") ) ?
    cfg.getParameter<int>("verbosity") : 0;

  if ( verbosity_ ) {
    std::cout << "<PFMETProducerMVA2::PFMETProducerMVA2>:" << std::endl;
    std::cout << " srcLeptons = " << format_vInputTag(srcLeptons_) << std::endl;
    std::cout << " srcMVAData = " << srcMVAData_.label() << std::endl;
  }

  produces<reco::PFMETCollection>();
}

PFMETProducerMVA2::~PFMETProducerMVA2() { }

void PFMETProducerMVA2::produce(edm::Event& evt, const edm::EventSetup& es)
{

  // Get pre-computed data about the event.
  edm::Handle<reco::JetInfoCollection> jetInfo;
  evt.getByLabel(srcMVAData_, jetInfo);

  // Get the DZs of all the PFCandidates to the PV
  edm::Handle<edm::ValueMap<float> > pfCandidateDZs;
  evt.getByLabel(srcMVAData_, pfCandidateDZs);
  edm::Handle<reco::PFCandidateCollection> pfCandidates;
  evt.getByLabel(srcPFCandidates_, pfCandidates);
  // Construct the MVA input object
  std::vector<mvaMEtUtilities::pfCandInfo>  pfCandInfos;
  pfCandInfos.reserve(pfCandidates->size());
  for (size_t i = 0; i < pfCandidates->size(); ++i) {
    reco::PFCandidateRef pfCandRef(pfCandidates, i);
    float dz = (*pfCandidateDZs)[pfCandRef];
    mvaMEtUtilities::pfCandInfo theInfo;
    theInfo.dZ_ = dz;
    theInfo.p4_ = pfCandRef->p4();
    pfCandInfos.push_back(theInfo);
  }

  edm::Handle<std::vector<reco::Vertex::Point> > vertexInfo;
  evt.getByLabel(srcMVAData_, vertexInfo);

  // "Regular" PFMET
  edm::Handle<reco::PFMETCollection> regularMET;
  evt.getByLabel(srcMVAData_, regularMET);

  reco::PFMET pfMEt = (*regularMET)[0];
  reco::Candidate::LorentzVector pfMEtP4_original = pfMEt.p4();

  // get leptons
  // (excluded from sum over PFCandidates when computing hadronic recoil)
  std::vector<reco::Candidate::LorentzVector> leptonInfo;
  for ( vInputTag::const_iterator srcLeptons_i = srcLeptons_.begin();
	srcLeptons_i != srcLeptons_.end(); ++srcLeptons_i ) {
    edm::Handle<CandidateView> leptons;
    evt.getByLabel(*srcLeptons_i, leptons);
    for ( CandidateView::const_iterator lepton = leptons->begin();
	  lepton != leptons->end(); ++lepton ) {
      leptonInfo.push_back(lepton->p4());
    }
  }

  // compute MVA based MET and estimate of its uncertainty
  mvaMEtAlgo_.setInput(leptonInfo, *jetInfo, pfCandInfos, *vertexInfo);
  mvaMEtAlgo_.evaluateMVA();

  pfMEt.setP4(mvaMEtAlgo_.getMEt());
  pfMEt.setSignificanceMatrix(mvaMEtAlgo_.getMEtCov());

  if ( verbosity_ ) {
    std::cout << "<PFMETProducerMVA2::produce>:" << std::endl;
    std::cout << " PFMET: Pt = " << pfMEtP4_original.pt() << ", phi = " << pfMEtP4_original.phi() << " "
	      << "(Px = " << pfMEtP4_original.px() << ", Py = " << pfMEtP4_original.py() << ")" << std::endl;
    std::cout << " MVA MET: Pt = " << pfMEt.pt() << ", phi = " << pfMEt.phi() << " "
	      << "(Px = " << pfMEt.px() << ", Py = " << pfMEt.py() << ")" << std::endl;
    std::cout << " Cov:" << std::endl;
    mvaMEtAlgo_.getMEtCov().Print();
    mvaMEtAlgo_.print(std::cout);
    std::cout << std::endl;
  }

  // add PFMET object to the event
  std::auto_ptr<reco::PFMETCollection> pfMEtCollection(new reco::PFMETCollection());
  pfMEtCollection->push_back(pfMEt);

  evt.put(pfMEtCollection);
}
