#include "RecoMET/METProducers/interface/PFMETProducerMVA.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "RecoMET/METAlgorithms/interface/METAlgo.h" 
#include "RecoMET/METAlgorithms/interface/PFSpecificAlgo.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/PFMETCollection.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/METReco/interface/CommonMETData.h"

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

  void printJets(std::ostream& stream, const reco::PFJetCollection& jets)
  {
    unsigned numJets = jets.size();
    for ( unsigned iJet = 0; iJet < numJets; ++iJet ) {
      const reco::Candidate::LorentzVector& jetP4 = jets.at(iJet).p4();
      stream << " #" << iJet << ": Pt = " << jetP4.pt() << "," 
	     << " eta = " << jetP4.eta() << ", phi = " << jetP4.phi() << std::endl;
    }
  }
}

PFMETProducerMVA::PFMETProducerMVA(const edm::ParameterSet& cfg) 
  : mvaMEtAlgo_(cfg),
    //looseJetIdAlgo_(0),
    mvaJetIdAlgo_(cfg)
{
  srcCorrJets_     = cfg.getParameter<edm::InputTag>("srcCorrJets");
  srcUncorrJets_   = cfg.getParameter<edm::InputTag>("srcUncorrJets");
  srcPFCandidates_ = cfg.getParameter<edm::InputTag>("srcPFCandidates");
  srcVertices_     = cfg.getParameter<edm::InputTag>("srcVertices");
  srcLeptons_      = cfg.getParameter<vInputTag>("srcLeptons");
  srcRho_          = cfg.getParameter<edm::InputTag>("srcRho");

  globalThreshold_ = cfg.getParameter<double>("globalThreshold");

  minCorrJetPt_    = cfg.getParameter<double>("minCorrJetPt");

  //edm::ParameterSet cfgPFJetIdAlgo;
  //cfgPFJetIdAlgo.addParameter<std::string>("version", "FIRSTDATA");
  //cfgPFJetIdAlgo.addParameter<std::string>("quality", "LOOSE");
  // looseJetIdAlgo_ = new PFJetIDSelectionFunctor(cfgPFJetIdAlgo);

  verbosity_ = ( cfg.exists("verbosity") ) ?
    cfg.getParameter<int>("verbosity") : 0;

  if ( verbosity_ ) {
    std::cout << "<PFMETProducerMVA::PFMETProducerMVA>:" << std::endl;
    std::cout << " srcCorrJets = " << srcCorrJets_.label() << std::endl;
    std::cout << " srcUncorrJets = " << srcUncorrJets_.label() << std::endl;
    std::cout << " srcPFCandidates = " << srcPFCandidates_.label() << std::endl;
    std::cout << " srcVertices = " << srcVertices_.label() << std::endl;
    std::cout << " srcLeptons = " << format_vInputTag(srcLeptons_) << std::endl;
    std::cout << " srcRho = " << srcVertices_.label() << std::endl;
  }

  produces<reco::PFMETCollection>();
}

PFMETProducerMVA::~PFMETProducerMVA()
{
  //delete looseJetIdAlgo_;
}

void PFMETProducerMVA::produce(edm::Event& evt, const edm::EventSetup& es) 
{ 
  // get jets (corrected and uncorrected)
  edm::Handle<reco::PFJetCollection> corrJets;
  evt.getByLabel(srcCorrJets_, corrJets);

  edm::Handle<reco::PFJetCollection> uncorrJets;
  evt.getByLabel(srcUncorrJets_, uncorrJets);

  // get PFCandidates
  edm::Handle<reco::PFCandidateCollection> pfCandidates;
  evt.getByLabel(srcPFCandidates_, pfCandidates);

  typedef edm::View<reco::Candidate> CandidateView;
  edm::Handle<CandidateView> pfCandidates_view;
  evt.getByLabel(srcPFCandidates_, pfCandidates_view);

  // get leptons
  // (excluded from sum over PFCandidates when computing hadronic recoil)
  //int lNMu = 0; 
  int lId  = 0;
  std::vector<mvaMEtUtilities::leptonInfo> leptonInfo;
  for ( vInputTag::const_iterator srcLeptons_i = srcLeptons_.begin();
	srcLeptons_i != srcLeptons_.end(); ++srcLeptons_i ) {
    std::cout << *srcLeptons_i << std::endl;
    edm::Handle<CandidateView> leptons;
    evt.getByLabel(*srcLeptons_i, leptons);
    for ( CandidateView::const_iterator lepton1 = leptons->begin();
	  lepton1 != leptons->end(); ++lepton1 ) {
      bool pMatch = false;
      for ( CandidateView::const_iterator lepton2 = leptons->begin();
            lepton2 != leptons->end(); ++lepton2 ) {
        if(lepton1 == lepton2) continue;
        if(deltaR(lepton1->p4(),lepton2->p4()) < 0.5 &&
           lepton1->pt() < lepton2->pt()) pMatch  = true;
      }
      if(pMatch) continue;
      mvaMEtUtilities::leptonInfo pLeptonInfo;
      pLeptonInfo.p4_          = lepton1->p4();
      pLeptonInfo.chargedFrac_ = chargedFrac(&(*lepton1));
      //if(lepton->pt() > 20. && (chargedFrac(&(*lepton)) < 1. || lId == 0) ) lNMu++;
      leptonInfo.push_back(pLeptonInfo); 
    }
    lId++;
  }
  //if(lNMu == 2) std::cout << "=====> Di Muon Cand =======>"  << leptonInfo[0].p4_.pt() << " -- " << leptonInfo[1].p4_.pt() << std::endl;

  // get vertices
  edm::Handle<reco::VertexCollection> vertices;
  evt.getByLabel(srcVertices_, vertices); 
  // take vertex with highest sum(trackPt) as the vertex of the "hard scatter" interaction
  // (= first entry in vertex collection)
  const reco::Vertex* hardScatterVertex = ( vertices->size() >= 1 ) ?
    &(vertices->front()) : 0;

  // get average energy density in the event
  edm::Handle<double> rho;
  evt.getByLabel(srcRho_, rho);

  // reconstruct "standard" particle-flow missing Et
  CommonMETData pfMEt_data;
  metAlgo_.run(pfCandidates_view, &pfMEt_data, globalThreshold_);
  reco::PFMET pfMEt = pfMEtSpecificAlgo_.addInfo(pfCandidates_view, pfMEt_data);
  reco::Candidate::LorentzVector pfMEtP4_original = pfMEt.p4();
  
  // compute objects specific to MVA based MET reconstruction
  std::vector<mvaMEtUtilities::JetInfo>    jetInfo         = computeJetInfo(*uncorrJets, *corrJets, *vertices, hardScatterVertex, *rho);
  std::vector<mvaMEtUtilities::pfCandInfo> pfCandidateInfo = computePFCandidateInfo(*pfCandidates, hardScatterVertex);
  std::vector<reco::Vertex::Point>         vertexInfo      = computeVertexInfo(*vertices);

  // compute MVA based MET and estimate of its uncertainty
  mvaMEtAlgo_.setInput(leptonInfo, jetInfo, pfCandidateInfo, vertexInfo);
  mvaMEtAlgo_.evaluateMVA();

  pfMEt.setP4(mvaMEtAlgo_.getMEt());
  pfMEt.setSignificanceMatrix(mvaMEtAlgo_.getMEtCov());
  //std::cout << " MVA===> " << pfMEt.pt() << " -- " << pfMEt.phi() << std::endl;

  if ( verbosity_ ) {
    std::cout << "<PFMETProducerMVA::produce>:" << std::endl;
    std::cout << " PFMET: Pt = " << pfMEtP4_original.pt() << ", phi = " << pfMEtP4_original.phi() << " "
	      << "(Px = " << pfMEtP4_original.px() << ", Py = " << pfMEtP4_original.py() << ")" << std::endl;
    std::cout << " MVA MET: Pt = " << pfMEt.pt() << ", phi = " << pfMEt.phi() << " "
	      << "(Px = " << pfMEt.px() << ", Py = " << pfMEt.py() << ")" << std::endl;
    std::cout << " Cov:" << std::endl;
    mvaMEtAlgo_.getMEtCov().Print();
    mvaMEtAlgo_.print(std::cout);
    //std::cout << "corrJets:" << std::endl;
    //printJets(std::cout, *corrJets);
    //std::cout << "uncorrJets:" << std::endl;
    //printJets(std::cout, *uncorrJets);
    std::cout << std::endl;
  }

  // add PFMET object to the event
  std::auto_ptr<reco::PFMETCollection> pfMEtCollection(new reco::PFMETCollection());
  pfMEtCollection->push_back(pfMEt);

  evt.put(pfMEtCollection);
}

std::vector<mvaMEtUtilities::JetInfo> PFMETProducerMVA::computeJetInfo(const reco::PFJetCollection& uncorrJets, 
								       const reco::PFJetCollection& corrJets, 
								       const reco::VertexCollection& vertices,
								       const reco::Vertex* hardScatterVertex,
								       double rho)
{
  std::vector<mvaMEtUtilities::JetInfo> retVal;
  for ( reco::PFJetCollection::const_iterator uncorrJet = uncorrJets.begin();
	uncorrJet != uncorrJets.end(); ++uncorrJet ) {
    for ( reco::PFJetCollection::const_iterator corrJet = corrJets.begin();
	  corrJet != corrJets.end(); ++corrJet ) {
      // match corrected and uncorrected jets
      if ( uncorrJet->jetArea() != corrJet->jetArea() ) continue;
      if ( !(fabs(uncorrJet->eta() - corrJet->eta()) < 0.01) ) continue;
      // check that jet passes loose PFJet id.
      
      //bool passesLooseJetId = (*looseJetIdAlgo_)(*corrJet);
      //if ( !passesLooseJetId ) continue; 
      if(!passPFLooseId(&(*uncorrJet))) continue;

      // compute jet energy correction factor
      // (= ratio of corrected/uncorrected jet Pt)
      double jetEnCorrFactor = corrJet->pt()/uncorrJet->pt();
      mvaMEtUtilities::JetInfo jetInfo;
      
      // PH: apply jet energy corrections for all Jets ignoring recommendations
      jetInfo.p4_ = corrJet->p4();

      // check that jet Pt used to compute MVA based jet id. is above threshold
      if ( !(jetInfo.p4_.pt() > minCorrJetPt_) ) continue;
      
      jetInfo.mva_ = mvaJetIdAlgo_.computeIdVariables(&(*corrJet), jetEnCorrFactor, hardScatterVertex, vertices, true).mva();
      jetInfo.neutralEnFrac_ = (uncorrJet->neutralEmEnergy() + uncorrJet->neutralHadronEnergy())/uncorrJet->energy();

      retVal.push_back(jetInfo);
      break;
    }
  }
  return retVal;
}

std::vector<mvaMEtUtilities::pfCandInfo> PFMETProducerMVA::computePFCandidateInfo(const reco::PFCandidateCollection& pfCandidates,
										  const reco::Vertex* hardScatterVertex)
{
  std::vector<mvaMEtUtilities::pfCandInfo> retVal;
  for ( reco::PFCandidateCollection::const_iterator pfCandidate = pfCandidates.begin();
	pfCandidate != pfCandidates.end(); ++pfCandidate ) {
    double dZ = -999.; // PH: If no vertex is reconstructed in the event
                       //     or PFCandidate has no track, set dZ to -999
    if ( hardScatterVertex ) {
      if      ( pfCandidate->trackRef().isNonnull()    ) dZ = fabs(pfCandidate->trackRef()->dz(hardScatterVertex->position()));
      else if ( pfCandidate->gsfTrackRef().isNonnull() ) dZ = fabs(pfCandidate->gsfTrackRef()->dz(hardScatterVertex->position()));
    }
    mvaMEtUtilities::pfCandInfo pfCandidateInfo;
    pfCandidateInfo.p4_ = pfCandidate->p4();
    pfCandidateInfo.dZ_ = dZ;
    retVal.push_back(pfCandidateInfo);
  }
  return retVal;
}

std::vector<reco::Vertex::Point> PFMETProducerMVA::computeVertexInfo(const reco::VertexCollection& vertices)
{
  std::vector<reco::Vertex::Point> retVal;
  for ( reco::VertexCollection::const_iterator vertex = vertices.begin();
	vertex != vertices.end(); ++vertex ) {
    if(fabs(vertex->z())           > 24.) continue;
    if(vertex->ndof()              <  4.) continue;
    if(vertex->position().Rho()    >  2.) continue;
    retVal.push_back(vertex->position());
  }
  return retVal;
}
double PFMETProducerMVA::chargedFrac(const reco::Candidate *iCand) { 
  if(iCand->isMuon())     return 1.;
  if(iCand->isElectron()) return 1.;
  if(iCand->isPhoton())   return 0.;
  double lPtTot = 0; double lPtCharged = 0;
  const reco::PFTau *lPFTau = 0; 
  lPFTau = dynamic_cast<const reco::PFTau*>(iCand);//} 
  if(lPFTau != 0) { 
    for (UInt_t i0 = 0; i0 < lPFTau->signalPFCands().size(); i0++) { 
      lPtTot += (lPFTau->signalPFCands())[i0]->pt(); 
      if((lPFTau->signalPFCands())[i0]->charge() == 0) continue;
      lPtCharged += (lPFTau->signalPFCands())[i0]->pt(); 
    }
  } else { 
    const pat::Tau *lPatPFTau = 0; 
    lPatPFTau = dynamic_cast<const pat::Tau*>(iCand);//} 
    for (UInt_t i0 = 0; i0 < lPatPFTau->signalPFCands().size(); i0++) { 
      lPtTot += (lPatPFTau->signalPFCands())[i0]->pt(); 
      if((lPatPFTau->signalPFCands())[i0]->charge() == 0) continue;
      lPtCharged += (lPatPFTau->signalPFCands())[i0]->pt(); 
    }
  }
  if(lPtTot == 0) lPtTot = 1.;
  return lPtCharged/lPtTot;
}
bool PFMETProducerMVA::passPFLooseId(const PFJet *iJet) { 
  if(iJet->energy()== 0)                                  return false;
  if(iJet->neutralHadronEnergy()/iJet->energy() > 0.99)   return false;
  if(iJet->neutralEmEnergy()/iJet->energy()     > 0.99)   return false;
  if(iJet->nConstituents() <  2)                          return false;
  if(iJet->chargedHadronEnergy()/iJet->energy() <= 0 && fabs(iJet->eta()) < 2.4 ) return false;
  if(iJet->chargedEmEnergy()/iJet->energy() >  0.99  && fabs(iJet->eta()) < 2.4 ) return false;
  if(iJet->chargedMultiplicity()            < 1      && fabs(iJet->eta()) < 2.4 ) return false;
  return true;
}
