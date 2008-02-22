/* \class ZToMuMuHistogrammer
 *
 * Z->mu+m- simple histogrammer module
 *
 * \author Luca Lista, INFN Naples
 *
 * \id $Id: ZToMuMuHistogrammer.cc,v 1.1 2007/10/12 11:28:57 llista Exp $
 *
 */
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Candidate/interface/Particle.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/Candidate/interface/CandAssociation.h"
#include "PhysicsTools/UtilAlgos/interface/HistoAnalyzer.h"
#include "TH1.h"
using namespace edm;
using namespace std;
using namespace reco;

typedef edm::AssociationVector<reco::CandidateRefProd, std::vector<double> > IsolationCollection;
typedef HistoAnalyzer<reco::CandidateCollection> BaseAnalyzer;

class ZToMuMuHistogrammer : public BaseAnalyzer {
public:
  ZToMuMuHistogrammer(const edm::ParameterSet& pset);

private:
  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup);
  InputTag zCands_, muIso1_, muIso2_;
  TH1 * h_pt1, * h_pt2, * h_eta1, * h_eta2, * h_iso1, * h_iso2;
  
};

ZToMuMuHistogrammer::ZToMuMuHistogrammer(const edm::ParameterSet& cfg) : 
  BaseAnalyzer(cfg),
  zCands_(cfg.getParameter<InputTag>("src")),
  muIso1_(cfg.getParameter<InputTag>("muonIsolations1")),
  muIso2_(cfg.getParameter<InputTag>("muonIsolations2")) {
  Service<TFileService> fs;
  h_pt1  = fs->make<TH1D>("mu1Pt",   "muon 1 p_{t} (GeV/c)", 2000,  0., 200.);
  h_pt2  = fs->make<TH1D>("mu2Pt",   "muon 2 p_{t} (GeV/c)", 2000,  0., 200.);
  h_eta1 = fs->make<TH1D>("mu1Eta",  "muon 1 #eta", 600,  -3, 3);
  h_eta2 = fs->make<TH1D>("mu2Eta",  "muon 2 #eta", 600,  -3, 3);
  h_iso1 = fs->make<TH1D>("mu1Iso",  "muon 1 isolation (#Sigma p_{t})", 1000, 0, 100);
  h_iso2 = fs->make<TH1D>("mu2Iso",  "muon 2 isolation (#Sigma p_{t})", 1000, 0, 100);
}

void ZToMuMuHistogrammer::analyze(const edm::Event& ev, const edm::EventSetup& setup) {
  // perform configurable set of Z histograms
  BaseAnalyzer::analyze(ev, setup);

  // perform customized plots for Z->l+l-
  Handle<CandidateCollection> zCands;
  ev.getByLabel(zCands_, zCands);
  Handle<CandDoubleAssociations> muIso1;
  ev.getByLabel(muIso1_, muIso1);
  Handle<CandDoubleAssociations> muIso2;
  ev.getByLabel(muIso2_, muIso2);

  size_t n = zCands->size();
  
  for(size_t i = 0; i < n; i++) {
    const Candidate & zCand = (*zCands)[i];
    const Candidate * dau1 = zCand.daughter(0);
    const Candidate * dau2 = zCand.daughter(1);
    CandidateRef mu1 = dau1->masterClone().castTo<CandidateRef>();
    CandidateRef mu2 = dau2->masterClone().castTo<CandidateRef>();
    double iso1 = (*muIso1)[mu1];
    double iso2 = (*muIso2)[mu2];    
    if (dau1->pt() < dau2->pt()) { 
      std::swap(dau1, dau2); 
      std::swap(iso1, iso2);
    }
    h_pt1->Fill(dau1->pt());
    h_pt2->Fill(dau2->pt());
    h_eta1->Fill(dau1->eta());
    h_eta2->Fill(dau2->eta());
    h_iso1->Fill(iso1);
    h_iso2->Fill(iso2);
  }
}  
   
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(ZToMuMuHistogrammer);
  
