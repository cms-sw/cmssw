#include <iostream>
#include <sstream>
#include <istream>
#include <fstream>
#include <iomanip>
#include <string>
#include <cmath>
#include <functional>

#include "JetMETCorrections/MCJet/plugins/CaloMCTruthTreeProducer.h"
#include "JetMETCorrections/MCJet/plugins/JetUtilMC.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/GenJet.h"
using namespace edm;
using namespace reco;
using namespace std;
//namespace cms
//{

CaloMCTruthTreeProducer::CaloMCTruthTreeProducer(edm::ParameterSet const& cfg)
{
  jets_          = consumes<CaloJetCollection>(edm::InputTag(cfg.getParameter<std::string> ("jets")));
  genjets_       = consumes<GenJetCollection>(edm::InputTag(cfg.getParameter<std::string> ("genjets")));
  gen_       = consumes<GenEventInfoProduct>(edm::InputTag("generator"));
  histogramFile_ = cfg.getParameter<std::string> ("histogramFile");
}
//////////////////////////////////////////////////////////////////////////////////////////
void CaloMCTruthTreeProducer::beginJob()
{
  file_          = new TFile(histogramFile_.c_str(),"RECREATE");
  mcTruthTree_   = new TTree("mcTruthTree","mcTruthTree");

  mcTruthTree_->Branch("ptJet",      &ptJet_,      "ptJet_/F");
  mcTruthTree_->Branch("ptGen",      &ptGen_,      "ptGen_/F");
  mcTruthTree_->Branch("ptHat",      &ptHat_,      "ptHat_/F");
  mcTruthTree_->Branch("emfJet",     &emfJet_,     "emfJet_/F");
  mcTruthTree_->Branch("etaJet",     &etaJet_,     "etaJet_/F");
  mcTruthTree_->Branch("etaGen",     &etaGen_,     "etaGen_/F");
  mcTruthTree_->Branch("phiJet",     &phiJet_,     "phiJet_/F");
  mcTruthTree_->Branch("phiGen",     &phiGen_,     "phiGen_/F");
  mcTruthTree_->Branch("dR",         &dR_,         "dR_/F");
  mcTruthTree_->Branch("rank",       &rank_,       "rank_/I");
}
//////////////////////////////////////////////////////////////////////////////////////////
void CaloMCTruthTreeProducer::endJob()
{
  if (file_ !=0)
    {
      file_->cd();
      mcTruthTree_->Write();
    }
  file_ = 0;
}
//////////////////////////////////////////////////////////////////////////////////////////
void CaloMCTruthTreeProducer::analyze(edm::Event const& event, edm::EventSetup const& iSetup)
{
  edm::Handle<GenJetCollection> genjets;
  edm::Handle<CaloJetCollection> jets;
  edm::Handle<GenEventInfoProduct> hEventInfo;
  CaloJetCollection::const_iterator i_jet,i_matched;
  GenJetCollection::const_iterator i_genjet;
  event.getByToken (genjets_,genjets);
  event.getByToken (jets_,jets);
  event.getByToken(gen_,hEventInfo);
  ptHat_ = hEventInfo->binningValues()[0];
  float rr;
  int njet(0);
  if (jets->size()>0 && genjets->size()>0)
    {
      for (i_genjet = genjets->begin(); i_genjet != genjets->end(); i_genjet++)
       {
         float rmin(99);
         for(i_jet = jets->begin();i_jet != jets->end(); i_jet++)
           {
	     rr=radius(i_genjet,i_jet);
	     if (rr<rmin)
               {
                 rmin = rr;
                 i_matched = i_jet;
               }
	   }
         ptGen_   = i_genjet->pt();
         etaGen_  = i_genjet->eta();
         phiGen_  = i_genjet->phi();
         ptJet_   = i_matched->pt();
         etaJet_  = i_matched->eta();
         phiJet_  = i_matched->phi();
         emfJet_  = i_matched->emEnergyFraction();
         dR_      = rmin;
         rank_    = njet;
         mcTruthTree_->Fill();
         njet++;
       }
    }
}
//////////////////////////////////////////////////////////////////////////////////////////
CaloMCTruthTreeProducer::~CaloMCTruthTreeProducer()
{
  delete file_;
  delete mcTruthTree_;
}
//}
