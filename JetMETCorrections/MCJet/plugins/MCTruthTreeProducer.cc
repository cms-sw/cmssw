#include <iostream>
#include <sstream>
#include <istream>
#include <fstream>
#include <iomanip>
#include <string>
#include <cmath>
#include <functional>

#include "MCTruthTreeProducer.h"
#include "JetUtilMC.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
using namespace edm;
using namespace reco;
using namespace std;
namespace cms
{
MCTruthTreeProducer::MCTruthTreeProducer(edm::ParameterSet const& cfg) 
{
  calojets_           = cfg.getParameter<std::string> ("calojets");
  genjets_            = cfg.getParameter<std::string> ("genjets");
  PFJet_              = cfg.getParameter<bool> ("PFJet");
  histogramFile_      = cfg.getParameter<std::string>("histogramFile");
  m_file_             = new TFile(histogramFile_.c_str(),"RECREATE");
  mcTruthTree_        = new TTree("mcTruthTree","mcTruthTree");
  
  mcTruthTree_->Branch("ptCalo",     &ptCalo_,     "ptCalo_/F");
  mcTruthTree_->Branch("ptGen",      &ptGen_,      "ptGen_/F");
  mcTruthTree_->Branch("etaCalo",    &etaCalo_,    "etaCalo_/F");
  mcTruthTree_->Branch("etaGen",     &etaGen_,     "etaGen_/F");
  mcTruthTree_->Branch("phiCalo",    &phiCalo_,    "phiCalo_/F");
  mcTruthTree_->Branch("phiGen",     &phiGen_,     "phiGen_/F");
  mcTruthTree_->Branch("dR",         &dR_,         "dR_/F");
  mcTruthTree_->Branch("rank",       &rank_,       "rank_/I");
}
//////////////////////////////////////////////////////////////////////////////////////////
void MCTruthTreeProducer::endJob() 
{
  if (m_file_ !=0) 
    {
      m_file_->cd();
      mcTruthTree_->Write();
      delete m_file_;
      m_file_ = 0;      
    }
}
//////////////////////////////////////////////////////////////////////////////////////////
void MCTruthTreeProducer::analyze(edm::Event const& event, edm::EventSetup const& iSetup) 
{ 
  edm::Handle<GenJetCollection> genjets;
  edm::Handle<CaloJetCollection> calojets;
  edm::Handle<PFJetCollection> pfjets;
  CaloJetCollection::const_iterator i_calojet,i_matched;
  PFJetCollection::const_iterator i_pfjet,i_pfmatched;
  GenJetCollection::const_iterator i_genjet;
  event.getByLabel (genjets_,genjets);
  float rr;  
  int njet(0);
  if (PFJet_)
    {
      event.getByLabel (calojets_,pfjets);
      if (pfjets->size()>0 && genjets->size()>0)
        {
          for (i_genjet = genjets->begin(); i_genjet != genjets->end(); i_genjet++)
            {    
              float rmin(99);
              for(i_pfjet = pfjets->begin();i_pfjet != pfjets->end(); i_pfjet++)
                {
	          rr=radius(i_genjet,i_pfjet);
	          if (rr<rmin)
                    {
                      rmin = rr;
                      i_pfmatched = i_pfjet;
                    }
	        }
              ptGen_ = i_genjet->pt();
              etaGen_ = i_genjet->eta();
              phiGen_ = i_genjet->phi();
              ptCalo_ = i_pfmatched->pt();
              etaCalo_ = i_pfmatched->eta();
              phiCalo_ = i_pfmatched->phi();
              dR_ = rmin;
              rank_ = njet; 	
              mcTruthTree_->Fill();
              njet++;
            }
        }  
    }
  else
    {
      event.getByLabel (calojets_,calojets);
      if (calojets->size()>0 && genjets->size()>0)
        {
          for (i_genjet = genjets->begin(); i_genjet != genjets->end(); i_genjet++)
            {    
              float rmin(99);
              for(i_calojet = calojets->begin();i_calojet != calojets->end(); i_calojet++)
                {
	          rr=radius(i_genjet,i_calojet);
	          if (rr<rmin)
                    {
                      rmin = rr;
                      i_matched = i_calojet;
                    }
	        }
              ptGen_ = i_genjet->pt();
              etaGen_ = i_genjet->eta();
              phiGen_ = i_genjet->phi();
              ptCalo_ = i_matched->pt();
              etaCalo_ = i_matched->eta();
              phiCalo_ = i_matched->phi();
              dR_ = rmin;
              rank_ = njet; 	
              mcTruthTree_->Fill();
              njet++;
            }
        }  
    }      
}
//////////////////////////////////////////////////////////////////////////////////////////
MCTruthTreeProducer::MCTruthTreeProducer() 
{
  m_file_=0;
}
}
