// -*- C++ -*-
//
// Package:     L1Analyzer
// Class  :     ResolutionHistograms
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Alex Tapper
//         Created:  Tue Dec  5 14:02:34 CET 2006
// $Id: ResolutionHistograms.cc,v 1.4 2008/07/21 20:37:26 llista Exp $
//

#include "L1TriggerOffline/L1Analyzer/interface/ResolutionHistograms.h"
#include "DataFormats/Math/interface/deltaR.h"

ResolutionHistograms::ResolutionHistograms(const std::string name, const edm::ParameterSet & cfg): 
  m_dirName(name),
  // Get all the damn bins for the 1D resolutions
  m_etNBins(cfg.getUntrackedParameter<int>("etResNBins")),
  m_etaNBins(cfg.getUntrackedParameter<int>("etaResNBins")),
  m_phiNBins(cfg.getUntrackedParameter<int>("phiResNBins")),
  m_delRNBins(cfg.getUntrackedParameter<int>("delRNBins")),
  m_etMin(cfg.getUntrackedParameter<double>("etResMin")),
  m_etaMin(cfg.getUntrackedParameter<double>("etaResMin")),
  m_phiMin(cfg.getUntrackedParameter<double>("phiResMin")),
  m_delRMin(cfg.getUntrackedParameter<double>("delRMin")),
  m_etMax(cfg.getUntrackedParameter<double>("etResMax")),
  m_etaMax(cfg.getUntrackedParameter<double>("etaResMax")),
  m_phiMax(cfg.getUntrackedParameter<double>("phiResMax")),
  m_delRMax(cfg.getUntrackedParameter<double>("delRMax")),
  // Get the bins for the 2D correlations
  m_etN2DBins(cfg.getUntrackedParameter<int>("etCorNBins")),
  m_etaN2DBins(cfg.getUntrackedParameter<int>("etaCorNBins")),
  m_phiN2DBins(cfg.getUntrackedParameter<int>("phiCorNBins")),
  m_et2DMin(cfg.getUntrackedParameter<double>("etCorMin")),
  m_eta2DMin(cfg.getUntrackedParameter<double>("etaCorMin")),
  m_phi2DMin(cfg.getUntrackedParameter<double>("phiCorMin")),
  m_et2DMax(cfg.getUntrackedParameter<double>("etCorMax")),
  m_eta2DMax(cfg.getUntrackedParameter<double>("etaCorMax")),
  m_phi2DMax(cfg.getUntrackedParameter<double>("phiCorMax")),
  // Get the bins for the profiles
  m_etProfNBins(cfg.getUntrackedParameter<int>("etProfNBins")),
  m_etaProfNBins(cfg.getUntrackedParameter<int>("etaProfNBins")),
  m_phiProfNBins(cfg.getUntrackedParameter<int>("phiProfNBins")),
  m_etProfMin(cfg.getUntrackedParameter<double>("etProfMin")),
  m_etaProfMin(cfg.getUntrackedParameter<double>("etaProfMin")),
  m_phiProfMin(cfg.getUntrackedParameter<double>("phiProfMin")),
  m_etProfMax(cfg.getUntrackedParameter<double>("etProfMax")),
  m_etaProfMax(cfg.getUntrackedParameter<double>("etaProfMax")),
  m_phiProfMax(cfg.getUntrackedParameter<double>("phiProfMax"))
{
  
  edm::Service<TFileService> fs;

  TFileDirectory dir = fs->mkdir(m_dirName);

  m_EtRes  = dir.make<TH1F>("EtRes", "E_{T} Resolution",m_etNBins,m_etMin,m_etMax); 
  m_EtaRes = dir.make<TH1F>("EtaRes","#eta Resolution", m_etaNBins,m_etaMin,m_etaMax); 
  m_PhiRes = dir.make<TH1F>("PhiRes","#phi Resolution", m_phiNBins,m_phiMin,m_phiMax); 
  m_DeltaR = dir.make<TH1F>("DeltaR","#Delta R",m_delRNBins,m_delRMin,m_delRMax);

  m_EtCor  = dir.make<TH2F>("EtCor", "E_{T} Correlation",m_etN2DBins,m_et2DMin,m_et2DMax,m_etN2DBins,m_et2DMin,m_et2DMax);
  m_EtaCor = dir.make<TH2F>("EtaCor","#eta Correlation", m_etaN2DBins,m_eta2DMin,m_eta2DMax,m_etaN2DBins,m_eta2DMin,m_eta2DMax);
  m_PhiCor = dir.make<TH2F>("PhiCor","#phi Correlation", m_phiN2DBins,m_phi2DMin,m_phi2DMax,m_phiN2DBins,m_phi2DMin,m_phi2DMax);

  m_EtProf  = dir.make<TProfile>("EtProf", "E_{T} Profile",m_etProfNBins,m_etProfMin,m_etProfMax,"S"); 
  m_EtaProf = dir.make<TProfile>("EtaProf","#eta Profile", m_etaProfNBins,m_etaProfMin,m_etaProfMax,"S"); 
  m_PhiProf = dir.make<TProfile>("PhiProf","#phi Profile", m_phiProfNBins,m_phiProfMin,m_phiProfMax,"S"); 

}

ResolutionHistograms::~ResolutionHistograms()
{
}

void ResolutionHistograms::Fill(const reco::CandidateRef &l1, const reco::CandidateRef &ref)
{

  float d_et =(l1->et()-ref->et())/ref->et();
//   float d_eta=(l1->eta()-ref->eta())/ref->eta();
//   float d_phi=(l1->phi()-ref->phi())/ref->phi();
  float d_eta=l1->eta()-ref->eta();
  float d_phi=l1->phi()-ref->phi();
  
  m_EtRes->Fill(d_et);
  m_EtaRes->Fill(d_eta);
  m_PhiRes->Fill(d_phi);
  m_DeltaR->Fill(ROOT::Math::VectorUtil::DeltaR(l1->p4(),ref->p4()));

  m_EtCor->Fill(ref->et(),l1->et());
  m_EtaCor->Fill(ref->eta(),l1->eta());
  m_PhiCor->Fill(ref->phi(),l1->phi());

  m_EtProf->Fill(ref->et(),d_et);
  m_EtaProf->Fill(ref->eta(),d_eta);
  m_PhiProf->Fill(ref->phi(),d_phi);  

}


