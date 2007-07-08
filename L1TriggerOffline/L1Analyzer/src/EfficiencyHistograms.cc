// -*- C++ -*-
//
// Package:     L1Analyzer
// Class  :     EfficiencyHistograms
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Alex Tapper
//         Created:  Tue Dec  5 14:02:46 CET 2006
// $Id: EfficiencyHistograms.cc,v 1.1 2007/07/06 19:52:57 tapper Exp $
//

#include "L1TriggerOffline/L1Analyzer/interface/EfficiencyHistograms.h"

EfficiencyHistograms::EfficiencyHistograms(const std::string name, const edm::ParameterSet & cfg): 
  m_dirName(name),
  m_etNBins(cfg.getUntrackedParameter<int>("etNBins")),
  m_etaNBins(cfg.getUntrackedParameter<int>("etaNBins")),
  m_phiNBins(cfg.getUntrackedParameter<int>("phiNBins")),
  m_etMin(cfg.getUntrackedParameter<double>("etMin")),
  m_etaMin(cfg.getUntrackedParameter<double>("etaMin")),
  m_phiMin(cfg.getUntrackedParameter<double>("phiMin")),
  m_etMax(cfg.getUntrackedParameter<double>("etMax")),
  m_etaMax(cfg.getUntrackedParameter<double>("etaMax")),
  m_phiMax(cfg.getUntrackedParameter<double>("phiMax"))
{
  
  edm::Service<TFileService> fs;

  TFileDirectory dir = fs->mkdir(m_dirName);

  m_EtEff  = dir.make<TH1F>("EtEff", "E_{T} Efficiency",m_etNBins,m_etMin,m_etMax); 
  m_EtaEff = dir.make<TH1F>("EtaEff","#eta Efficiency", m_etaNBins,m_etaMin,m_etaMax);
  m_PhiEff = dir.make<TH1F>("PhiEff","#phi Efficiency", m_phiNBins,m_phiMin,m_phiMax);

  m_L1EtEff  = new TH1F("L1EtEff", "E_{T}",m_etNBins,m_etMin,m_etMax);    m_L1EtEff->Sumw2();
  m_L1EtaEff = new TH1F("L1EtaEff","#eta", m_etaNBins,m_etaMin,m_etaMax); m_L1EtaEff->Sumw2();
  m_L1PhiEff = new TH1F("L1PhiEff","#phi", m_phiNBins,m_phiMin,m_phiMax); m_L1PhiEff->Sumw2();

  m_RefEtEff  = new TH1F("RefEtEff", "E_{T}",m_etNBins,m_etMin,m_etMax);    m_RefEtEff->Sumw2();
  m_RefEtaEff = new TH1F("RefEtaEff","#eta", m_etaNBins,m_etaMin,m_etaMax); m_RefEtaEff->Sumw2();
  m_RefPhiEff = new TH1F("RefPhiEff","#phi", m_phiNBins,m_phiMin,m_phiMax); m_RefPhiEff->Sumw2();

}

EfficiencyHistograms::~EfficiencyHistograms()
{
  // Divide histograms to get efficiencies (careful with the errors!)

  m_EtEff->Divide(m_L1EtEff,m_RefEtEff,1.,1.,"B");
  m_EtaEff->Divide(m_L1EtaEff,m_RefEtaEff,1.,1.,"B");
  m_PhiEff->Divide(m_L1PhiEff,m_RefPhiEff,1.,1.,"B");

  // Clean up
  delete m_L1EtEff;
  delete m_L1EtaEff;
  delete m_L1PhiEff;

  delete m_RefEtEff; 
  delete m_RefEtaEff;
  delete m_RefPhiEff; 

}

void EfficiencyHistograms::FillL1(const reco::CandidateRef &ref)
{
  m_L1EtEff->Fill(ref->et());
  m_L1EtaEff->Fill(ref->eta());
  m_L1PhiEff->Fill(ref->phi());
}

void EfficiencyHistograms::FillReference(const reco::CandidateRef &ref)
{
  m_RefEtEff->Fill(ref->et());
  m_RefEtaEff->Fill(ref->eta());
  m_RefPhiEff->Fill(ref->phi());
}
