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
// $Id: EfficiencyHistograms.cc,v 1.2 2007/07/08 08:14:05 elmer Exp $
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

  m_L1EtEff  = dir.make<TH1F>("L1EtEff", "E_{T}",m_etNBins,m_etMin,m_etMax);    m_L1EtEff->Sumw2();
  m_L1EtaEff = dir.make<TH1F>("L1EtaEff","#eta", m_etaNBins,m_etaMin,m_etaMax); m_L1EtaEff->Sumw2();
  m_L1PhiEff = dir.make<TH1F>("L1PhiEff","#phi", m_phiNBins,m_phiMin,m_phiMax); m_L1PhiEff->Sumw2();

  m_RefEtEff  = dir.make<TH1F>("RefEtEff", "E_{T}",m_etNBins,m_etMin,m_etMax);    m_RefEtEff->Sumw2();
  m_RefEtaEff = dir.make<TH1F>("RefEtaEff","#eta", m_etaNBins,m_etaMin,m_etaMax); m_RefEtaEff->Sumw2();
  m_RefPhiEff = dir.make<TH1F>("RefPhiEff","#phi", m_phiNBins,m_phiMin,m_phiMax); m_RefPhiEff->Sumw2();

}

EfficiencyHistograms::~EfficiencyHistograms(){}

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
