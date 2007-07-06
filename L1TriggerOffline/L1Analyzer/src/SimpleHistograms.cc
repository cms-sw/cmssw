// -*- C++ -*-
//
// Package:     L1Analyzer
// Class  :     SimpleHistograms
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Alex Tapper 
//         Created:  Tue Dec  5 10:07:30 CET 2006
// $Id: SimpleHistograms.cc,v 1.1 2007/02/13 14:49:21 tapper Exp $
//

#include "L1Trigger/L1Analyzer/interface/SimpleHistograms.h"

SimpleHistograms::SimpleHistograms(const std::string name, const edm::ParameterSet & cfg): 
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

  m_Et  = dir.make<TH1F>("Et", "E_{T}",m_etNBins,m_etMin,m_etMax); 
  m_Eta = dir.make<TH1F>("Eta","#eta", m_etaNBins,m_etaMin,m_etaMax); 
  m_Phi = dir.make<TH1F>("Phi","#phi", m_phiNBins,m_phiMin,m_phiMax); 

}

SimpleHistograms::~SimpleHistograms()
{
}

void SimpleHistograms::Fill(const reco::CandidateRef cand)
{
  m_Et->Fill(cand->et());
  m_Eta->Fill(cand->eta());
  m_Phi->Fill(cand->phi());
}


