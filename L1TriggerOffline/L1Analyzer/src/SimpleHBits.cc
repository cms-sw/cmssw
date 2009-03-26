// -*- C++ -*-
//
// Package:     L1Analyzer
// Class  :     SimpleHBits
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Alex Tapper 
//         Created:  Tue Dec  5 10:07:30 CET 2006
// $Id: SimpleHBits.cc,v 1.2 2007/07/08 08:14:05 elmer Exp $
//

#include "L1TriggerOffline/L1Analyzer/interface/SimpleHBits.h"

SimpleHBits::SimpleHBits(const std::string name, const edm::ParameterSet & cfg): 
  m_dirName(name),
  m_bitsNBins(cfg.getUntrackedParameter<int>("bitsNBins")),
  m_bitsMin(cfg.getUntrackedParameter<double>("bitsMin")),
  m_bitsMax(cfg.getUntrackedParameter<double>("bitsMax")),
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

  m_Bits = dir.make<TH1F>("nBits","Trigger bits",m_bitsNBins,m_bitsMin,m_bitsMax);

}

SimpleHBits::~SimpleHBits()
{
}

void SimpleHBits::FillTB(float wgt) 
{

  m_Bits->Fill(wgt);
  
}
