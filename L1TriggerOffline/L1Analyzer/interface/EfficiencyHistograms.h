#ifndef L1Analyzer_EfficiencyHistograms_h
#define L1Analyzer_EfficiencyHistograms_h
// -*- C++ -*-
//
// Package:     L1Analyzer
// Class  :     EfficiencyHistograms
// 
/**\class EfficiencyHistograms EfficiencyHistograms.h L1TriggerOffline/L1Analyzer/interface/EfficiencyHistograms.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Alex Tapper
//         Created:  Tue Dec  5 14:02:48 CET 2006
// $Id: EfficiencyHistograms.h,v 1.1 2007/07/06 19:52:57 tapper Exp $
//

#include "FWCore/ParameterSet/interface/ParameterSet.h" // Paramters
#include "FWCore/ServiceRegistry/interface/Service.h" // Framework services
#include "PhysicsTools/UtilAlgos/interface/TFileService.h" // Framework service for histograms
#include "DataFormats/Candidate/interface/Candidate.h" // Candidate definition

#include "TH1.h" // RooT histogram class

class EfficiencyHistograms
{

   public:
      EfficiencyHistograms(const std::string name, const edm::ParameterSet & cfg);
      virtual ~EfficiencyHistograms();
      void FillL1(const reco::CandidateRef &l1);
      void FillReference(const reco::CandidateRef &ref);

   private:
      EfficiencyHistograms();

      std::string m_dirName; // Name for folder

      int m_etNBins, m_etaNBins,m_phiNBins;  // Bins
      double m_etMin, m_etaMin, m_phiMin; 
      double m_etMax, m_etaMax, m_phiMax; 

      TH1F *m_EtEff, *m_EtaEff, *m_PhiEff; // Histograms for efficiencies

      TH1F *m_L1EtEff,  *m_L1EtaEff,  *m_L1PhiEff; // Histograms for L1
      TH1F *m_RefEtEff, *m_RefEtaEff, *m_RefPhiEff; // Histograms for reference
 
};


#endif

