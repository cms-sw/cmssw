#ifndef L1Analyzer_SimpleHistograms_h
#define L1Analyzer_SimpleHistograms_h
// -*- C++ -*-
//
// Package:     L1Analyzer
// Class  :     SimpleHistograms
// 
/**\class SimpleHistograms SimpleHistograms.h L1Trigger/L1Analyzer/interface/SimpleHistograms.h

 Description: Class for simple histograms of ET, eta and phi distributions.

 Usage:
    <usage>

*/
//
// Original Author:  Alex Tapper
//         Created:  Tue Dec  5 10:07:41 CET 2006
// $Id: SimpleHistograms.h,v 1.1 2007/02/13 14:49:19 tapper Exp $
//

#include "FWCore/ParameterSet/interface/ParameterSet.h" // Paramters
#include "FWCore/ServiceRegistry/interface/Service.h" // Framework services
#include "PhysicsTools/UtilAlgos/interface/TFileService.h" // Framework service for histograms
#include "DataFormats/Candidate/interface/Candidate.h" // Candidate definition

#include "TH1.h" // RooT histogram class

class SimpleHistograms
{

   public:
      SimpleHistograms(const std::string name, const edm::ParameterSet & cfg);
      virtual ~SimpleHistograms();
      void Fill(const reco::CandidateRef cand);

   private:
      SimpleHistograms();

      std::string m_dirName; // Name for folder

      int m_etNBins, m_etaNBins,m_phiNBins;  // Bins
      double m_etMin, m_etaMin, m_phiMin; 
      double m_etMax, m_etaMax, m_phiMax; 
      
      TH1F *m_Et, *m_Eta, *m_Phi; // Histograms
     

};

#endif
