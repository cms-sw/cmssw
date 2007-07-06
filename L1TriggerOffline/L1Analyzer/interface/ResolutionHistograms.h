#ifndef L1Analyzer_ResolutionHistograms_h
#define L1Analyzer_ResolutionHistograms_h
// -*- C++ -*-
//
// Package:     L1Analyzer
// Class  :     ResolutionHistograms
// 
/**\class ResolutionHistograms ResolutionHistograms.h L1Trigger/L1Analyzer/interface/ResolutionHistograms.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Alex Tapper
//         Created:  Tue Dec  5 14:02:36 CET 2006
// $Id: ResolutionHistograms.h,v 1.1 2007/02/13 14:49:19 tapper Exp $
//

#include "FWCore/ParameterSet/interface/ParameterSet.h" // Paramters
#include "FWCore/ServiceRegistry/interface/Service.h" // Framework services
#include "PhysicsTools/UtilAlgos/interface/TFileService.h" // Framework service for histograms
#include "DataFormats/Candidate/interface/Candidate.h" // Candidate definition

#include "DataFormats/Math/interface/LorentzVector.h" // Maths stuff for Delta R
#include <Math/VectorUtil.h>

#include "TH1.h" // RooT histogram class
#include "TH2.h" 
#include "TProfile.h" 

class ResolutionHistograms
{

   public:
      ResolutionHistograms(const std::string name, const edm::ParameterSet & cfg);
      virtual ~ResolutionHistograms();
      void Fill(const reco::CandidateRef &l1, const reco::CandidateRef &ref);

   private:
      ResolutionHistograms();

      std::string m_dirName; // Name for folder

      int m_etNBins, m_etaNBins,m_phiNBins, m_delRNBins;  // Bins for 1D resolutions
      double m_etMin, m_etaMin, m_phiMin, m_delRMin; 
      double m_etMax, m_etaMax, m_phiMax, m_delRMax; 

      int m_etN2DBins, m_etaN2DBins,m_phiN2DBins;  // Bins for 2D correlations
      double m_et2DMin, m_eta2DMin, m_phi2DMin; 
      double m_et2DMax, m_eta2DMax, m_phi2DMax; 

      int m_etProfNBins, m_etaProfNBins,m_phiProfNBins;  // Bins for profiles
      double m_etProfMin, m_etaProfMin, m_phiProfMin; 
      double m_etProfMax, m_etaProfMax, m_phiProfMax;      

      TH1F *m_DeltaR;
      TH1F *m_EtRes, *m_EtaRes, *m_PhiRes; // 1D resolution histograms
      TH2F *m_EtCor, *m_EtaCor, *m_PhiCor; // 2D correlation histograms
      TProfile *m_EtProf, *m_EtaProf, *m_PhiProf; // Profile plots of resolutions
 
};


#endif
