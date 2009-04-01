#ifndef L1Analyzer_SimpleHBits_h
#define L1Analyzer_SimpleHBits_h
// -*- C++ -*-
//
// Package:     L1Analyzer
// Class  :     SimpleHBits
// 
/**\class SimpleHBits SimpleHBits.h L1TriggerOffline/L1Analyzer/interface/SimpleHBits.h

 Description: Class for simple histograms of ET, eta and phi distributions.

 Usage:
    <usage>

*/
//
// Original Author:  Alex Tapper
//         Created:  Tue Dec  5 10:07:41 CET 2006
// $Id: SimpleHBits.h,v 1.1 2009/03/26 16:33:08 tapper Exp $
//

#include <cmath>
#include <vector>

#include "FWCore/ParameterSet/interface/ParameterSet.h" // Paramters
#include "FWCore/ServiceRegistry/interface/Service.h" // Framework services
#include "PhysicsTools/UtilAlgos/interface/TFileService.h" // Framework service for histograms
#include "DataFormats/Candidate/interface/Candidate.h" // Candidate definition

#include "TH1.h" // RooT histogram class

class SimpleHBits
{

   public:
      SimpleHBits(const std::string name, const edm::ParameterSet & cfg);
      virtual ~SimpleHBits();
      void FillTB(float wgt);

   private:
      SimpleHBits();

      std::string m_dirName; // Name for folder

      int m_bitsNBins;
      double m_bitsMin, m_bitsMax;      

      int m_etNBins, m_etaNBins,m_phiNBins;  // Bins
      double m_etMin, m_etaMin, m_phiMin; 
      double m_etMax, m_etaMax, m_phiMax; 
      
      //   TH1F *m_Et, *m_Eta, *m_Phi; // Histograms
      // The trigger bits histogram
      TH1F *m_Bits;

};

#endif
