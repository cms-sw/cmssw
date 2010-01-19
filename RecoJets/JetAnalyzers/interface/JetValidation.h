// Class: JetValidation.h
// Description:  Some Basic validation plots for jets.
// Author: K. Kousouris
// Date:  27 - August - 2008
//
#ifndef JetValidation_h
#define JetValidation_h
#include <TH1.h>
#include <TH2.h>
#include <TProfile.h>
#include <TFile.h>
#include "TNamed.h"
#include <vector>
#include <map>
#include "FWCore/Framework/interface/EDAnalyzer.h"

class JetValidation : public edm::EDAnalyzer 
   {
     public:
       JetValidation(edm::ParameterSet const& cfg);
     private:
       void beginJob();
       void analyze(edm::Event const& e, edm::EventSetup const& iSetup);
       void endJob();
       void FillHist1D(const TString& histName, const Double_t& x);
       void FillHist2D(const TString& histName, const Double_t& x, const Double_t& y);
       void FillHistProfile(const TString& histName, const Double_t& x, const Double_t& y);

       std::map<TString, TH1*> m_HistNames1D;  
       std::map<TString, TH2*> m_HistNames2D;
       std::map<TString, TProfile*> m_HistNamesProfile; 
       TFile* m_file;
   
       double PtMin;
       double dRmatch;
       int Njets;
       bool MCarlo;
       std::string histoFileName; 
       std::string genAlgo;   
       std::string calAlgo;
       std::string jetTracksAssociator; 
  };

#endif
