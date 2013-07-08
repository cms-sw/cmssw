// Template class: JetPlotsExample
// Description:  Example of simple EDAnalyzer for jets.
// Author: K. Kousouris
// Date:  25 - August - 2008
#ifndef JetPlotsExample_h
#define JetPlotsExample_h
#include <TH1.h>
#include <TFile.h>
#include "TNamed.h"
#include <vector>
#include <map>
#include "FWCore/Framework/interface/EDAnalyzer.h"

template<class Jet>
class JetPlotsExample : public edm::EDAnalyzer 
   {
     public:
       JetPlotsExample(edm::ParameterSet const& cfg);
     private:
       typedef std::vector<Jet> JetCollection;
       void FillHist1D(const TString& histName, const Double_t& x);
       void beginJob();
       void analyze(edm::Event const& e, edm::EventSetup const& iSetup);
       void endJob();
       std::map<TString, TH1*> m_HistNames1D;  
       TFile* m_file;
       /////// Configurable parameters /////////////////////////////////////
       /////// Jet algorithm: it can be any Calo, Gen or PF algorithm //////
       std::string JetAlgorithm;
       /////// Histogram where the plots are stored //////////////////////// 
       std::string HistoFileName;
       /////// Number of jets used for the plots /////////////////////////// 
       int NJets;    
   };
#endif
