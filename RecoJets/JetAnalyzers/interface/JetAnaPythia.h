// Template class: JetAnaPythia
// Description:  Example of simple analyzer for jets produced by Pythia
// Author: R. Harris
// Date:  28 - October - 2008
#ifndef JetAnaPythia_h
#define JetAnaPythia_h
#include <TH1.h>
#include <TFile.h>
#include "TNamed.h"
#include <vector>
#include <map>
#include "FWCore/Framework/interface/EDAnalyzer.h"

template<class Jet>
class JetAnaPythia : public edm::EDAnalyzer 
   {
     public:
       JetAnaPythia(edm::ParameterSet const& cfg);
     private:
       typedef std::vector<Jet> JetCollection;
       void FillHist1D(const TString& histName, const Double_t& x, const Double_t& wt);
       void beginJob(edm::EventSetup const& iSetup);
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
       /////   Debug printout //////////////////
       bool debug;
       /////  Number used to calculate weight: total events gen in pthat bin ///
       int eventsGen;
       ////   Analysis level string.  Can speed up job by looking at less  ///
       std::string anaLevel; 
   };
#endif
