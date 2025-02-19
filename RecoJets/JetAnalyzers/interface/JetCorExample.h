// Template class: JetCorExample
// Description:  Example of simple EDAnalyzer correcting jets "on the fly".
// Author: K. Kousouris
// Date:  25 - August - 2008
#ifndef JetCorExample_h
#define JetCorExample_h
#include <TH1.h>
#include <TFile.h>
#include "TNamed.h"
#include <vector>
#include <map>
#include "FWCore/Framework/interface/EDAnalyzer.h"

template<class Jet>
class JetCorExample : public edm::EDAnalyzer 
   {
     public:
       JetCorExample(edm::ParameterSet const& cfg);
     private:
       typedef std::vector<Jet> JetCollection;
       void FillHist1D(const TString& histName, const Double_t& x);
       void beginJob();
       void analyze(edm::Event const& e, edm::EventSetup const& iSetup);
       void endJob();
       std::map<TString, TH1*> m_HistNames1D;  
       TFile* m_file;
       /////// Configurable parameters /////////////////////////////////////
       /////// Jet algorithm: it can be any Calo or PF algorithm ///////////
       std::string JetAlgorithm;
       /////// Histogram where the plots are stored //////////////////////// 
       std::string HistoFileName;
       /////// Jet correction service: service providing jet corrections ///
       std::string JetCorrectionService;
   };
#endif
