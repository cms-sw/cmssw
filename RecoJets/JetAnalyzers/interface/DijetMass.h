#ifndef RecoExamples_DijetMass_h
#define RecoExamples_DijetMass_h
#include <TH1.h>
#include <TProfile.h>
#include <TH2.h>

#include <vector>
#include <map>

class TFile;

/* \class DijetMass
 *
 * \author Robert Harris
 *
 * Kalanand Mishra (November 22, 2009): 
     Modified and cleaned up to work in 3.3.X
 *
 */
#include "FWCore/Framework/interface/EDAnalyzer.h"


template<class Jet>
class DijetMass : public edm::EDAnalyzer {
public:
  DijetMass( const edm::ParameterSet & );

private:
  typedef std::vector<Jet> JetCollection;
  //Framwework stuff
  void beginJob( );
  void analyze( const edm::Event& , const edm::EventSetup& );
  void endJob();

  // Parameters passed via the config file
  double PtHistMax;      // Maximum edge of Pt histograms
  double EtaMax;  
  std::string histogramFile;
  std::string AKJets; 
  std::string AKCorJets; 
  std::string ICJets; 
  std::string ICCorJets; 
  std::string SCJets; 
  std::string SCCorJets; 
  std::string KTJets; 
  std::string KTCorJets; 


  //Simple Hists
  TH1F ptAKunc, etaAKunc, phiAKunc, m2jAKunc;
  TH1F ptAKcor, etaAKcor, phiAKcor, m2jAKcor;
  TH1F ptICunc, etaICunc, phiICunc, m2jICunc;
  TH1F ptICcor, etaICcor, phiICcor, m2jICcor;
  TH1F ptKTunc, etaKTunc, phiKTunc, m2jKTunc;
  TH1F ptKTcor, etaKTcor, phiKTcor, m2jKTcor;
  TH1F ptSCunc, etaSCunc, phiSCunc, m2jSCunc;
  TH1F ptSCcor, etaSCcor, phiSCcor, m2jSCcor;

  //Histo File 
  TFile* m_file;

  //Internal parameters
  int evtCount;
  int numJets;

};

#endif
