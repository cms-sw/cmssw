#ifndef RecoExamples_JetToDigiDump_h
#define RecoExamples_JetToDigiDump_h
#include <TH1.h>
#include <TProfile.h>
#include <TH2.h>
/* \class JetToDigiDump
 *
 * \author Robert Harris
 *
 * \version 1
 *
 */
#include "FWCore/Framework/interface/EDAnalyzer.h"

class JetToDigiDump : public edm::EDAnalyzer {
public:
  JetToDigiDump( const edm::ParameterSet & );

private:
  //Framwework stuff
  void beginJob( );
  void analyze( const edm::Event& , const edm::EventSetup& );
  void endJob();

  // Parameters passed via the config file
  std::string DumpLevel;   //How deep into calorimeter reco to dump
  std::string CaloJetAlg;  //Jet Algorithm to dump
  int DebugLevel;          //0 = no debug prints
  bool ShowECal;           //if true, ECAL hits are ignored 
    
  //Internal parameters
  int Dump;
  int evtCount;

};

#endif
