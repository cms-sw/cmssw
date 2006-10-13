//-------------------------------------------------
//
/**  \class DTTrigTest
 *
 *   EDAnalyzer that generates a rootfile useful
 *   for L1-DTTrigger debugging and performance 
 *   studies
 *
 *
 *   $Date: 2006/09/18 10:40:16 $
 *   $Revision: 1.1 $
 *
 *   \author C. Battilana
 */
//
//--------------------------------------------------

#ifndef L1Trigger_DTTrigger_DTTrigTest_h
#define L1Trigger_DTTrigger_DTTrigTest_h

// Framework related headers
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/Event.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// Trigger related headers
#include "L1Trigger/DTTrigger/interface/DTTrig.h"

// Root related headers
#include "TTree.h"
#include "TFile.h"



using namespace edm;

class DTTrigTest: public EDAnalyzer{
public:

  //! Constructor
  DTTrigTest(const ParameterSet& pset);
  
  //! Destructor
  ~DTTrigTest();

  //! Executed at the end of the job
  void endJob();

  //! Executed at the beginning of the job
  void beginJob(const EventSetup & iEventSetup);
  
  //! Analyze function executed on all the events
  void analyze(const Event & iEvent, const EventSetup& iEventSetup);
  
private:

  // time to TDC_time conversion
  static const double myTtoTDC;

  // trigger istance
  DTTrig* MyTrig;

  // debug flag
  bool debug;

  // tree
  TTree* theTree;
  // TFile
  TFile *f;

  //GENERAL block
  int             runn;
  int             eventn;
  float           weight;

  //GEANT block
  int             ngen;
  float           pxgen[10];
  float           pygen[10];
  float           pzgen[10];
  float           ptgen[10];
  float           etagen[10];
  float           phigen[10];
  int             chagen[10];
  float           vxgen[10];
  float           vygen[10];
  float           vzgen[10];
  
  // BTI
  int nbti;
  int bwh[100];
  int bstat[100];
  int bsect[100];
  int bsl[100];
  int bnum[100];
  int bbx[100];
  int bcod[100];
  int bk[100];
  int bx[100];
  float bposx[100];
  float bposy[100];
  float bposz[100];
  float bdirx[100];
  float bdiry[100];
  float bdirz[100];
  
  // TRACO
  int ntraco;
  int twh[80];
  int tstat[80];
  int tsect[80];
  int tnum[80];
  int tbx[80];
  int tcod[80];
  int tk[80];
  int tx[80];
  float tposx[100];
  float tposy[100];
  float tposz[100];
  float tdirx[100];
  float tdiry[100];
  float tdirz[100];
  
  // TSPHI
  int ntsphi;
  int swh[40];
  int sstat[40]; 
  int ssect[40];
  int sbx[40];
  int scod[40];
  int sphi[40];
  int sphib[40];
  float sposx[100];
  float sposy[100];
  float sposz[100];
  float sdirx[100];
  float sdiry[100];
  float sdirz[100]; 

  // TSTHETA
  int ntstheta;
  int thwh[40];
  int thstat[40]; 
  int thsect[40];
  int thbx[40];
  int thcode[40][7];
  int thpos[40][7];
  int thqual[40][7];

};
 
#endif

