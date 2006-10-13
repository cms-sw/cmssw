//-------------------------------------------------
//
/**  \class DTTrigFineSync
 *
 *   Analyzer used to generate BTI fine sync
 *   parameters
 *
 *
 *   $Date: 2006/09/18 10:40:16 $
 *   $Revision: 1.1 $
 *
 *   \author C. Battilana
 */
//
//--------------------------------------------------

#ifndef L1Trigger_DTTrigger_DTTrigFineSync_h
#define L1Trigger_DTTrigger_DTTrigFineSync_h

// Framework headers
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// Muon and Trigger headers
#include "L1Trigger/DTTrigger/interface/DTTrig.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"

// Root headers
#include "TROOT.h"
#include "TFile.h"



using namespace std;
using namespace edm;



//! Structure used to store sync information
struct QualArr {
  
  int nHH[25];     // number of HH
  int nHL[25];     // number of HL
  int nBX[25][25]; // bx distribution for HH trigs

};

// Type redefinition
typedef map< DTChamberId,QualArr,less<DTChamberId> > DelayContainer;
typedef DelayContainer::iterator DelayIterator;



class DTTrigFineSync: public EDAnalyzer{

public:

  //! Constructor
  DTTrigFineSync (const ParameterSet& pset);

  //! Destructor
  ~DTTrigFineSync();

  //! Executed at the end of the job
  void endJob();

  //! Executed at the begin of the job
  void beginJob(const EventSetup & iEventSetup);

  //! Executed every on event
  void analyze(const Event & iEvent, const EventSetup& iEventSetup);

private :

   // Trigger istance
  DTTrig* MyTrig;

  // Delay calculations variable
  DelayContainer QualMap;

  // Correct BX identifier
  int CorrectBX;
  
  // Outputfile
  fstream *txtfile;
  
  // Cfg file
  fstream *cfgfile;
  
  // Root File
  TFile *rootfile;

  // time to TDC_time conversion
  static const double myTtoTDC;

  // BTI ofsett step in ns
  double FTStep;

};
 
#endif

