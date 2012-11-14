#ifndef EVTHDR_H
#define EVTHDR_H

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "DataFormats/Luminosity/interface/LumiSummary.h" 
#include "DataFormats/Luminosity/interface/LumiDetails.h" 
#include "FWCore/Framework/interface/ESHandle.h"
#include "RecoLuminosity/LumiProducer/interface/LumiCorrectionParam.h"

#include "TChain.h"


/** \class EventHeader
  *  
  * $Date: November 2006
  * $Revision: 
  * \author V. Rekovic - UMinn
  */
class EventHeader {
public:
  EventHeader(); 
	~EventHeader();

  void setup(TTree* tree);

  /** Analyze the Data */
  void analyze(edm::Event const& iEvent, const edm::ESHandle<LumiCorrectionParam> & lumiCorr, TTree* tree);

private:

  // Tree variables
	int fRun;
	int fEvent;
	int fLumiBlock;
	int fBx;
	int fOrbit;
	double fAvgInstDelLumi;

  // input variables
  bool _Debug;

  // trigger names
};

#endif
