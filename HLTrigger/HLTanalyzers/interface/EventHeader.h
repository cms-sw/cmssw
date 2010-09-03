#ifndef EVTHDR_H
#define EVTHDR_H

#include "FWCore/Framework/interface/Event.h"
#include "TChain.h"
#include "HepMC/HeavyIon.h"
#include "HepMC/GenEvent.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "DataFormats/Luminosity/interface/LumiSummary.h" 

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
  void analyze(edm::Event const& iEvent, TTree* tree);

private:

  // Tree variables
	int fRun;
	int fEvent;
	int fLumiBlock;
        int fBx;
        int fOrbit;
	float fAvgInstLumi;

  // input variables
  bool _Debug;

  // trigger names
};

#endif
