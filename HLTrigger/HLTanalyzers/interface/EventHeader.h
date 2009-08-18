#ifndef EVTHDR_H
#define EVTHDR_H

#include "FWCore/Framework/interface/Event.h"
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
  void analyze(edm::Event const& iEvent, TTree* tree);

private:

  // Tree variables
	int fRun;
	int fEvent;
	int fLumiBlock;

  // input variables
  bool _Debug;

  // trigger names
};

#endif
