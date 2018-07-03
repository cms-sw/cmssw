#include "L1Trigger/L1TNtuples/interface/L1AnalysisSimulation.h"

#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"

L1Analysis::L1AnalysisSimulation::L1AnalysisSimulation()   
{
}

L1Analysis::L1AnalysisSimulation::~L1AnalysisSimulation()
{
}

void L1Analysis::L1AnalysisSimulation::Set(const edm::Event& e)
{


  if (!(e.eventAuxiliary().isRealData())) {

// Grab the pileup information for this event
    edm::Handle<std::vector< PileupSummaryInfo > >  puInfo;
    e.getByLabel(edm::InputTag("addPileupInfo"), puInfo);
    
    if (puInfo.isValid()) {
      std::vector<PileupSummaryInfo>::const_iterator pvi;
      
      for(pvi = puInfo->begin(); pvi != puInfo->end(); ++pvi) {
	
	int bx = pvi->getBunchCrossing();
	
	if(bx == 0) { 
	  sim_.meanInt   = pvi->getTrueNumInteractions();
	  sim_.actualInt = pvi->getPU_NumInteractions();
	  continue;
	}
	
      }      
    }
   } else {
         sim_.meanInt   = -1.;
	 sim_.actualInt = -1;
   } 
}

