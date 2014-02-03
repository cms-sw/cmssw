
/*
 *  See header file for a description of this class.
 *
 *  \author S.Bologensi - INFN Torino
 */

#include "DQM/DTMonitorModule/interface/DTTriggerCheck.h"

#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/LTCDigi/interface/LTCDigi.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include <iterator>

using namespace edm;
using namespace std;

DTTriggerCheck::DTTriggerCheck(const ParameterSet& pset) :
  isLocalRun(pset.getUntrackedParameter<bool>("localrun", true)) {

  if (!isLocalRun) {
    ltcDigiCollectionTag = pset.getParameter<edm::InputTag>("ltcDigiCollectionTag");
  }

 theDbe = edm::Service<DQMStore>().operator->();

 debug = pset.getUntrackedParameter<bool>("debug",false);

  theDbe->setCurrentFolder("DT/DTTriggerTask");
  histo = theDbe->book1D("hNTriggerPerType",
			 "# of trigger per type",21, -1, 20);
}

DTTriggerCheck::~DTTriggerCheck(){
}

void DTTriggerCheck::beginJob(){
}

void DTTriggerCheck::endJob(){

  if(debug)
    cout<<"[DTTriggerCheck] endjob called!"<<endl;
  
  theDbe->rmdir("DT/DTTriggerTask");
}

void DTTriggerCheck::analyze(const Event& event, const EventSetup& setup) {
  if(debug)
    cout << "[DTTriggerCheck] Analyze #Run: " << event.id().run()
	 << " #Event: " << event.id().event() << endl;
  
  //Get the trigger source from ltc digis
  edm::Handle<LTCDigiCollection> ltcdigis;
  if (!isLocalRun)
    {
      event.getByLabel(ltcDigiCollectionTag, ltcdigis);
      for (std::vector<LTCDigi>::const_iterator ltc_it = ltcdigis->begin(); ltc_it != ltcdigis->end(); ltc_it++){
	if (((*ltc_it).HasTriggered(0)) || 
	    ((*ltc_it).HasTriggered(1)) || 
	    ((*ltc_it).HasTriggered(2)) || 
	    ((*ltc_it).HasTriggered(3)) || 
	    ((*ltc_it).HasTriggered(4)))
	  histo->Fill(-1);
	if ((*ltc_it).HasTriggered(0))
	  histo->Fill(0);
	if ((*ltc_it).HasTriggered(1))
	  histo->Fill(1);
	if ((*ltc_it).HasTriggered(2))
	  histo->Fill(2);
	if ((*ltc_it).HasTriggered(3))
	  histo->Fill(3);
	if ((*ltc_it).HasTriggered(4))
	  histo->Fill(4);
      }
    }
  else
    histo->Fill(0);   
}  
