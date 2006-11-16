#include "L1Trigger/GlobalCaloTrigger/src/L1FakeRctProducer.h"

#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloRegion.h"


#include <memory>

L1FakeRctProducer::L1FakeRctProducer(const edm::ParameterSet& iConfig)
{
  produces<L1CaloEmCollection>("fake");
  produces<L1CaloRegionCollection>("fake");

  rgnMode_ = iConfig.getParameter<int>("regionMode");
  iemMode_ = iConfig.getParameter<int>("isoEmMode");
  niemMode_ = iConfig.getParameter<int>("nonIsoEmMode");

}


L1FakeRctProducer::~L1FakeRctProducer()
{
 
}

void
L1FakeRctProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  
  std::auto_ptr<L1CaloEmCollection> emCands (new L1CaloEmCollection);
  std::auto_ptr<L1CaloRegionCollection> regions (new L1CaloRegionCollection);

  // decide which region to fill for mode 1
  int nCrt = int(18*rand()/(RAND_MAX+1.));
  int nCrd = int(7*rand()/(RAND_MAX+1.));
  int nRgn = int(2*rand()/(RAND_MAX+1.));

  // make regions
  for (int iCrt=0; iCrt<18; iCrt++) {
    for (int iCrd=0; iCrd<7; iCrd++) {
      for (int iRgn=0; iRgn<2; iRgn++) {
	unsigned et=0;

	if (rgnMode_==0) {
	  // throw random Et
	  et = int(100*rand()/(RAND_MAX+1.));
	}
	else if (rgnMode_==1 &&
	       iCrt==nCrt && 
	       iCrd==nCrd && 
	       iRgn==nRgn ) {
	  et = 1;
	} 
	
	regions->push_back(L1CaloRegion(et, false, false, false, false, iCrt, iCrd, iRgn));

      }
    }
  }

  // decide where to put a single iso em (mode 2)
  nCrt = int(18*rand()/(RAND_MAX+1.));
  nCrd = int(7*rand()/(RAND_MAX+1.));
  nRgn = int(2*rand()/(RAND_MAX+1.));

  // make iso em
  for (int iCrt=0; iCrt<18; iCrt++) {
    for (int iCrd=0; iCrd<7; iCrd++) {
      for (int iRgn=0; iRgn<2; iRgn++) {
	unsigned et=0;

	if (iemMode_==1) {
	  // throw random Et
	  et = int(100*rand()/(RAND_MAX+1.));
	}
	else if (iemMode_==2 &&
	       iCrt==nCrt && 
	       iCrd==nCrd && 
	       iRgn==nRgn ) {
	  et = 1;
	}
	  
	// make iso em
	//	regions.push_back(L1CaloRegion(et, false, false, false, false, iCrt, iCrd, iRgn));

      }
    }
  }

  // decide where to put a single noniso em (mode 2)
  nCrt = int(18*rand()/(RAND_MAX+1.));
  nCrd = int(7*rand()/(RAND_MAX+1.));
  nRgn = int(2*rand()/(RAND_MAX+1.));

  // make noniso em
  for (int iCrt=0; iCrt<18; iCrt++) {
    for (int iCrd=0; iCrd<7; iCrd++) {
      for (int iRgn=0; iRgn<2; iRgn++) {
	unsigned et=0;

	if (niemMode_==1) {
	  // throw random Et
	  et = int(100*rand()/(RAND_MAX+1.));
	}
	else if (niemMode_==2 &&
	       iCrt==nCrt && 
	       iCrd==nCrd && 
	       iRgn==nRgn ) {
	  et = 1;
	} 
	  
	// make noniso em
	//	regions.push_back(L1CaloRegion(et, false, false, false, false, iCrt, iCrd, iRgn));

      }
    }
  }

  iEvent.put(emCands);
  iEvent.put(regions);
  

}


// ------------ method called once each job just before starting event loop  ------------
void 
L1FakeRctProducer::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
L1FakeRctProducer::endJob() {
}

//define this as a plug-in
