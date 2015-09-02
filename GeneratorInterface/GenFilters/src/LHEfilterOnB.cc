#include "GeneratorInterface/GenFilters/interface/LHEfilterOnB.h"


using namespace edm;
using namespace std;

LHEfilterOnB::LHEfilterOnB(const edm::ParameterSet & iConfig): 
  // LHE collection
  lhesrc_(iConfig.getParameter<InputTag>( "LHEEventProduct" ) ),
  totalEvents_(0), passedEvents_(0)
{

}

LHEfilterOnB::~LHEfilterOnB()
{
 
}

bool LHEfilterOnB::filter(edm::Event & iEvent, const edm::EventSetup & iSetup) {

  Handle<LHEEventProduct> lheevt;
  iEvent.getByLabel( lhesrc_, lheevt );

  totalEvents_++;

  const lhef::HEPEUP hepeup = lheevt->hepeup();
  int nb_lhe = 0;
	for(int i = 0; i < hepeup.NUP; i++) {
    // check outgoing b partons only
		if (hepeup.ISTUP[i] == 1 && TMath::Abs(hepeup.IDUP[i])==5){
      nb_lhe++;
      // if more than one is found, reject event
      if(nb_lhe>1){
        return false; // skip event
      }
    }
    cout << endl;
	}
  
  passedEvents_++;
  return true;
  
}

// ------------ method called once each job just after ending the event loop  ------------
void LHEfilterOnB::endJob() {
  edm::LogInfo("LHEfilterOnB") << "=== Results of LHEfilterOnB: passed "
                                        << passedEvents_ << "/" << totalEvents_ << " events" << std::endl;
}

//define this as a plug-in
DEFINE_FWK_MODULE(LHEfilterOnB);
