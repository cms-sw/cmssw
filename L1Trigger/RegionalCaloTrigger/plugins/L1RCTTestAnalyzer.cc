// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"

#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTTestAnalyzer.h"

using std::string;
using std::cout;
using std::endl;

//
// constructors and destructor
//
L1RCTTestAnalyzer::L1RCTTestAnalyzer(const edm::ParameterSet& iConfig) :
  showEmCands(iConfig.getUntrackedParameter<bool>("showEmCands")),
  showRegionSums(iConfig.getUntrackedParameter<bool>("showRegionSums"))
{
   //now do what ever initialization is needed

  // get names of modules, producing object collections
}


L1RCTTestAnalyzer::~L1RCTTestAnalyzer()
{

   // do anything here that needs to be done at destruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
L1RCTTestAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
#ifdef THIS_IS_AN_EVENT_EXAMPLE
   Handle<ExampleData> pIn;
   iEvent.getByLabel("example",pIn);
#endif

#ifdef THIS_IS_AN_EVENTSETUP_EXAMPLE
   ESHandle<SetupData> pSetup;
   iSetup.get<SetupRecord>().get(pSetup);
#endif

   // as in L1GctTestAnalyzer.cc
   Handle<L1CaloEmCollection> rctEmCands;
   Handle<L1CaloRegionCollection> rctRegions;

   L1CaloEmCollection::const_iterator em;
   L1CaloRegionCollection::const_iterator rgn;

   iEvent.getByType(rctEmCands);
   iEvent.getByType(rctRegions);

   if(showEmCands)
     {
       cout << endl << "L1 RCT EmCand objects" << endl;
       for (em=rctEmCands->begin(); em!=rctEmCands->end(); em++){
	 //  cout << "(Analyzer)\n" << (*em) << endl;
	 unsigned short n_emcands = 0;
	 //cout << endl << "rank: " << (*em).rank() ;
	 if ((*em).rank() > 0){
	   cout << endl << "rank: " << (*em).rank();
	   unsigned short rgnPhi = 999;
	   unsigned short rgn = (unsigned short) (*em).rctRegion();
	   unsigned short card = (unsigned short) (*em).rctCard();
	   unsigned short crate = (unsigned short) (*em).rctCrate();
	   
	   if (card == 6){
	     rgnPhi = rgn;
	   }
	   else if (card < 6){
	     rgnPhi = (card % 2);
	   }
	   else {
	     cout << "rgnPhi not assigned (still " << rgnPhi << ") -- Weird card number! " << card ;
	   }
	   unsigned short phi_bin = ((crate % 9) * 2) + rgnPhi;
	   short eta_bin = (card/2) * 2 + 1;
	   if (card < 6){
	     eta_bin = eta_bin + rgn;
	   }
	   if (crate < 9){
	     eta_bin = -eta_bin;
	   }
	   n_emcands++;
	   cout << /* "rank: " << (*em).rank() << */ "  eta_bin: " << eta_bin << "  phi_bin: " << phi_bin << ".  crate: " << crate << "  card: " << card << "  region: " << rgn << ".  isolated: " << (*em).isolated();
	 }
       }
       cout << endl;
     }

   if(showRegionSums)
     {
       cout << "Regions" << endl;
       for (rgn=rctRegions->begin(); rgn!=rctRegions->end(); rgn++){
	 cout << /* "(Analyzer)\n" << */ (*rgn) << endl;
       }
       cout << endl;
     }

}
