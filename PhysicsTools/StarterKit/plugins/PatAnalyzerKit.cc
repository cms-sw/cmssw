#include "PhysicsTools/StarterKit/interface/PatAnalyzerKit.h"

using namespace std;
using namespace pat;


//
// constructors and destructor
//
PatAnalyzerKit::PatAnalyzerKit(const edm::ParameterSet& iConfig)
  :
  helper_(iConfig),
  verboseLevel_(0) 
{
  helper_.bookHistos(this);
}


PatAnalyzerKit::~PatAnalyzerKit()
{
}


//
// member functions
//

// ------------ method called to for each event  ------------
// void PatAnalyzerKit::analyze(const edm::Event& evt, const edm::EventSetup& iSetup)
void PatAnalyzerKit::produce( edm::Event & evt, const edm::EventSetup & es )
{
  using namespace edm;
  using namespace std;

  if ( verboseLevel_ > 10 )
    std::cout << "PatAnalyzerKit:: in analyze()." << std::endl;

  // --------------------------------------------------
  //    Step 1: Retrieve objects from data stream
  // --------------------------------------------------
  helper_.getHandles( evt,
		      muonHandle_,
		      electronHandle_,
		      tauHandle_,
		      jetHandle_,
		      METHandle_,
		      photonHandle_,
		      trackHandle_,
		      genParticlesHandle_);



  // --------------------------------------------------
  //    Step 2: invoke PhysicsHistograms to deal with all this.
  //
  //    Note that each handle will dereference into a vector<>,
  //    however the fillCollection() method takes a reference,
  //    so the collections are not copied...
  // --------------------------------------------------
  if ( verboseLevel_ > 10 )
    std::cout << "PatAnalyzerKit::analyze: calling fillCollection()." << std::endl;
  helper_.fillHistograms( evt,
			  muonHandle_,
			  electronHandle_,
			  tauHandle_,
			  jetHandle_,
			  METHandle_,
			  photonHandle_,
			  trackHandle_,
			  genParticlesHandle_);
}






// ------------ method called once each job just before starting event loop  ------------
void
PatAnalyzerKit::beginJob(const edm::EventSetup&)
{
}



// ------------ method called once each job just after ending the event loop  ------------
void
PatAnalyzerKit::endJob() {
}



//define this as a plug-in
DEFINE_FWK_MODULE(PatAnalyzerKit);
