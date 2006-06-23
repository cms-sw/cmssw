// -*- C++ -*-
//
// Package:    anlzrname
// Class:      anlzrname
// 
/**\class anlzrname anlzrname.cc skelsubsys/anlzrname/src/anlzrname.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  John Doe
//         Created:  day-mon-xx
// RCS(Id)
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
@example_track #include "FWCore/ParameterSet/interface/InputTag.h"
@example_track #include "DataFormats/TrackReco/interface/Track.h"
//
// class decleration
//

class anlzrname : public edm::EDAnalyzer {
   public:
      explicit anlzrname(const edm::ParameterSet&);
      ~anlzrname();


      virtual void analyze(const edm::Event&, const edm::EventSetup&);
   private:
      // ----------member data ---------------------------
@example_track       edm::InputTag trackTags_; //used to select what tracks to read from configuration file
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
anlzrname::anlzrname(const edm::ParameterSet& iConfig):
@example_track  trackTags_(iConfig.getUntrackedParameter<edm::InputTag>("tracks"))
{
   //now do what ever initialization is needed

}


anlzrname::~anlzrname()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
anlzrname::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
@example_track    using reco::TrackCollection;
  
@example_track    Handle<TrackCollection> tracks;
@example_track    iEvent.getByLabel(trackTags_,tracks);
@example_track    for(TrackCollection::const_iterator itTrack = tracks->begin();
@example_track        itTrack != tracks->end();                      
@example_track        ++itTrack) {
@example_track       int charge = itTrack->charge();  
@example_track    }
#ifdef THIS_IS_AN_EVENT_EXAMPLE
   Handle<ExampleData> pIn;
   iEvent.getByLabel("example",pIn);
#endif
   
#ifdef THIS_IS_AN_EVENTSETUP_EXAMPLE
   ESHandle<SetupData> pSetup;
   iSetup.get<SetupRecord>().get(pSetup);
#endif
}

//define this as a plug-in
DEFINE_FWK_MODULE(anlzrname)
