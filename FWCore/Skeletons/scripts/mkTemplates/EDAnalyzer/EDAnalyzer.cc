// -*- C++ -*-
//
// Package:    __subsys__/__pkgname__
// Class:      __class__
// 
/**\class __class__ __class__.cc __subsys__/__pkgname__/plugins/__class__.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  __author__
//         Created:  __date__
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
@example_track #include "FWCore/Utilities/interface/InputTag.h"
@example_track #include "DataFormats/TrackReco/interface/Track.h"
@example_track #include "DataFormats/TrackReco/interface/TrackFwd.h"
@example_histo #include "FWCore/ServiceRegistry/interface/Service.h"
@example_histo #include "CommonTools/UtilAlgos/interface/TFileService.h"
@example_histo #include "TH1.h"
//
// class declaration
//

class __class__ : public edm::EDAnalyzer {
   public:
      explicit __class__(const edm::ParameterSet&);
      ~__class__();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


   private:
      virtual void beginJob() override;
      virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() override;

      //virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
      //virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
      //virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
      //virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

      // ----------member data ---------------------------
@example_track       edm::InputTag trackTags_; //used to select what tracks to read from configuration file
@example_histo       TH1D * histo; 
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
__class__::__class__(const edm::ParameterSet& iConfig)
@example_track :
@example_track  trackTags_(iConfig.getUntrackedParameter<edm::InputTag>("tracks"))

{
   //now do what ever initialization is needed
@example_histo   edm::Service<TFileService> fs;
@example_histo   histo = fs->make<TH1D>("charge" , "Charges" , 200 , -2 , 2 );

}


__class__::~__class__()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
void
__class__::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

@example_track   using reco::TrackCollection;

@example_track    Handle<TrackCollection> tracks;
@example_track    iEvent.getByLabel(trackTags_,tracks);
@example_track    for(TrackCollection::const_iterator itTrack = tracks->begin();
@example_track        itTrack != tracks->end();                      
@example_track        ++itTrack) {
@example_track       int charge = 0;
@example_track       charge = itTrack->charge();  
@example_histo       histo->Fill( charge );
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


// ------------ method called once each job just before starting event loop  ------------
void 
__class__::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
__class__::endJob() 
{
}

// ------------ method called when starting to processes a run  ------------
/*
void 
__class__::beginRun(edm::Run const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a run  ------------
/*
void 
__class__::endRun(edm::Run const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when starting to processes a luminosity block  ------------
/*
void 
__class__::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a luminosity block  ------------
/*
void 
__class__::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
__class__::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
@example_track
@example_track  //Specify that only 'tracks' is allowed
@example_track  //To use, remove the default given above and uncomment below
@example_track  //ParameterSetDescription desc;
@example_track  //desc.addUntracked<edm::InputTag>("tracks","ctfWithMaterialTracks");
@example_track  //descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(__class__);
