// -*- C++ -*-
//
// Package:    DAFTest/DAFValidator_v1
// Class:      DAFValidator_v1
// 
/**\class DAFValidator_v1 DAFValidator_v1.cc DAFTest/DAFValidator_v1/plugins/DAFValidator_v1.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Erica Brondolin
//         Created:  Wed, 07 May 2014 10:31:58 GMT
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
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TH1.h"
//
// class declaration
//

class DAFValidator_v1 : public edm::EDAnalyzer {
   public:
      explicit DAFValidator_v1(const edm::ParameterSet&);
      ~DAFValidator_v1();

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
  edm::InputTag tracksTag_; //used to select what tracks to read from configuration file
  TH1F * h_track_q, *h_track_pt, *h_track_nlayers, *h_track_missoutlay;

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
DAFValidator_v1::DAFValidator_v1(const edm::ParameterSet& iConfig):
  tracksTag_(iConfig.getUntrackedParameter<edm::InputTag>("tracks"))
{
   //now do what ever initialization is needed
   edm::Service<TFileService> fs;
   h_track_q = fs->make<TH1F>("track_charge" , "track_charge" , 200 , -2 , 2 );
   h_track_pt = fs->make<TH1F>("track_pt", "track_pt", 100, 8., 12.);
   h_track_nlayers = fs->make<TH1F>("track_nlayers","track_nlayers",20,-0.5,19.5);
   h_track_missoutlay = fs->make<TH1F>("track_missoutlay","track_missoutlay",10,-0.5,9.5);

}


DAFValidator_v1::~DAFValidator_v1()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
void
DAFValidator_v1::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace reco;
   using namespace std;

   Handle<TrackCollection> tracks;
   iEvent.getByLabel(tracksTag_,tracks);
   for(TrackCollection::const_iterator itTrack = tracks->begin(); itTrack != tracks->end(); ++itTrack) {
     h_track_q->Fill( itTrack->charge() );
     h_track_pt->Fill(itTrack->pt());
     //hit pattern
     HitPattern track_hitpatt = itTrack->hitPattern();
     h_track_nlayers->Fill(track_hitpatt.trackerLayersWithMeasurement());
     HitPattern track_missoutlay = itTrack->trackerExpectedHitsOuter();
     h_track_missoutlay->Fill(track_missoutlay.numberOfHits());
   }

}


// ------------ method called once each job just before starting event loop  ------------
void 
DAFValidator_v1::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
DAFValidator_v1::endJob() 
{
}

// ------------ method called when starting to processes a run  ------------
/*
void 
DAFValidator_v1::beginRun(edm::Run const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a run  ------------
/*
void 
DAFValidator_v1::endRun(edm::Run const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when starting to processes a luminosity block  ------------
/*
void 
DAFValidator_v1::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a luminosity block  ------------
/*
void 
DAFValidator_v1::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
DAFValidator_v1::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);

  //Specify that only 'tracks' is allowed
  //To use, remove the default given above and uncomment below
  //ParameterSetDescription desc;
  //desc.addUntracked<edm::InputTag>("tracks","ctfWithMaterialTracks");
  //descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(DAFValidator_v1);
