#ifndef TRACKSFILTER_H
#define TRACKSFILTER_H

// system include files
#include <memory>
#include <algorithm>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
//
// class declaration
//

class TracksFilter : public edm::global::EDFilter<> {
   public:
      explicit TracksFilter(const edm::ParameterSet&);
      ~TracksFilter()=default;

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
  virtual bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

      // ----------member data ---------------------------
  edm::EDGetTokenT<reco::TrackCollection> tracksToken_;
  StringCutObjectSelector<reco::Track> selector_;

  unsigned int nmin_;
};
#endif

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
TracksFilter::TracksFilter(const edm::ParameterSet& iConfig)
  : tracksToken_ ( consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("src")) ),
    selector_    ( iConfig.getParameter<std::string> ("cut" ) ),
    nmin_        ( iConfig.getParameter<unsigned int> ("nmin" ) )
{
   //now do what ever initialization is needed

}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
TracksFilter::filter(edm::StreamID iStream, edm::Event& iEvent, const edm::EventSetup& iSetup) const
{
  bool pass = false;

  edm::Handle<reco::TrackCollection> tracks;
  iEvent.getByToken(tracksToken_, tracks);
//  unsigned int count = 0;
//  for ( auto const &track : *tracks )
//    if ( selector_(track) ) count++;
  unsigned int count = std::count_if(tracks->begin(), tracks->end(), selector_);
  pass = ( count >= nmin_ );

   return pass;
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
TracksFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  // tracksFilter
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>( "src", edm::InputTag("tracks")) ;
  desc.add<std::string>  ( "cut", "" );
  desc.add<unsigned int> ( "nmin", 0 );
  descriptions.add("tracksFilter", desc);

}
//define this as a plug-in
DEFINE_FWK_MODULE(TracksFilter);
