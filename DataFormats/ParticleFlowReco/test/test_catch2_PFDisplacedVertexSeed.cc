#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertexSeed.h"

#include "catch.hpp"

static constexpr auto s_tag = "[PFDisplacedVertexSeed]";
TEST_CASE("Check adding elements", s_tag) {
  reco::PFDisplacedVertexSeed seed;

  REQUIRE(seed.elements().empty());

  SECTION("updateSeedPoint") {

    //empty tracks are fine
    std::vector<reco::Track> tracks(5);
    
    seed.updateSeedPoint(GlobalPoint(0.01,0.01,0.01),
                         reco::TrackBaseRef(reco::TrackRef(&tracks,0)),
                         reco::TrackBaseRef(reco::TrackRef(&tracks,1)) );
    REQUIRE(seed.elements().size() == 2);
    REQUIRE(seed.nTracks() == 2);

    seed.updateSeedPoint(GlobalPoint(0.01,0.01,0.01),
                         reco::TrackBaseRef(reco::TrackRef(&tracks,0)),
                         reco::TrackBaseRef(reco::TrackRef(&tracks,1)) );
    REQUIRE(seed.elements().size() == 2);
    REQUIRE(seed.nTracks() == 2);

    seed.updateSeedPoint(GlobalPoint(0.01,0.01,0.01),
                         reco::TrackBaseRef(reco::TrackRef(&tracks,0)),
                         reco::TrackBaseRef(reco::TrackRef(&tracks,2)) );
    REQUIRE(seed.elements().size() == 3);
    REQUIRE(seed.nTracks() == 3);

    seed.updateSeedPoint(GlobalPoint(0.01,0.01,0.01),
                         reco::TrackBaseRef(reco::TrackRef(&tracks,3)),
                         reco::TrackBaseRef(reco::TrackRef(&tracks,4)) );
    REQUIRE(seed.elements().size() == 5);
    REQUIRE(seed.nTracks() == 5);

  }
  
  SECTION("addElement") {
    //empty tracks are fine
    std::vector<reco::Track> tracks(3);
    
    seed.updateSeedPoint(GlobalPoint(0.01,0.01,0.01),
                         reco::TrackBaseRef(reco::TrackRef(&tracks,0)),
                         reco::TrackBaseRef(reco::TrackRef(&tracks,1)) );
    REQUIRE(seed.elements().size() == 2);
    REQUIRE(seed.nTracks() == 2);

    seed.addElement(reco::TrackBaseRef(reco::TrackRef(&tracks,0)));
    REQUIRE(seed.elements().size() == 2);
    REQUIRE(seed.nTracks() == 2);

    seed.addElement(reco::TrackBaseRef(reco::TrackRef(&tracks,2)));
    REQUIRE(seed.elements().size() == 3);
    REQUIRE(seed.nTracks() == 3);

  }
  SECTION("mergeWith") {
    std::vector<reco::Track> tracks(5);
    
    seed.updateSeedPoint(GlobalPoint(0.01,0.01,0.01),
                         reco::TrackBaseRef(reco::TrackRef(&tracks,0)),
                         reco::TrackBaseRef(reco::TrackRef(&tracks,1)) );


    SECTION("completely overlapping seeds") {
      reco::PFDisplacedVertexSeed otherSeed;
      otherSeed.updateSeedPoint(GlobalPoint(0.01,0.01,0.01),
                                       reco::TrackBaseRef(reco::TrackRef(&tracks,0)),
                                       reco::TrackBaseRef(reco::TrackRef(&tracks,1)) );

      seed.mergeWith(otherSeed);
      REQUIRE(seed.elements().size() == 2);
    }

    SECTION("partially overlapping seeds") {
      reco::PFDisplacedVertexSeed otherSeed;
      otherSeed.updateSeedPoint(GlobalPoint(0.01,0.01,0.01),
                                       reco::TrackBaseRef(reco::TrackRef(&tracks,0)),
                                       reco::TrackBaseRef(reco::TrackRef(&tracks,2)) );

      seed.mergeWith(otherSeed);
      REQUIRE(seed.elements().size() == 3);
    }

    SECTION("non overlapping seeds") {
      REQUIRE(seed.elements().size()==2);
      reco::PFDisplacedVertexSeed otherSeed;
      otherSeed.updateSeedPoint(GlobalPoint(0.01,0.01,0.01),
                                       reco::TrackBaseRef(reco::TrackRef(&tracks,3)),
                                       reco::TrackBaseRef(reco::TrackRef(&tracks,2)) );
      REQUIRE(otherSeed.elements().size() == 2);

      seed.mergeWith(otherSeed);
      REQUIRE(seed.elements().size() == 4);
    }

  }
}

