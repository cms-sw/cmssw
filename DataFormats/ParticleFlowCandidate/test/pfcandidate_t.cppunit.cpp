#include <cppunit/extensions/HelperMacros.h>
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "DataFormats/Common/interface/EDProductGetter.h"

namespace {
  class DummyGetter : public edm::EDProductGetter {
  public:
    edm::EDProduct const* getIt(edm::ProductID const&) const {
      return 0;
    }
    
  };
}

class testPFCandidate: public CppUnit::TestFixture
{
CPPUNIT_TEST_SUITE(testPFCandidate);

CPPUNIT_TEST(test);

CPPUNIT_TEST_SUITE_END();
public:
  void setUp(){}
  void tearDown(){}

  void test();
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testPFCandidate);

void testPFCandidate::test()
{
  using namespace reco;

  const edm::ProductID dummyID(2,1);
  DummyGetter dummyGetter;
  {
    //Test trackRef alone
    PFCandidate proto(1,reco::Candidate::LorentzVector(),PFCandidate::e);
  
    CPPUNIT_ASSERT(proto.trackRef().isNull());
    CPPUNIT_ASSERT(proto.gsfTrackRef().isNull());
    CPPUNIT_ASSERT(proto.v0Ref().isNull());
    
    const unsigned int key = 10;
    reco::TrackRef ref(dummyID, 10,&dummyGetter);
    
    proto.setTrackRef(ref);
    
    CPPUNIT_ASSERT(!proto.trackRef().isNull());
    CPPUNIT_ASSERT(proto.gsfTrackRef().isNull());
    CPPUNIT_ASSERT(proto.v0Ref().isNull());
    
    reco::TrackRef returnedRef = proto.trackRef();
    CPPUNIT_ASSERT(returnedRef.refCore().id()==dummyID);
    CPPUNIT_ASSERT(returnedRef.key() == key);
    
    proto.setTrackRef(reco::TrackRef());
    CPPUNIT_ASSERT(proto.trackRef().isNull());
    CPPUNIT_ASSERT(proto.gsfTrackRef().isNull());
    CPPUNIT_ASSERT(proto.v0Ref().isNull());
  }

  {
    //Test gsfTrackRef alone
    PFCandidate proto(1,reco::Candidate::LorentzVector(),PFCandidate::e);
  
    CPPUNIT_ASSERT(proto.trackRef().isNull());
    CPPUNIT_ASSERT(proto.gsfTrackRef().isNull());
    CPPUNIT_ASSERT(proto.v0Ref().isNull());
    
    const unsigned int key = 10;
    reco::GsfTrackRef ref(dummyID, 10,&dummyGetter);
    
    proto.setGsfTrackRef(ref);
    
    CPPUNIT_ASSERT(proto.trackRef().isNull());
    CPPUNIT_ASSERT(!proto.gsfTrackRef().isNull());
    CPPUNIT_ASSERT(proto.v0Ref().isNull());
    
    reco::GsfTrackRef returnedRef = proto.gsfTrackRef();
    CPPUNIT_ASSERT(returnedRef.refCore().id()==dummyID);
    CPPUNIT_ASSERT(returnedRef.key() == key);
    
    proto.setGsfTrackRef(reco::GsfTrackRef());
    CPPUNIT_ASSERT(proto.trackRef().isNull());
    CPPUNIT_ASSERT(proto.gsfTrackRef().isNull());
    CPPUNIT_ASSERT(proto.v0Ref().isNull());
  }

  {
    //Test muonRef alone
    PFCandidate proto(1,reco::Candidate::LorentzVector(),PFCandidate::e);
  
    CPPUNIT_ASSERT(proto.trackRef().isNull());
    CPPUNIT_ASSERT(proto.gsfTrackRef().isNull());
    CPPUNIT_ASSERT(proto.v0Ref().isNull());
    
    const unsigned int key = 10;
    reco::VertexCompositeCandidateRef ref(dummyID, 10,&dummyGetter);
    
    proto.setV0Ref(ref);
    
    CPPUNIT_ASSERT(proto.trackRef().isNull());
    CPPUNIT_ASSERT(proto.gsfTrackRef().isNull());
    CPPUNIT_ASSERT(!proto.v0Ref().isNull());
    
    reco::VertexCompositeCandidateRef returnedRef = proto.v0Ref();
    CPPUNIT_ASSERT(returnedRef.refCore().id()==dummyID);
    CPPUNIT_ASSERT(returnedRef.key() == key);
    
    proto.setV0Ref(reco::VertexCompositeCandidateRef());
    CPPUNIT_ASSERT(proto.trackRef().isNull());
    CPPUNIT_ASSERT(proto.gsfTrackRef().isNull());
    CPPUNIT_ASSERT(proto.v0Ref().isNull());
  }

  {
    //Insert in order and the remove in the same order
    PFCandidate proto(1,reco::Candidate::LorentzVector(),PFCandidate::e);
  
    CPPUNIT_ASSERT(proto.trackRef().isNull());
    CPPUNIT_ASSERT(proto.gsfTrackRef().isNull());
    CPPUNIT_ASSERT(proto.v0Ref().isNull());
    
    const unsigned int key = 10;
    reco::TrackRef ref(dummyID, 10,&dummyGetter);
    
    proto.setTrackRef(ref);
    CPPUNIT_ASSERT(!proto.trackRef().isNull());
    CPPUNIT_ASSERT(proto.gsfTrackRef().isNull());
    CPPUNIT_ASSERT(proto.v0Ref().isNull());

    const edm::ProductID dummy2(3,4);
    const unsigned int key2(11);
    reco::GsfTrackRef ref2(dummy2,key2,&dummyGetter);

    const edm::ProductID dummy3(5,6);
    const unsigned int key3(12);
    reco::VertexCompositeCandidateRef ref3(dummy3,key3,&dummyGetter);
    
    proto.setGsfTrackRef(ref2);
    {
      CPPUNIT_ASSERT(!proto.trackRef().isNull());
      CPPUNIT_ASSERT(!proto.gsfTrackRef().isNull());
      CPPUNIT_ASSERT(proto.v0Ref().isNull());
    
      reco::TrackRef returnedRef = proto.trackRef();
      CPPUNIT_ASSERT(returnedRef.refCore().id()==dummyID);
      CPPUNIT_ASSERT(returnedRef.key() == key);

      reco::GsfTrackRef returnedRef2 = proto.gsfTrackRef();
      CPPUNIT_ASSERT(returnedRef2.refCore().id()==dummy2);
      CPPUNIT_ASSERT(returnedRef2.key() == key2);
    }

    proto.setV0Ref(ref3);
    {
      CPPUNIT_ASSERT(!proto.trackRef().isNull());
      CPPUNIT_ASSERT(!proto.gsfTrackRef().isNull());
      CPPUNIT_ASSERT(!proto.v0Ref().isNull());
    
      reco::TrackRef returnedRef = proto.trackRef();
      CPPUNIT_ASSERT(returnedRef.refCore().id()==dummyID);
      CPPUNIT_ASSERT(returnedRef.key() == key);

      reco::GsfTrackRef returnedRef2 = proto.gsfTrackRef();
      CPPUNIT_ASSERT(returnedRef2.refCore().id()==dummy2);
      CPPUNIT_ASSERT(returnedRef2.key() == key2);

      reco::VertexCompositeCandidateRef returnedRef3 = proto.v0Ref();
      CPPUNIT_ASSERT(returnedRef3.refCore().id()==dummy3);
      CPPUNIT_ASSERT(returnedRef3.key() == key3);
    }

    
    proto.setTrackRef(reco::TrackRef());
    {
      CPPUNIT_ASSERT(proto.trackRef().isNull());
      CPPUNIT_ASSERT(!proto.gsfTrackRef().isNull());
      CPPUNIT_ASSERT(!proto.v0Ref().isNull());
      
      reco::GsfTrackRef returnedRef2 = proto.gsfTrackRef();
      CPPUNIT_ASSERT(returnedRef2.refCore().id()==dummy2);
      CPPUNIT_ASSERT(returnedRef2.key() == key2);

      reco::VertexCompositeCandidateRef returnedRef3 = proto.v0Ref();
      CPPUNIT_ASSERT(returnedRef3.refCore().id()==dummy3);
      CPPUNIT_ASSERT(returnedRef3.key() == key3);
    }

    proto.setGsfTrackRef(reco::GsfTrackRef());
    {
      CPPUNIT_ASSERT(proto.trackRef().isNull());
      CPPUNIT_ASSERT(proto.gsfTrackRef().isNull());
      CPPUNIT_ASSERT(!proto.v0Ref().isNull());
      
      reco::VertexCompositeCandidateRef returnedRef3 = proto.v0Ref();
      CPPUNIT_ASSERT(returnedRef3.refCore().id()==dummy3);
      CPPUNIT_ASSERT(returnedRef3.key() == key3);
    } 
    
  }


  {
    //Insert in order and the remove in the opposite order
    PFCandidate proto(1,reco::Candidate::LorentzVector(),PFCandidate::e);
  
    CPPUNIT_ASSERT(proto.trackRef().isNull());
    CPPUNIT_ASSERT(proto.gsfTrackRef().isNull());
    CPPUNIT_ASSERT(proto.v0Ref().isNull());
    
    const unsigned int key = 10;
    reco::TrackRef ref(dummyID, 10,&dummyGetter);
    
    proto.setTrackRef(ref);
    CPPUNIT_ASSERT(!proto.trackRef().isNull());
    CPPUNIT_ASSERT(proto.gsfTrackRef().isNull());
    CPPUNIT_ASSERT(proto.v0Ref().isNull());

    const edm::ProductID dummy2(3,4);
    const unsigned int key2(11);
    reco::GsfTrackRef ref2(dummy2,key2,&dummyGetter);

    const edm::ProductID dummy3(5,6);
    const unsigned int key3(12);
    reco::VertexCompositeCandidateRef ref3(dummy3,key3,&dummyGetter);
    
    proto.setGsfTrackRef(ref2);
    {
      CPPUNIT_ASSERT(!proto.trackRef().isNull());
      CPPUNIT_ASSERT(!proto.gsfTrackRef().isNull());
      CPPUNIT_ASSERT(proto.v0Ref().isNull());
    
      reco::TrackRef returnedRef = proto.trackRef();
      CPPUNIT_ASSERT(returnedRef.refCore().id()==dummyID);
      CPPUNIT_ASSERT(returnedRef.key() == key);

      reco::GsfTrackRef returnedRef2 = proto.gsfTrackRef();
      CPPUNIT_ASSERT(returnedRef2.refCore().id()==dummy2);
      CPPUNIT_ASSERT(returnedRef2.key() == key2);
    }

    proto.setV0Ref(ref3);
    {
      CPPUNIT_ASSERT(!proto.trackRef().isNull());
      CPPUNIT_ASSERT(!proto.gsfTrackRef().isNull());
      CPPUNIT_ASSERT(!proto.v0Ref().isNull());
    
      reco::TrackRef returnedRef = proto.trackRef();
      CPPUNIT_ASSERT(returnedRef.refCore().id()==dummyID);
      CPPUNIT_ASSERT(returnedRef.key() == key);

      reco::GsfTrackRef returnedRef2 = proto.gsfTrackRef();
      CPPUNIT_ASSERT(returnedRef2.refCore().id()==dummy2);
      CPPUNIT_ASSERT(returnedRef2.key() == key2);

      reco::VertexCompositeCandidateRef returnedRef3 = proto.v0Ref();
      CPPUNIT_ASSERT(returnedRef3.refCore().id()==dummy3);
      CPPUNIT_ASSERT(returnedRef3.key() == key3);
    }

    
    proto.setV0Ref(reco::VertexCompositeCandidateRef());
    {
      CPPUNIT_ASSERT(!proto.trackRef().isNull());
      CPPUNIT_ASSERT(!proto.gsfTrackRef().isNull());
      CPPUNIT_ASSERT(proto.v0Ref().isNull());
      
      reco::TrackRef returnedRef = proto.trackRef();
      CPPUNIT_ASSERT(returnedRef.refCore().id()==dummyID);
      CPPUNIT_ASSERT(returnedRef.key() == key);

      reco::GsfTrackRef returnedRef2 = proto.gsfTrackRef();
      CPPUNIT_ASSERT(returnedRef2.refCore().id()==dummy2);
      CPPUNIT_ASSERT(returnedRef2.key() == key2);

    }

    proto.setGsfTrackRef(reco::GsfTrackRef());
    {
      CPPUNIT_ASSERT(!proto.trackRef().isNull());
      CPPUNIT_ASSERT(proto.gsfTrackRef().isNull());
      CPPUNIT_ASSERT(proto.v0Ref().isNull());
      
      reco::TrackRef returnedRef = proto.trackRef();
      CPPUNIT_ASSERT(returnedRef.refCore().id()==dummyID);
      CPPUNIT_ASSERT(returnedRef.key() == key);
    } 
    
  }
  
  {
    //Insert in reverse order
    PFCandidate proto(1,reco::Candidate::LorentzVector(),PFCandidate::e);
  
    CPPUNIT_ASSERT(proto.trackRef().isNull());
    CPPUNIT_ASSERT(proto.gsfTrackRef().isNull());
    CPPUNIT_ASSERT(proto.v0Ref().isNull());
    
    const unsigned int key = 10;
    reco::TrackRef ref(dummyID, 10,&dummyGetter);

    const edm::ProductID dummy2(3,4);
    const unsigned int key2(11);
    reco::GsfTrackRef ref2(dummy2,key2,&dummyGetter);

    const edm::ProductID dummy3(5,6);
    const unsigned int key3(12);
    reco::VertexCompositeCandidateRef ref3(dummy3,key3,&dummyGetter);
    
    
    proto.setV0Ref(ref3);
    CPPUNIT_ASSERT(proto.trackRef().isNull());
    CPPUNIT_ASSERT(proto.gsfTrackRef().isNull());
    CPPUNIT_ASSERT(!proto.v0Ref().isNull());

    proto.setGsfTrackRef(ref2);
    {
      CPPUNIT_ASSERT(proto.trackRef().isNull());
      CPPUNIT_ASSERT(!proto.gsfTrackRef().isNull());
      CPPUNIT_ASSERT(!proto.v0Ref().isNull());
    
      reco::GsfTrackRef returnedRef2 = proto.gsfTrackRef();
      CPPUNIT_ASSERT(returnedRef2.refCore().id()==dummy2);
      CPPUNIT_ASSERT(returnedRef2.key() == key2);

      reco::VertexCompositeCandidateRef returnedRef3 = proto.v0Ref();
      CPPUNIT_ASSERT(returnedRef3.refCore().id()==dummy3);
      CPPUNIT_ASSERT(returnedRef3.key() == key3);
    }

    proto.setTrackRef(ref);
    {
      CPPUNIT_ASSERT(!proto.trackRef().isNull());
      CPPUNIT_ASSERT(!proto.gsfTrackRef().isNull());
      CPPUNIT_ASSERT(!proto.v0Ref().isNull());
    
      reco::TrackRef returnedRef = proto.trackRef();
      CPPUNIT_ASSERT(returnedRef.refCore().id()==dummyID);
      CPPUNIT_ASSERT(returnedRef.key() == key);

      reco::GsfTrackRef returnedRef2 = proto.gsfTrackRef();
      CPPUNIT_ASSERT(returnedRef2.refCore().id()==dummy2);
      CPPUNIT_ASSERT(returnedRef2.key() == key2);

      reco::VertexCompositeCandidateRef returnedRef3 = proto.v0Ref();
      CPPUNIT_ASSERT(returnedRef3.refCore().id()==dummy3);
      CPPUNIT_ASSERT(returnedRef3.key() == key3);
    }
  }

  {
    //Insert 2,3,1
    PFCandidate proto(1,reco::Candidate::LorentzVector(),PFCandidate::e);
  
    CPPUNIT_ASSERT(proto.trackRef().isNull());
    CPPUNIT_ASSERT(proto.gsfTrackRef().isNull());
    CPPUNIT_ASSERT(proto.v0Ref().isNull());
    
    const unsigned int key = 10;
    reco::TrackRef ref(dummyID, 10,&dummyGetter);

    const edm::ProductID dummy2(3,4);
    const unsigned int key2(11);
    reco::GsfTrackRef ref2(dummy2,key2,&dummyGetter);

    const edm::ProductID dummy3(5,6);
    const unsigned int key3(12);
    reco::VertexCompositeCandidateRef ref3(dummy3,key3,&dummyGetter);
    
    
    proto.setGsfTrackRef(ref2);
    CPPUNIT_ASSERT(proto.trackRef().isNull());
    CPPUNIT_ASSERT(!proto.gsfTrackRef().isNull());
    CPPUNIT_ASSERT(proto.v0Ref().isNull());

    proto.setV0Ref(ref3);
    {
      CPPUNIT_ASSERT(proto.trackRef().isNull());
      CPPUNIT_ASSERT(!proto.gsfTrackRef().isNull());
      CPPUNIT_ASSERT(!proto.v0Ref().isNull());
    
      reco::GsfTrackRef returnedRef2 = proto.gsfTrackRef();
      CPPUNIT_ASSERT(returnedRef2.refCore().id()==dummy2);
      CPPUNIT_ASSERT(returnedRef2.key() == key2);

      reco::VertexCompositeCandidateRef returnedRef3 = proto.v0Ref();
      CPPUNIT_ASSERT(returnedRef3.refCore().id()==dummy3);
      CPPUNIT_ASSERT(returnedRef3.key() == key3);
    }

    proto.setTrackRef(ref);
    {
      CPPUNIT_ASSERT(!proto.trackRef().isNull());
      CPPUNIT_ASSERT(!proto.gsfTrackRef().isNull());
      CPPUNIT_ASSERT(!proto.v0Ref().isNull());
    
      reco::TrackRef returnedRef = proto.trackRef();
      CPPUNIT_ASSERT(returnedRef.refCore().id()==dummyID);
      CPPUNIT_ASSERT(returnedRef.key() == key);

      reco::GsfTrackRef returnedRef2 = proto.gsfTrackRef();
      CPPUNIT_ASSERT(returnedRef2.refCore().id()==dummy2);
      CPPUNIT_ASSERT(returnedRef2.key() == key2);

      reco::VertexCompositeCandidateRef returnedRef3 = proto.v0Ref();
      CPPUNIT_ASSERT(returnedRef3.refCore().id()==dummy3);
      CPPUNIT_ASSERT(returnedRef3.key() == key3);
    }
  }

  {
    //Insert 2,1,3
    PFCandidate proto(1,reco::Candidate::LorentzVector(),PFCandidate::e);
  
    CPPUNIT_ASSERT(proto.trackRef().isNull());
    CPPUNIT_ASSERT(proto.gsfTrackRef().isNull());
    CPPUNIT_ASSERT(proto.v0Ref().isNull());
    
    const unsigned int key = 10;
    reco::TrackRef ref(dummyID, 10,&dummyGetter);

    const edm::ProductID dummy2(3,4);
    const unsigned int key2(11);
    reco::GsfTrackRef ref2(dummy2,key2,&dummyGetter);

    const edm::ProductID dummy3(5,6);
    const unsigned int key3(12);
    reco::VertexCompositeCandidateRef ref3(dummy3,key3,&dummyGetter);
    
    
    proto.setGsfTrackRef(ref2);
    CPPUNIT_ASSERT(proto.trackRef().isNull());
    CPPUNIT_ASSERT(!proto.gsfTrackRef().isNull());
    CPPUNIT_ASSERT(proto.v0Ref().isNull());

    proto.setTrackRef(ref);
    {
      CPPUNIT_ASSERT(!proto.trackRef().isNull());
      CPPUNIT_ASSERT(!proto.gsfTrackRef().isNull());
      CPPUNIT_ASSERT(proto.v0Ref().isNull());
    
      reco::TrackRef returnedRef = proto.trackRef();
      CPPUNIT_ASSERT(returnedRef.refCore().id()==dummyID);
      CPPUNIT_ASSERT(returnedRef.key() == key);

      reco::GsfTrackRef returnedRef2 = proto.gsfTrackRef();
      CPPUNIT_ASSERT(returnedRef2.refCore().id()==dummy2);
      CPPUNIT_ASSERT(returnedRef2.key() == key2);
    }

    proto.setV0Ref(ref3);
    {
      CPPUNIT_ASSERT(!proto.trackRef().isNull());
      CPPUNIT_ASSERT(!proto.gsfTrackRef().isNull());
      CPPUNIT_ASSERT(!proto.v0Ref().isNull());
    
      reco::TrackRef returnedRef = proto.trackRef();
      CPPUNIT_ASSERT(returnedRef.refCore().id()==dummyID);
      CPPUNIT_ASSERT(returnedRef.key() == key);

      reco::GsfTrackRef returnedRef2 = proto.gsfTrackRef();
      CPPUNIT_ASSERT(returnedRef2.refCore().id()==dummy2);
      CPPUNIT_ASSERT(returnedRef2.key() == key2);

      reco::VertexCompositeCandidateRef returnedRef3 = proto.v0Ref();
      CPPUNIT_ASSERT(returnedRef3.refCore().id()==dummy3);
      CPPUNIT_ASSERT(returnedRef3.key() == key3);
    }
  }
 
}
