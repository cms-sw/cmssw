#include <cppunit/extensions/HelperMacros.h>
#include "DataFormats/TrackReco/interface/Track.h"

class testTrack : public CppUnit::TestFixture {
    CPPUNIT_TEST_SUITE(testTrack);
    CPPUNIT_TEST(checkAll);
    CPPUNIT_TEST_SUITE_END();

public:
    void setUp() {}
    void tearDown() {}
    void checkAll(); 
};

CPPUNIT_TEST_SUITE_REGISTRATION(testTrack);

void testTrack::checkAll() {
    const double chi2 = 20.0;
    const int ndof = 10;
    reco::Track::Point v(1, 2, 3);
    reco::Track::Vector p(10.1, 20.2, 30.3);
    double e[] = { 1.1, 
         1.2, 2.2, 
         1.3, 2.3, 3.3,
         1.4, 2.4, 3.4, 4.4,
         1.5, 2.5, 3.5, 4.5, 5.5 
    };

    reco::TrackBase::CovarianceMatrix cov(e, e + 15);
    reco::Track t(chi2, ndof, v, p, +1, cov);
    CPPUNIT_ASSERT(t.chi2() == chi2);
    CPPUNIT_ASSERT(t.ndof() == ndof);
    CPPUNIT_ASSERT(t.normalizedChi2() == chi2 / ndof);
    CPPUNIT_ASSERT(t.vertex() == v);
    CPPUNIT_ASSERT(t.momentum() == p);
    //CPPUNIT_ASSERT( t.covariance() == cov );

    int k = 0;
    CPPUNIT_ASSERT_DOUBLES_EQUAL(cov(0, 0), e[k++], 1e-6);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(cov(1, 0), e[k++], 1e-6);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(cov(1, 1), e[k++], 1e-6);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(cov(2, 0), e[k++], 1e-6);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(cov(2, 1), e[k++], 1e-6);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(cov(2, 2), e[k++], 1e-6);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(cov(3, 0), e[k++], 1e-6);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(cov(3, 1), e[k++], 1e-6);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(cov(3, 2), e[k++], 1e-6);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(cov(3, 3), e[k++], 1e-6);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(cov(4, 0), e[k++], 1e-6);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(cov(4, 1), e[k++], 1e-6);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(cov(4, 2), e[k++], 1e-6);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(cov(4, 3), e[k++], 1e-6);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(cov(4, 4), e[k++], 1e-6);

    CPPUNIT_ASSERT_DOUBLES_EQUAL(t.covariance(0, 0), cov(0, 0), 1e-6);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(t.covariance(0, 1), cov(0, 1), 1e-6);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(t.covariance(0, 2), cov(0, 2), 1e-6);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(t.covariance(0, 3), cov(0, 3), 1e-6);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(t.covariance(0, 4), cov(0, 4), 1e-6);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(t.covariance(1, 1), cov(1, 1), 1e-6);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(t.covariance(1, 2), cov(1, 2), 1e-6);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(t.covariance(1, 3), cov(1, 3), 1e-6);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(t.covariance(1, 4), cov(1, 4), 1e-6);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(t.covariance(2, 2), cov(2, 2), 1e-6);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(t.covariance(2, 3), cov(2, 3), 1e-6);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(t.covariance(2, 4), cov(2, 4), 1e-6);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(t.covariance(3, 3), cov(3, 3), 1e-6);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(t.covariance(3, 4), cov(3, 4), 1e-6);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(t.covariance(4, 4), cov(4, 4), 1e-6);

    auto qual = reco::TrackBase::qualityByName("highPurity");
    CPPUNIT_ASSERT(qual==reco::TrackBase::highPurity);
    t.setQuality(reco::TrackBase::loose);
    CPPUNIT_ASSERT(t.quality(reco::TrackBase::loose) == true);
    CPPUNIT_ASSERT(t.quality(reco::TrackBase::highPurity) == false);
    CPPUNIT_ASSERT(t.quality(reco::TrackBase::confirmed) == false);
    CPPUNIT_ASSERT(t.quality(reco::TrackBase::goodIterative) == false);
    t.setQuality(reco::TrackBase::highPurity);
    CPPUNIT_ASSERT(t.quality(reco::TrackBase::loose) == true);
    CPPUNIT_ASSERT(t.quality(reco::TrackBase::highPurity) == true);
    CPPUNIT_ASSERT(t.quality(reco::TrackBase::confirmed) == false);
    CPPUNIT_ASSERT(t.quality(reco::TrackBase::goodIterative) == true);
    t.setQuality(reco::TrackBase::confirmed);
    CPPUNIT_ASSERT(t.quality(reco::TrackBase::loose) == true);
    CPPUNIT_ASSERT(t.quality(reco::TrackBase::highPurity) == true);
    CPPUNIT_ASSERT(t.quality(reco::TrackBase::confirmed) == true);
    CPPUNIT_ASSERT(t.quality(reco::TrackBase::goodIterative) == true);
    t.setQuality(reco::TrackBase::undefQuality);
    CPPUNIT_ASSERT(t.quality(reco::TrackBase::loose) == false);
    CPPUNIT_ASSERT(t.quality(reco::TrackBase::highPurity) == false);
    CPPUNIT_ASSERT(t.quality(reco::TrackBase::confirmed) == false);
    CPPUNIT_ASSERT(t.quality(reco::TrackBase::goodIterative) == false);
    t.setQuality(reco::TrackBase::loose);
    t.setQuality(reco::TrackBase::confirmed);
    CPPUNIT_ASSERT(t.quality(reco::TrackBase::loose) == true);
    CPPUNIT_ASSERT(t.quality(reco::TrackBase::highPurity) == false);
    CPPUNIT_ASSERT(t.quality(reco::TrackBase::confirmed) == true);
    CPPUNIT_ASSERT(t.quality(reco::TrackBase::goodIterative) == false);


    reco::TrackBase::AlgoMask am;
    t.setAlgorithm(reco::TrackBase::ctf);
    am.set(reco::TrackBase::ctf);
    CPPUNIT_ASSERT(t.algo() == reco::TrackBase::ctf);
    CPPUNIT_ASSERT(t.algoMask()==am);
    am.set(reco::TrackBase::rs);
    t.setOriginalAlgorithm(reco::TrackBase::rs);
    CPPUNIT_ASSERT(t.algo() == reco::TrackBase::ctf);
    CPPUNIT_ASSERT(t.originalAlgo() == reco::TrackBase::rs);
    CPPUNIT_ASSERT(t.algoMask()==am);
    
}

