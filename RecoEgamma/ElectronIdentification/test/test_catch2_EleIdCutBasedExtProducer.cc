#include "catch.hpp"
#include "FWCore/TestProcessor/interface/TestProcessor.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include <vector>

static constexpr auto s_tag = "[EleIdCutBasedExtProducer]";

TEST_CASE("Standard checks of EleIdCutBasedExtProducer", s_tag) {
  SECTION("classbased tight default") {
    const std::string baseConfig{
        R"_(from FWCore.TestProcessor.TestProcess import *
process = TestProcess()
process.toTest = cms.EDProducer("EleIdCutBasedExtProducer",
    additionalCategories = cms.bool(True),
    algorithm = cms.string('eIDCB'),
    electronIDType = cms.string('classbased'),
    electronQuality = cms.string('tight'),
    electronVersion = cms.string(''),
    etBinning = cms.bool(True),
    reducedBarrelRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    reducedEndcapRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    src = cms.InputTag("gedGsfElectrons"),
    verticesCollection = cms.InputTag("offlinePrimaryVerticesWithBS"),
    classbasedtightEleIDCuts = cms.PSet(
        cutdcotdist = cms.vdouble(
            9999.0, 9999.0, 9999.0, 9999.0, 9999.0,
            9999.0, 9999.0, 9999.0, 9999.0
        ),
        cutdetain = cms.vdouble(
            0.0116, 0.00449, 0.00938, 0.0184, 0.00678,
            0.0109, 0.0252, 0.0268, 0.0139
        ),
        cutdetainl = cms.vdouble(
            0.00816, 0.00401, 0.0081, 0.019, 0.00588,
            0.00893, 0.0171, 0.0434, 0.0143
        ),
        cutdphiin = cms.vdouble(
            0.0897, 0.0993, 0.295, 0.0979, 0.151,
            0.252, 0.341, 0.308, 0.328
        ),
        cutdphiinl = cms.vdouble(
            0.061, 0.14, 0.286, 0.0921, 0.197,
            0.24, 0.333, 0.303, 0.258
        ),
        cuteseedopcor = cms.vdouble(
            0.637, 0.943, 0.742, 0.748, 0.763,
            0.631, 0.214, 0.873, 0.473
        ),
        cutfmishits = cms.vdouble(
            1.5, 1.5, 1.5, 2.5, 2.5,
            1.5, 1.5, 2.5, 0.5
        ),
        cuthoe = cms.vdouble(
            0.215, 0.0608, 0.147, 0.369, 0.0349,
            0.102, 0.52, 0.422, 0.404
        ),
        cuthoel = cms.vdouble(
            0.228, 0.0836, 0.143, 0.37, 0.0392,
            0.0979, 0.3, 0.381, 0.339
        ),
        cutip_gsf = cms.vdouble(
            0.0131, 0.0586, 0.0839, 0.0366, 0.452,
            0.204, 0.0913, 0.0802, 0.0731
        ),
        cutip_gsfl = cms.vdouble(
            0.0119, 0.0527, 0.0471, 0.0212, 0.233,
            0.267, 0.109, 0.122, 0.0479
        ),
        cutiso_sum = cms.vdouble(
            15.5, 12.2, 12.2, 11.7, 7.16,
            9.71, 8.66, 11.9, 2.98
        ),
        cutiso_sumoet = cms.vdouble(
            11.9, 7.81, 6.28, 8.92, 4.65,
            5.49, 9.36, 8.84, 5.94
        ),
        cutiso_sumoetl = cms.vdouble(
            6.21, 6.81, 5.3, 5.39, 2.73,
            4.73, 4.84, 3.46, 3.73
        ),
        cutsee = cms.vdouble(
            0.0145, 0.0116, 0.012, 0.039, 0.0297,
            0.0311, 0.00987, 0.0347, 0.0917
        ),
        cutseel = cms.vdouble(
            0.0132, 0.0117, 0.0112, 0.0387, 0.0281,
            0.0287, 0.00987, 0.0296, 0.0544
        )
    )
 )
process.moduleToTest(process.toTest)
)_"};

    edm::test::TestProcessor::Config config{baseConfig};
    SECTION("base configuration is OK") { REQUIRE_NOTHROW(edm::test::TestProcessor(config)); }

    SECTION("No event data") {
      edm::test::TestProcessor tester(config);

      REQUIRE_THROWS_AS(tester.test(), cms::Exception);
      //If the module does not throw when given no data, substitute
      //REQUIRE_NOTHROW for REQUIRE_THROWS_AS
    }

    SECTION("beginJob and endJob only") {
      edm::test::TestProcessor tester(config);

      REQUIRE_NOTHROW(tester.testBeginAndEndJobOnly());
    }

    SECTION("Run with no LuminosityBlocks") {
      edm::test::TestProcessor tester(config);

      REQUIRE_NOTHROW(tester.testRunWithNoLuminosityBlocks());
    }

    SECTION("LuminosityBlock with no Events") {
      edm::test::TestProcessor tester(config);

      REQUIRE_NOTHROW(tester.testLuminosityBlockWithNoEvents());
    }
    SECTION("Event with dummy data") {
      auto electronToken = config.produces<std::vector<reco::GsfElectron>>("gedGsfElectrons");
      auto vertexToken = config.produces<std::vector<reco::Vertex>>("offlinePrimaryVerticesWithBS");
      {
        edm::test::TestProcessor tester(config);

        using namespace reco;
        reco::SuperCluster cluster;
        std::vector<reco::SuperCluster> clusters(1, cluster);

        TrackExtra extra;
        std::vector<TrackExtra> extras(1, extra);
        GsfTrack track;
        track.setExtra(TrackExtraRef(&extras, 0));
        std::vector<GsfTrack> tracks(1, track);
        GsfElectronCore core(GsfTrackRef(&tracks, 0));
        core.setSuperCluster(SuperClusterRef(&clusters, 0));
        std::vector<GsfElectronCore> cores(1, core);
        reco::GsfElectron electron(1,
                                   GsfElectron::ChargeInfo(),
                                   GsfElectronCoreRef(&cores, 0),
                                   GsfElectron::TrackClusterMatching(),
                                   GsfElectron::TrackExtrapolations(),
                                   GsfElectron::ClosestCtfTrack(),
                                   GsfElectron::FiducialFlags(),
                                   GsfElectron::ShowerShape(),
                                   GsfElectron::ConversionRejection());
        std::vector<reco::GsfElectron> electrons(1, electron);

        reco::Vertex vertex;
        std::vector<reco::Vertex> vertices(1, vertex);

        REQUIRE_NOTHROW(tester.test(std::make_pair(electronToken, std::make_unique<decltype(electrons)>(electrons)),
                                    std::make_pair(vertexToken, std::make_unique<decltype(vertices)>(vertices))));
      }
    }
  }
  SECTION("classbased tight V00") {
    const std::string baseConfig{
        R"_(from FWCore.TestProcessor.TestProcess import *
process = TestProcess()
process.toTest = cms.EDProducer("EleIdCutBasedExtProducer",
    additionalCategories = cms.bool(True),
    algorithm = cms.string('eIDCB'),
    electronIDType = cms.string('classbased'),
    electronQuality = cms.string('tight'),
    electronVersion = cms.string('V00'),
    etBinning = cms.bool(True),
    reducedBarrelRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    reducedEndcapRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    src = cms.InputTag("gedGsfElectrons"),
    verticesCollection = cms.InputTag("offlinePrimaryVerticesWithBS"),
    classbasedtightEleIDCutsV00 = cms.PSet(
        deltaEtaIn = cms.vdouble(
            0.0055, 0.003, 0.0065, 0.0, 0.006,
            0.0055, 0.0075, 0.0
        ),
        deltaPhiIn = cms.vdouble(
            0.032, 0.016, 0.0525, 0.09, 0.025,
            0.035, 0.065, 0.092
        ),
        eSeedOverPin = cms.vdouble(
            0.24, 0.94, 0.11, 0.0, 0.32,
            0.83, 0.0, 0.0
        ),
        hOverE = cms.vdouble(
            0.05, 0.042, 0.045, 0.0, 0.055,
            0.037, 0.05, 0.0
        ),
        sigmaEtaEta = cms.vdouble(
            0.0125, 0.011, 0.01, 0.0, 0.0265,
            0.0252, 0.026, 0.0
        )
    )
 )
process.moduleToTest(process.toTest)
)_"};

    edm::test::TestProcessor::Config config{baseConfig};
    SECTION("base configuration is OK") { REQUIRE_NOTHROW(edm::test::TestProcessor(config)); }

    SECTION("Event with dummy data") {
      auto electronToken = config.produces<std::vector<reco::GsfElectron>>("gedGsfElectrons");
      auto vertexToken = config.produces<std::vector<reco::Vertex>>("offlinePrimaryVerticesWithBS");
      {
        edm::test::TestProcessor tester(config);

        using namespace reco;
        reco::SuperCluster cluster;
        std::vector<reco::SuperCluster> clusters(1, cluster);

        TrackExtra extra;
        std::vector<TrackExtra> extras(1, extra);
        GsfTrack track;
        track.setExtra(TrackExtraRef(&extras, 0));
        std::vector<GsfTrack> tracks(1, track);
        GsfElectronCore core(GsfTrackRef(&tracks, 0));
        core.setSuperCluster(SuperClusterRef(&clusters, 0));
        std::vector<GsfElectronCore> cores(1, core);
        reco::GsfElectron electron(1,
                                   GsfElectron::ChargeInfo(),
                                   GsfElectronCoreRef(&cores, 0),
                                   GsfElectron::TrackClusterMatching(),
                                   GsfElectron::TrackExtrapolations(),
                                   GsfElectron::ClosestCtfTrack(),
                                   GsfElectron::FiducialFlags(),
                                   GsfElectron::ShowerShape(),
                                   GsfElectron::ConversionRejection());
        std::vector<reco::GsfElectron> electrons(1, electron);

        reco::Vertex vertex;
        std::vector<reco::Vertex> vertices(1, vertex);

        REQUIRE_NOTHROW(tester.test(std::make_pair(electronToken, std::make_unique<decltype(electrons)>(electrons)),
                                    std::make_pair(vertexToken, std::make_unique<decltype(vertices)>(vertices))));
      }
    }
  }

  SECTION("classbased tight V01") {
    const std::string baseConfig{
        R"_(from FWCore.TestProcessor.TestProcess import *
process = TestProcess()
process.toTest = cms.EDProducer("EleIdCutBasedExtProducer",
    additionalCategories = cms.bool(True),
    algorithm = cms.string('eIDCB'),
    electronIDType = cms.string('classbased'),
    electronQuality = cms.string('tight'),
    electronVersion = cms.string('V01'),
    etBinning = cms.bool(True),
    reducedBarrelRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    reducedEndcapRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    src = cms.InputTag("gedGsfElectrons"),
    verticesCollection = cms.InputTag("offlinePrimaryVerticesWithBS"),
    classbasedtightEleIDCutsV01 = cms.PSet(
        deltaEtaIn = cms.vdouble(
            0.0043, 0.00282, 0.0036, 0.0, 0.0066,
            0.0049, 0.0041, 0.0
        ),
        deltaPhiIn = cms.vdouble(
            0.0225, 0.0114, 0.0234, 0.039, 0.0215,
            0.0095, 0.0148, 0.0167
        ),
        eSeedOverPin = cms.vdouble(
            0.32, 0.94, 0.221, 0.0, 0.74,
            0.89, 0.66, 0.0
        ),
        hOverE = cms.vdouble(
            0.056, 0.0221, 0.037, 0.0, 0.0268,
            0.0102, 0.0104, 0.0
        ),
        sigmaEtaEta = cms.vdouble(
            0.0095, 0.0094, 0.0094, 0.0, 0.026,
            0.0257, 0.0246, 0.0
        )
    ),
 )
process.moduleToTest(process.toTest)
)_"};

    edm::test::TestProcessor::Config config{baseConfig};
    SECTION("base configuration is OK") { REQUIRE_NOTHROW(edm::test::TestProcessor(config)); }

    SECTION("Event with dummy data") {
      auto electronToken = config.produces<std::vector<reco::GsfElectron>>("gedGsfElectrons");
      auto vertexToken = config.produces<std::vector<reco::Vertex>>("offlinePrimaryVerticesWithBS");
      {
        edm::test::TestProcessor tester(config);

        using namespace reco;
        reco::SuperCluster cluster;
        std::vector<reco::SuperCluster> clusters(1, cluster);

        TrackExtra extra;
        std::vector<TrackExtra> extras(1, extra);
        GsfTrack track;
        track.setExtra(TrackExtraRef(&extras, 0));
        std::vector<GsfTrack> tracks(1, track);
        GsfElectronCore core(GsfTrackRef(&tracks, 0));
        core.setSuperCluster(SuperClusterRef(&clusters, 0));
        std::vector<GsfElectronCore> cores(1, core);
        reco::GsfElectron electron(1,
                                   GsfElectron::ChargeInfo(),
                                   GsfElectronCoreRef(&cores, 0),
                                   GsfElectron::TrackClusterMatching(),
                                   GsfElectron::TrackExtrapolations(),
                                   GsfElectron::ClosestCtfTrack(),
                                   GsfElectron::FiducialFlags(),
                                   GsfElectron::ShowerShape(),
                                   GsfElectron::ConversionRejection());
        std::vector<reco::GsfElectron> electrons(1, electron);

        reco::Vertex vertex;
        std::vector<reco::Vertex> vertices(1, vertex);

        REQUIRE_NOTHROW(tester.test(std::make_pair(electronToken, std::make_unique<decltype(electrons)>(electrons)),
                                    std::make_pair(vertexToken, std::make_unique<decltype(vertices)>(vertices))));
      }
    }
  }

  SECTION("classbased tight V02") {
    const std::string baseConfig{
        R"_(from FWCore.TestProcessor.TestProcess import *
process = TestProcess()
process.toTest = cms.EDProducer("EleIdCutBasedExtProducer",
    additionalCategories = cms.bool(True),
    algorithm = cms.string('eIDCB'),
    electronIDType = cms.string('classbased'),
    electronQuality = cms.string('tight'),
    electronVersion = cms.string('V02'),
    etBinning = cms.bool(True),
    reducedBarrelRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    reducedEndcapRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    src = cms.InputTag("gedGsfElectrons"),
    verticesCollection = cms.InputTag("offlinePrimaryVerticesWithBS"),
    classbasedtightEleIDCutsV02 = cms.PSet(
#these were missing in the config this was original taken from and were copied from V03 below
        cutdetain = cms.vdouble(
            0.00811, 0.00341, 0.00633, 0.0103, 0.00667,
            0.01, 0.0106, 0.0145, 0.0163, 0.0076,
            0.00259, 0.00511, 0.00941, 0.0043, 0.00857,
            0.012, 0.0169, 0.00172, 0.00861, 0.00362,
            0.00601, 0.00925, 0.00489, 0.00832, 0.0119,
            0.0169, 0.000996
        ),
        cutdphiin = cms.vdouble(
            0.0404, 0.0499, 0.263, 0.042, 0.0484,
            0.241, 0.242, 0.231, 0.286, 0.0552,
            0.0338, 0.154, 0.0623, 0.0183, 0.0392,
            0.0547, 0.0588, 0.00654, 0.042, 0.0217,
            0.0885, 0.0445, 0.0141, 0.0234, 0.065,
            0.0258, 0.0346
        ),
        cuteseedopcor = cms.vdouble(
            0.784, 0.366, 0.57, 0.911, 0.298,
            0.645, 0.51, 0.497, 0.932, 0.835,
            0.968, 0.969, 0.923, 0.898, 0.98,
            0.63, 0.971, 1.0, 0.515, 0.963,
            0.986, 0.823, 0.879, 1.01, 0.931,
            0.937, 1.05
        ),
        cutet = cms.vdouble(
            -100000.0, -100000.0, -100000.0, -100000.0, -100000.0,
            -100000.0, -100000.0, -100000.0, -100000.0, -100000.0,
            -100000.0, -100000.0, -100000.0, -100000.0, -100000.0,
            -100000.0, -100000.0, -100000.0, 13.7, 13.2,
            13.6, 14.2, 14.1, 13.9, 12.9,
            14.9, 17.7
        ),
#the following are NOT used by V02
        cutdeta = cms.vdouble(
            0.00915, 0.00302, 0.0061, 0.0135, 0.00565,
            0.00793, 0.0102, 0.00266, 0.0106, 0.00903,
            0.00766, 0.00723, 0.0116, 0.00203, 0.00659,
            0.0148, 0.00555, 0.0128
        ),
        cutdphi = cms.vdouble(
            0.0369, 0.0307, 0.117, 0.0475, 0.0216,
            0.117, 0.0372, 0.0246, 0.0426, 0.0612,
            0.0142, 0.039, 0.0737, 0.0566, 0.0359,
            0.0187, 0.012, 0.0358
        ),
        cuteopin = cms.vdouble(
            0.878, 0.859, 0.874, 0.944, 0.737,
            0.773, 0.86, 0.967, 0.917, 0.812,
            0.915, 1.01, 0.847, 0.953, 0.979,
            0.841, 0.771, 1.09
        ),
#the rest are used
        cuthoe = cms.vdouble(
            0.0871, 0.0289, 0.0783, 0.0946, 0.0245,
            0.0363, 0.0671, 0.048, 0.0614, 0.0924,
            0.0158, 0.049, 0.0382, 0.0915, 0.0451,
            0.0452, 0.00196, 0.0043
        ),
        cutip = cms.vdouble(
            0.0239, 0.027, 0.0768, 0.0231, 0.178,
            0.0957, 0.0102, 0.0168, 0.043, 0.0166,
            0.0594, 0.0308, 2.1, 0.00527, 3.17,
            4.91, 0.769, 5.9
        ),
        cutisoecal = cms.vdouble(
            20.0, 27.2, 4.48, 13.5, 4.56,
            3.19, 12.2, 13.1, 7.42, 7.67,
            4.12, 4.85, 10.1, 12.4, 11.1,
            11.0, 10.6, 13.4
        ),
        cutisohcal = cms.vdouble(
            10.9, 7.01, 8.75, 3.51, 7.75,
            1.62, 11.6, 9.9, 4.97, 5.33,
            3.18, 2.32, 0.164, 5.46, 12.0,
            0.00604, 4.1, 0.000628
        ),
        cutisotk = cms.vdouble(
            6.53, 4.6, 6.0, 8.63, 3.11,
            7.77, 5.42, 4.81, 4.06, 6.47,
            2.8, 3.45, 5.29, 5.18, 15.4,
            5.38, 4.47, 0.0347
        ),
        cutmishits = cms.vdouble(
            5.5, 1.5, 0.5, 1.5, 2.5,
            0.5, 3.5, 5.5, 0.5, 0.5,
            0.5, 0.5, 0.5, 1.5, 0.5,
            0.5, 0.5, 0.5
        ),
        cutsee = cms.vdouble(
            0.0131, 0.0106, 0.0115, 0.0306, 0.028,
            0.0293, 0.0131, 0.0106, 0.0115, 0.0317,
            0.029, 0.0289, 0.0142, 0.0106, 0.0103,
            0.035, 0.0296, 0.0333
        )
    )
 )
process.moduleToTest(process.toTest)
)_"};

    edm::test::TestProcessor::Config config{baseConfig};
    SECTION("base configuration is OK") { REQUIRE_NOTHROW(edm::test::TestProcessor(config)); }

    SECTION("Event with dummy data") {
      auto electronToken = config.produces<std::vector<reco::GsfElectron>>("gedGsfElectrons");
      auto vertexToken = config.produces<std::vector<reco::Vertex>>("offlinePrimaryVerticesWithBS");
      {
        edm::test::TestProcessor tester(config);

        using namespace reco;
        reco::SuperCluster cluster;
        std::vector<reco::SuperCluster> clusters(1, cluster);

        TrackExtra extra;
        std::vector<TrackExtra> extras(1, extra);
        GsfTrack track;
        track.setExtra(TrackExtraRef(&extras, 0));
        std::vector<GsfTrack> tracks(1, track);
        GsfElectronCore core(GsfTrackRef(&tracks, 0));
        core.setSuperCluster(SuperClusterRef(&clusters, 0));
        std::vector<GsfElectronCore> cores(1, core);
        reco::GsfElectron electron(1,
                                   GsfElectron::ChargeInfo(),
                                   GsfElectronCoreRef(&cores, 0),
                                   GsfElectron::TrackClusterMatching(),
                                   GsfElectron::TrackExtrapolations(),
                                   GsfElectron::ClosestCtfTrack(),
                                   GsfElectron::FiducialFlags(),
                                   GsfElectron::ShowerShape(),
                                   GsfElectron::ConversionRejection());
        std::vector<reco::GsfElectron> electrons(1, electron);

        reco::Vertex vertex;
        std::vector<reco::Vertex> vertices(1, vertex);

        REQUIRE_NOTHROW(tester.test(std::make_pair(electronToken, std::make_unique<decltype(electrons)>(electrons)),
                                    std::make_pair(vertexToken, std::make_unique<decltype(vertices)>(vertices))));
      }
    }
  }

  SECTION("classbased tight V03") {
    const std::string baseConfig{
        R"_(from FWCore.TestProcessor.TestProcess import *
process = TestProcess()
process.toTest = cms.EDProducer("EleIdCutBasedExtProducer",
    additionalCategories = cms.bool(True),
    algorithm = cms.string('eIDCB'),
    electronIDType = cms.string('classbased'),
    electronQuality = cms.string('tight'),
    electronVersion = cms.string('V03'),
    etBinning = cms.bool(True),
    reducedBarrelRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    reducedEndcapRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    src = cms.InputTag("gedGsfElectrons"),
    verticesCollection = cms.InputTag("offlinePrimaryVerticesWithBS"),
    classbasedtightEleIDCutsV03 = cms.PSet(
        cutdcotdist = cms.vdouble(
            0.0393, 0.0256, 0.00691, 0.0394, 0.0386,
            0.039, 0.0325, 0.0384, 0.0382, 0.0245,
            0.000281, 5.46e-05, 0.0342, 0.0232, 0.00107,
            0.0178, 0.0193, 0.000758, 0.000108, 0.0248,
            0.000458, 0.0129, 0.00119, 0.0182, 4.53e-05,
            0.0189, 0.000928
        ),
        cutdetain = cms.vdouble(
            0.00811, 0.00341, 0.00633, 0.0103, 0.00667,
            0.01, 0.0106, 0.0145, 0.0163, 0.0076,
            0.00259, 0.00511, 0.00941, 0.0043, 0.00857,
            0.012, 0.0169, 0.00172, 0.00861, 0.00362,
            0.00601, 0.00925, 0.00489, 0.00832, 0.0119,
            0.0169, 0.000996
        ),
        cutdphiin = cms.vdouble(
            0.0404, 0.0499, 0.263, 0.042, 0.0484,
            0.241, 0.242, 0.231, 0.286, 0.0552,
            0.0338, 0.154, 0.0623, 0.0183, 0.0392,
            0.0547, 0.0588, 0.00654, 0.042, 0.0217,
            0.0885, 0.0445, 0.0141, 0.0234, 0.065,
            0.0258, 0.0346
        ),
        cuteseedopcor = cms.vdouble(
            0.784, 0.366, 0.57, 0.911, 0.298,
            0.645, 0.51, 0.497, 0.932, 0.835,
            0.968, 0.969, 0.923, 0.898, 0.98,
            0.63, 0.971, 1.0, 0.515, 0.963,
            0.986, 0.823, 0.879, 1.01, 0.931,
            0.937, 1.05
        ),
        cutet = cms.vdouble(
            -100000.0, -100000.0, -100000.0, -100000.0, -100000.0,
            -100000.0, -100000.0, -100000.0, -100000.0, -100000.0,
            -100000.0, -100000.0, -100000.0, -100000.0, -100000.0,
            -100000.0, -100000.0, -100000.0, 13.7, 13.2,
            13.6, 14.2, 14.1, 13.9, 12.9,
            14.9, 17.7
        ),
        cutfmishits = cms.vdouble(
            2.5, 1.5, 1.5, 1.5, 1.5,
            0.5, 2.5, 0.5, 0.5, 2.5,
            1.5, 0.5, 0.5, 0.5, 0.5,
            0.5, 0.5, -0.5, 2.5, 1.5,
            0.5, 0.5, 0.5, 0.5, 0.5,
            0.5, 0.5
        ),
        cuthoe = cms.vdouble(
            0.0783, 0.0387, 0.105, 0.118, 0.0227,
            0.062, 0.13, 2.47, 0.38, 0.0888,
            0.0503, 0.0955, 0.0741, 0.015, 0.03,
            0.589, 1.13, 0.612, 0.0494, 0.0461,
            0.0292, 0.0369, 0.0113, 0.0145, 0.124,
            2.05, 0.61
        ),
        cutip_gsf = cms.vdouble(
            0.0213, 0.0422, 0.0632, 0.0361, 0.073,
            0.126, 0.171, 0.119, 0.0372, 0.0131,
            0.0146, 0.0564, 0.0152, 0.0222, 0.0268,
            0.0314, 0.0884, 0.00374, 0.00852, 0.00761,
            0.0143, 0.0106, 0.0127, 0.0119, 0.0123,
            0.0235, 0.00363
        ),
        cutiso_sum = cms.vdouble(
            11.8, 8.31, 6.26, 6.18, 3.28,
            4.38, 4.17, 5.4, 1.57, 100000.0,
            100000.0, 100000.0, 100000.0, 100000.0, 100000.0,
            100000.0, 100000.0, 100000.0, 100000.0, 100000.0,
            100000.0, 100000.0, 100000.0, 100000.0, 100000.0,
            100000.0, 100000.0
        ),
        cutiso_sumoet = cms.vdouble(
            13.7, 11.6, 7.14, 9.98, 3.52,
            4.87, 6.24, 7.96, 2.53, 11.2,
            11.9, 7.88, 8.16, 5.58, 5.03,
            11.4, 8.15, 5.79, 10.4, 11.1,
            10.4, 7.47, 5.08, 5.9, 11.8,
            14.1, 11.7
        ),
        cutsee = cms.vdouble(
            0.0143, 0.0105, 0.0123, 0.0324, 0.0307,
            0.0301, 0.0109, 0.027, 0.0292, 0.0133,
            0.0104, 0.0116, 0.0332, 0.0296, 0.031,
            0.00981, 0.0307, 0.072, 0.0149, 0.0105,
            0.011, 0.0342, 0.0307, 0.0303, 0.00954,
            0.0265, 0.0101
        )
    ),
 )
process.moduleToTest(process.toTest)
)_"};

    edm::test::TestProcessor::Config config{baseConfig};
    SECTION("base configuration is OK") { REQUIRE_NOTHROW(edm::test::TestProcessor(config)); }

    SECTION("Event with dummy data") {
      auto electronToken = config.produces<std::vector<reco::GsfElectron>>("gedGsfElectrons");
      auto vertexToken = config.produces<std::vector<reco::Vertex>>("offlinePrimaryVerticesWithBS");
      {
        edm::test::TestProcessor tester(config);

        using namespace reco;
        reco::SuperCluster cluster;
        std::vector<reco::SuperCluster> clusters(1, cluster);

        TrackExtra extra;
        std::vector<TrackExtra> extras(1, extra);
        GsfTrack track;
        track.setExtra(TrackExtraRef(&extras, 0));
        std::vector<GsfTrack> tracks(1, track);
        GsfElectronCore core(GsfTrackRef(&tracks, 0));
        core.setSuperCluster(SuperClusterRef(&clusters, 0));
        std::vector<GsfElectronCore> cores(1, core);
        reco::GsfElectron electron(1,
                                   GsfElectron::ChargeInfo(),
                                   GsfElectronCoreRef(&cores, 0),
                                   GsfElectron::TrackClusterMatching(),
                                   GsfElectron::TrackExtrapolations(),
                                   GsfElectron::ClosestCtfTrack(),
                                   GsfElectron::FiducialFlags(),
                                   GsfElectron::ShowerShape(),
                                   GsfElectron::ConversionRejection());
        std::vector<reco::GsfElectron> electrons(1, electron);

        reco::Vertex vertex;
        std::vector<reco::Vertex> vertices(1, vertex);

        REQUIRE_NOTHROW(tester.test(std::make_pair(electronToken, std::make_unique<decltype(electrons)>(electrons)),
                                    std::make_pair(vertexToken, std::make_unique<decltype(vertices)>(vertices))));
      }
    }
  }

  SECTION("classbased tight V04") {
    const std::string baseConfig{
        R"_(from FWCore.TestProcessor.TestProcess import *
process = TestProcess()
process.toTest = cms.EDProducer("EleIdCutBasedExtProducer",
    additionalCategories = cms.bool(True),
    algorithm = cms.string('eIDCB'),
    electronIDType = cms.string('classbased'),
    electronQuality = cms.string('tight'),
    electronVersion = cms.string('V04'),
    etBinning = cms.bool(True),
    reducedBarrelRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    reducedEndcapRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    src = cms.InputTag("gedGsfElectrons"),
    verticesCollection = cms.InputTag("offlinePrimaryVerticesWithBS"),
    classbasedtightEleIDCutsV04 = cms.PSet(
        cutdcotdist = cms.vdouble(
            0.0393, 0.0256, 0.00691, 0.0394, 0.0386,
            0.039, 0.0325, 0.0384, 0.0382, 0.0245,
            0.000281, 5.46e-05, 0.0342, 0.0232, 0.00107,
            0.0178, 0.0193, 0.000758, 0.000108, 0.0248,
            0.000458, 0.0129, 0.00119, 0.0182, 4.53e-05,
            0.0189, 0.000928
        ),
        cutdetain = cms.vdouble(
            0.00811, 0.00341, 0.00633, 0.0103, 0.00667,
            0.01, 0.0106, 0.0145, 0.0163, 0.0076,
            0.00259, 0.00511, 0.00941, 0.0043, 0.00857,
            0.012, 0.0169, 0.00172, 0.00861, 0.00362,
            0.00601, 0.00925, 0.00489, 0.00832, 0.0119,
            0.0169, 0.000996
        ),
        cutdphiin = cms.vdouble(
            0.0404, 0.0499, 0.263, 0.042, 0.0484,
            0.241, 0.242, 0.231, 0.286, 0.0552,
            0.0338, 0.154, 0.0623, 0.0183, 0.0392,
            0.0547, 0.0588, 0.00654, 0.042, 0.0217,
            0.0885, 0.0445, 0.0141, 0.0234, 0.065,
            0.0258, 0.0346
        ),
        cuteseedopcor = cms.vdouble(
            0.784, 0.366, 0.57, 0.911, 0.298,
            0.645, 0.51, 0.497, 0.932, 0.835,
            0.968, 0.969, 0.923, 0.898, 0.98,
            0.63, 0.971, 1.0, 0.515, 0.963,
            0.986, 0.823, 0.879, 1.01, 0.931,
            0.937, 1.05
        ),
        cutet = cms.vdouble(
            -100000.0, -100000.0, -100000.0, -100000.0, -100000.0,
            -100000.0, -100000.0, -100000.0, -100000.0, -100000.0,
            -100000.0, -100000.0, -100000.0, -100000.0, -100000.0,
            -100000.0, -100000.0, -100000.0, 13.7, 13.2,
            13.6, 14.2, 14.1, 13.9, 12.9,
            14.9, 17.7
        ),
        cutfmishits = cms.vdouble(
            2.5, 1.5, 1.5, 1.5, 1.5,
            0.5, 2.5, 0.5, 0.5, 2.5,
            1.5, 0.5, 0.5, 0.5, 0.5,
            0.5, 0.5, -0.5, 2.5, 1.5,
            0.5, 0.5, 0.5, 0.5, 0.5,
            0.5, 0.5
        ),
        cuthoe = cms.vdouble(
            0.0783, 0.0387, 0.105, 0.118, 0.0227,
            0.062, 0.13, 2.47, 0.38, 0.0888,
            0.0503, 0.0955, 0.0741, 0.015, 0.03,
            0.589, 1.13, 0.612, 0.0494, 0.0461,
            0.0292, 0.0369, 0.0113, 0.0145, 0.124,
            2.05, 0.61
        ),
        cutip_gsf = cms.vdouble(
            0.0213, 0.0422, 0.0632, 0.0361, 0.073,
            0.126, 0.171, 0.119, 0.0372, 0.0131,
            0.0146, 0.0564, 0.0152, 0.0222, 0.0268,
            0.0314, 0.0884, 0.00374, 0.00852, 0.00761,
            0.0143, 0.0106, 0.0127, 0.0119, 0.0123,
            0.0235, 0.00363
        ),
        cutiso_sum = cms.vdouble(
            11.8, 8.31, 6.26, 6.18, 3.28,
            4.38, 4.17, 5.4, 1.57, 100000.0,
            100000.0, 100000.0, 100000.0, 100000.0, 100000.0,
            100000.0, 100000.0, 100000.0, 100000.0, 100000.0,
            100000.0, 100000.0, 100000.0, 100000.0, 100000.0,
            100000.0, 100000.0
        ),
        cutiso_sumoet = cms.vdouble(
            13.7, 11.6, 7.14, 9.98, 3.52,
            4.87, 6.24, 7.96, 2.53, 11.2,
            11.9, 7.88, 8.16, 5.58, 5.03,
            11.4, 8.15, 5.79, 10.4, 11.1,
            10.4, 7.47, 5.08, 5.9, 11.8,
            14.1, 11.7
        ),
        cutsee = cms.vdouble(
            0.0143, 0.0105, 0.0123, 0.0324, 0.0307,
            0.0301, 0.0109, 0.027, 0.0292, 0.0133,
            0.0104, 0.0116, 0.0332, 0.0296, 0.031,
            0.00981, 0.0307, 0.072, 0.0149, 0.0105,
            0.011, 0.0342, 0.0307, 0.0303, 0.00954,
            0.0265, 0.0101
        )
    ),
 )
process.moduleToTest(process.toTest)
)_"};

    edm::test::TestProcessor::Config config{baseConfig};
    SECTION("base configuration is OK") { REQUIRE_NOTHROW(edm::test::TestProcessor(config)); }

    SECTION("Event with dummy data") {
      auto electronToken = config.produces<std::vector<reco::GsfElectron>>("gedGsfElectrons");
      auto vertexToken = config.produces<std::vector<reco::Vertex>>("offlinePrimaryVerticesWithBS");
      {
        edm::test::TestProcessor tester(config);

        using namespace reco;
        reco::SuperCluster cluster;
        std::vector<reco::SuperCluster> clusters(1, cluster);

        TrackExtra extra;
        std::vector<TrackExtra> extras(1, extra);
        GsfTrack track;
        track.setExtra(TrackExtraRef(&extras, 0));
        std::vector<GsfTrack> tracks(1, track);
        GsfElectronCore core(GsfTrackRef(&tracks, 0));
        core.setSuperCluster(SuperClusterRef(&clusters, 0));
        std::vector<GsfElectronCore> cores(1, core);
        reco::GsfElectron electron(1,
                                   GsfElectron::ChargeInfo(),
                                   GsfElectronCoreRef(&cores, 0),
                                   GsfElectron::TrackClusterMatching(),
                                   GsfElectron::TrackExtrapolations(),
                                   GsfElectron::ClosestCtfTrack(),
                                   GsfElectron::FiducialFlags(),
                                   GsfElectron::ShowerShape(),
                                   GsfElectron::ConversionRejection());
        std::vector<reco::GsfElectron> electrons(1, electron);

        reco::Vertex vertex;
        std::vector<reco::Vertex> vertices(1, vertex);

        REQUIRE_NOTHROW(tester.test(std::make_pair(electronToken, std::make_unique<decltype(electrons)>(electrons)),
                                    std::make_pair(vertexToken, std::make_unique<decltype(vertices)>(vertices))));
      }
    }
  }
  //NOTE no example of V05 was available in CMSSW
  SECTION("classbased tight V06") {
    const std::string baseConfig{
        R"_(from FWCore.TestProcessor.TestProcess import *
process = TestProcess()
process.toTest = cms.EDProducer("EleIdCutBasedExtProducer",
    additionalCategories = cms.bool(True),
    algorithm = cms.string('eIDCB'),
    electronIDType = cms.string('classbased'),
    electronQuality = cms.string('tight'),
    electronVersion = cms.string('V06'),
    etBinning = cms.bool(True),
    reducedBarrelRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    reducedEndcapRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    src = cms.InputTag("gedGsfElectrons"),
    verticesCollection = cms.InputTag("offlinePrimaryVerticesWithBS"),
    classbasedtightEleIDCutsV06 = cms.PSet(
        cutdcotdist = cms.vdouble(
            9999.0, 9999.0, 9999.0, 9999.0, 9999.0,
            9999.0, 9999.0, 9999.0, 9999.0
        ),
        cutdetain = cms.vdouble(
            0.0116, 0.00449, 0.00938, 0.0184, 0.00678,
            0.0109, 0.0252, 0.0268, 0.0139
        ),
        cutdetainl = cms.vdouble(
            0.00816, 0.00401, 0.0081, 0.019, 0.00588,
            0.00893, 0.0171, 0.0434, 0.0143
        ),
        cutdphiin = cms.vdouble(
            0.0897, 0.0993, 0.295, 0.0979, 0.151,
            0.252, 0.341, 0.308, 0.328
        ),
        cutdphiinl = cms.vdouble(
            0.061, 0.14, 0.286, 0.0921, 0.197,
            0.24, 0.333, 0.303, 0.258
        ),
        cuteseedopcor = cms.vdouble(
            0.637, 0.943, 0.742, 0.748, 0.763,
            0.631, 0.214, 0.873, 0.473
        ),
        cutfmishits = cms.vdouble(
            1.5, 1.5, 1.5, 2.5, 2.5,
            1.5, 1.5, 2.5, 0.5
        ),
        cuthoe = cms.vdouble(
            0.215, 0.0608, 0.147, 0.369, 0.0349,
            0.102, 0.52, 0.422, 0.404
        ),
        cuthoel = cms.vdouble(
            0.228, 0.0836, 0.143, 0.37, 0.0392,
            0.0979, 0.3, 0.381, 0.339
        ),
        cutip_gsf = cms.vdouble(
            0.0131, 0.0586, 0.0839, 0.0366, 0.452,
            0.204, 0.0913, 0.0802, 0.0731
        ),
        cutip_gsfl = cms.vdouble(
            0.0119, 0.0527, 0.0471, 0.0212, 0.233,
            0.267, 0.109, 0.122, 0.0479
        ),
        cutiso_sum = cms.vdouble(
            15.5, 12.2, 12.2, 11.7, 7.16,
            9.71, 8.66, 11.9, 2.98
        ),
        cutiso_sumoet = cms.vdouble(
            11.9, 7.81, 6.28, 8.92, 4.65,
            5.49, 9.36, 8.84, 5.94
        ),
        cutiso_sumoetl = cms.vdouble(
            6.21, 6.81, 5.3, 5.39, 2.73,
            4.73, 4.84, 3.46, 3.73
        ),
        cutsee = cms.vdouble(
            0.0145, 0.0116, 0.012, 0.039, 0.0297,
            0.0311, 0.00987, 0.0347, 0.0917
        ),
        cutseel = cms.vdouble(
            0.0132, 0.0117, 0.0112, 0.0387, 0.0281,
            0.0287, 0.00987, 0.0296, 0.0544
        )
    ),
 )
process.moduleToTest(process.toTest)
)_"};

    edm::test::TestProcessor::Config config{baseConfig};
    SECTION("base configuration is OK") { REQUIRE_NOTHROW(edm::test::TestProcessor(config)); }

    SECTION("Event with dummy data") {
      auto electronToken = config.produces<std::vector<reco::GsfElectron>>("gedGsfElectrons");
      auto vertexToken = config.produces<std::vector<reco::Vertex>>("offlinePrimaryVerticesWithBS");
      {
        edm::test::TestProcessor tester(config);

        using namespace reco;
        reco::SuperCluster cluster;
        std::vector<reco::SuperCluster> clusters(1, cluster);

        TrackExtra extra;
        std::vector<TrackExtra> extras(1, extra);
        GsfTrack track;
        track.setExtra(TrackExtraRef(&extras, 0));
        std::vector<GsfTrack> tracks(1, track);
        GsfElectronCore core(GsfTrackRef(&tracks, 0));
        core.setSuperCluster(SuperClusterRef(&clusters, 0));
        std::vector<GsfElectronCore> cores(1, core);
        reco::GsfElectron electron(1,
                                   GsfElectron::ChargeInfo(),
                                   GsfElectronCoreRef(&cores, 0),
                                   GsfElectron::TrackClusterMatching(),
                                   GsfElectron::TrackExtrapolations(),
                                   GsfElectron::ClosestCtfTrack(),
                                   GsfElectron::FiducialFlags(),
                                   GsfElectron::ShowerShape(),
                                   GsfElectron::ConversionRejection());
        std::vector<reco::GsfElectron> electrons(1, electron);

        reco::Vertex vertex;
        std::vector<reco::Vertex> vertices(1, vertex);

        REQUIRE_NOTHROW(tester.test(std::make_pair(electronToken, std::make_unique<decltype(electrons)>(electrons)),
                                    std::make_pair(vertexToken, std::make_unique<decltype(vertices)>(vertices))));
      }
    }
  }

  SECTION("classbased robust default") {
    const std::string baseConfig{
        R"_(from FWCore.TestProcessor.TestProcess import *
process = TestProcess()
process.toTest = cms.EDProducer("EleIdCutBasedExtProducer",
    additionalCategories = cms.bool(True),
    algorithm = cms.string('eIDCB'),
    electronIDType = cms.string('robust'),
    electronQuality = cms.string('tight'),
    electronVersion = cms.string(''),
    etBinning = cms.bool(True),
    reducedBarrelRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    reducedEndcapRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    src = cms.InputTag("gedGsfElectrons"),
    verticesCollection = cms.InputTag("offlinePrimaryVerticesWithBS"),
    robusttightEleIDCuts = cms.PSet(
        barrel = cms.vdouble(
            0.0201, 0.0102, 0.0211, 0.00606, -1,
            -1, 2.34, 3.24, 4.51, 9999.0,
            9999.0, 9999.0, 9999.0, 9999.0, 9999.0,
            9999.0, 9999.0, 9999.0, 0.0, -9999.0,
            9999.0, 9999.0, 9999, -1, 0,
            0
        ),
        endcap = cms.vdouble(
            0.00253, 0.0291, 0.022, 0.0032, -1,
            -1, 0.826, 2.7, 0.255, 9999.0,
            9999.0, 9999.0, 9999.0, 9999.0, 9999.0,
            9999.0, 9999.0, 9999.0, 0.0, -9999.0,
            9999.0, 9999.0, 9999, -1, 0,
            0
        )
    )
 )
process.moduleToTest(process.toTest)
)_"};

    edm::test::TestProcessor::Config config{baseConfig};
    SECTION("base configuration is OK") { REQUIRE_NOTHROW(edm::test::TestProcessor(config)); }

    SECTION("Event with dummy data") {
      auto electronToken = config.produces<std::vector<reco::GsfElectron>>("gedGsfElectrons");
      auto vertexToken = config.produces<std::vector<reco::Vertex>>("offlinePrimaryVerticesWithBS");
      {
        edm::test::TestProcessor tester(config);

        using namespace reco;
        reco::SuperCluster cluster;
        std::vector<reco::SuperCluster> clusters(1, cluster);

        TrackExtra extra;
        std::vector<TrackExtra> extras(1, extra);
        GsfTrack track;
        track.setExtra(TrackExtraRef(&extras, 0));
        std::vector<GsfTrack> tracks(1, track);
        GsfElectronCore core(GsfTrackRef(&tracks, 0));
        core.setSuperCluster(SuperClusterRef(&clusters, 0));
        std::vector<GsfElectronCore> cores(1, core);
        reco::GsfElectron electron(1,
                                   GsfElectron::ChargeInfo(),
                                   GsfElectronCoreRef(&cores, 0),
                                   GsfElectron::TrackClusterMatching(),
                                   GsfElectron::TrackExtrapolations(),
                                   GsfElectron::ClosestCtfTrack(),
                                   GsfElectron::FiducialFlags(),
                                   GsfElectron::ShowerShape(),
                                   GsfElectron::ConversionRejection());
        std::vector<reco::GsfElectron> electrons(1, electron);

        reco::Vertex vertex;
        std::vector<reco::Vertex> vertices(1, vertex);

        REQUIRE_NOTHROW(tester.test(std::make_pair(electronToken, std::make_unique<decltype(electrons)>(electrons)),
                                    std::make_pair(vertexToken, std::make_unique<decltype(vertices)>(vertices))));
      }
    }
  }

  SECTION("classbased robust V00") {
    const std::string baseConfig{
        R"_(from FWCore.TestProcessor.TestProcess import *
process = TestProcess()
process.toTest = cms.EDProducer("EleIdCutBasedExtProducer",
    additionalCategories = cms.bool(True),
    algorithm = cms.string('eIDCB'),
    electronIDType = cms.string('robust'),
    electronQuality = cms.string('tight'),
    electronVersion = cms.string('V00'),
    etBinning = cms.bool(True),
    reducedBarrelRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    reducedEndcapRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    src = cms.InputTag("gedGsfElectrons"),
    verticesCollection = cms.InputTag("offlinePrimaryVerticesWithBS"),
    robusttightEleIDCutsV00 = cms.PSet(
        barrel = cms.vdouble(
            0.015, 0.0092, 0.02, 0.0025, -1,
            -1, 9999.0, 9999.0, 9999.0, 9999.0,
            9999.0, 9999.0, 9999.0, 9999.0, 9999.0,
            9999.0, 9999.0, 9999.0, 0.0, -9999.0,
            9999.0, 9999.0, 9999, -1, 0,
            0
        ),
        endcap = cms.vdouble(
            0.018, 0.025, 0.02, 0.004, -1,
            -1, 9999.0, 9999.0, 9999.0, 9999.0,
            9999.0, 9999.0, 9999.0, 9999.0, 9999.0,
            9999.0, 9999.0, 9999.0, 0.0, -9999.0,
            9999.0, 9999.0, 9999, -1, 0,
            0
        )
    ),
 )
process.moduleToTest(process.toTest)
)_"};

    edm::test::TestProcessor::Config config{baseConfig};
    SECTION("base configuration is OK") { REQUIRE_NOTHROW(edm::test::TestProcessor(config)); }

    SECTION("Event with dummy data") {
      auto electronToken = config.produces<std::vector<reco::GsfElectron>>("gedGsfElectrons");
      auto vertexToken = config.produces<std::vector<reco::Vertex>>("offlinePrimaryVerticesWithBS");
      {
        edm::test::TestProcessor tester(config);

        using namespace reco;
        reco::SuperCluster cluster;
        std::vector<reco::SuperCluster> clusters(1, cluster);

        TrackExtra extra;
        std::vector<TrackExtra> extras(1, extra);
        GsfTrack track;
        track.setExtra(TrackExtraRef(&extras, 0));
        std::vector<GsfTrack> tracks(1, track);
        GsfElectronCore core(GsfTrackRef(&tracks, 0));
        core.setSuperCluster(SuperClusterRef(&clusters, 0));
        std::vector<GsfElectronCore> cores(1, core);
        reco::GsfElectron electron(1,
                                   GsfElectron::ChargeInfo(),
                                   GsfElectronCoreRef(&cores, 0),
                                   GsfElectron::TrackClusterMatching(),
                                   GsfElectron::TrackExtrapolations(),
                                   GsfElectron::ClosestCtfTrack(),
                                   GsfElectron::FiducialFlags(),
                                   GsfElectron::ShowerShape(),
                                   GsfElectron::ConversionRejection());
        std::vector<reco::GsfElectron> electrons(1, electron);

        reco::Vertex vertex;
        std::vector<reco::Vertex> vertices(1, vertex);

        REQUIRE_NOTHROW(tester.test(std::make_pair(electronToken, std::make_unique<decltype(electrons)>(electrons)),
                                    std::make_pair(vertexToken, std::make_unique<decltype(vertices)>(vertices))));
      }
    }
  }

  SECTION("classbased robust V01") {
    const std::string baseConfig{
        R"_(from FWCore.TestProcessor.TestProcess import *
process = TestProcess()
process.toTest = cms.EDProducer("EleIdCutBasedExtProducer",
    additionalCategories = cms.bool(True),
    algorithm = cms.string('eIDCB'),
    electronIDType = cms.string('robust'),
    electronQuality = cms.string('tight'),
    electronVersion = cms.string('V01'),
    etBinning = cms.bool(True),
    reducedBarrelRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    reducedEndcapRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    src = cms.InputTag("gedGsfElectrons"),
    verticesCollection = cms.InputTag("offlinePrimaryVerticesWithBS"),
    robusttightEleIDCutsV01 = cms.PSet(
        barrel = cms.vdouble(
            0.01, 0.0099, 0.025, 0.004, -1,
            -1, 9999.0, 9999.0, 9999.0, 9999.0,
            9999.0, 9999.0, 9999.0, 9999.0, 9999.0,
            9999.0, 9999.0, 9999.0, 0.0, -9999.0,
            9999.0, 9999.0, 9999, -1, 0,
            0
        ),
        endcap = cms.vdouble(
            0.01, 0.028, 0.02, 0.0066, -1,
            -1, 9999.0, 9999.0, 9999.0, 9999.0,
            9999.0, 9999.0, 9999.0, 9999.0, 9999.0,
            9999.0, 9999.0, 9999.0, 0.0, -9999.0,
            9999.0, 9999.0, 9999, -1, 0,
            0
        )
    ),
 )
process.moduleToTest(process.toTest)
)_"};

    edm::test::TestProcessor::Config config{baseConfig};
    SECTION("base configuration is OK") { REQUIRE_NOTHROW(edm::test::TestProcessor(config)); }

    SECTION("Event with dummy data") {
      auto electronToken = config.produces<std::vector<reco::GsfElectron>>("gedGsfElectrons");
      auto vertexToken = config.produces<std::vector<reco::Vertex>>("offlinePrimaryVerticesWithBS");
      {
        edm::test::TestProcessor tester(config);

        using namespace reco;
        reco::SuperCluster cluster;
        std::vector<reco::SuperCluster> clusters(1, cluster);

        TrackExtra extra;
        std::vector<TrackExtra> extras(1, extra);
        GsfTrack track;
        track.setExtra(TrackExtraRef(&extras, 0));
        std::vector<GsfTrack> tracks(1, track);
        GsfElectronCore core(GsfTrackRef(&tracks, 0));
        core.setSuperCluster(SuperClusterRef(&clusters, 0));
        std::vector<GsfElectronCore> cores(1, core);
        reco::GsfElectron electron(1,
                                   GsfElectron::ChargeInfo(),
                                   GsfElectronCoreRef(&cores, 0),
                                   GsfElectron::TrackClusterMatching(),
                                   GsfElectron::TrackExtrapolations(),
                                   GsfElectron::ClosestCtfTrack(),
                                   GsfElectron::FiducialFlags(),
                                   GsfElectron::ShowerShape(),
                                   GsfElectron::ConversionRejection());
        std::vector<reco::GsfElectron> electrons(1, electron);

        reco::Vertex vertex;
        std::vector<reco::Vertex> vertices(1, vertex);

        REQUIRE_NOTHROW(tester.test(std::make_pair(electronToken, std::make_unique<decltype(electrons)>(electrons)),
                                    std::make_pair(vertexToken, std::make_unique<decltype(vertices)>(vertices))));
      }
    }
  }

  SECTION("classbased robust V02") {
    const std::string baseConfig{
        R"_(from FWCore.TestProcessor.TestProcess import *
process = TestProcess()
process.toTest = cms.EDProducer("EleIdCutBasedExtProducer",
    additionalCategories = cms.bool(True),
    algorithm = cms.string('eIDCB'),
    electronIDType = cms.string('robust'),
    electronQuality = cms.string('tight'),
    electronVersion = cms.string('V02'),
    etBinning = cms.bool(True),
    reducedBarrelRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    reducedEndcapRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    src = cms.InputTag("gedGsfElectrons"),
    verticesCollection = cms.InputTag("offlinePrimaryVerticesWithBS"),
    robusttightEleIDCutsV02 = cms.PSet(
        barrel = cms.vdouble(
            0.0201, 0.0102, 0.0211, 0.00606, -1,
            -1, 2.34, 3.24, 4.51, 9999.0,
            9999.0, 9999.0, 9999.0, 9999.0, 9999.0,
            9999.0, 9999.0, 9999.0, 0.0, -9999.0,
            9999.0, 9999.0, 9999, -1, 0,
            0
        ),
        endcap = cms.vdouble(
            0.00253, 0.0291, 0.022, 0.0032, -1,
            -1, 0.826, 2.7, 0.255, 9999.0,
            9999.0, 9999.0, 9999.0, 9999.0, 9999.0,
            9999.0, 9999.0, 9999.0, 0.0, -9999.0,
            9999.0, 9999.0, 9999, -1, 0,
            0
        )
    ),
 )
process.moduleToTest(process.toTest)
)_"};

    edm::test::TestProcessor::Config config{baseConfig};
    SECTION("base configuration is OK") { REQUIRE_NOTHROW(edm::test::TestProcessor(config)); }

    SECTION("Event with dummy data") {
      auto electronToken = config.produces<std::vector<reco::GsfElectron>>("gedGsfElectrons");
      auto vertexToken = config.produces<std::vector<reco::Vertex>>("offlinePrimaryVerticesWithBS");
      {
        edm::test::TestProcessor tester(config);

        using namespace reco;
        reco::SuperCluster cluster;
        std::vector<reco::SuperCluster> clusters(1, cluster);

        TrackExtra extra;
        std::vector<TrackExtra> extras(1, extra);
        GsfTrack track;
        track.setExtra(TrackExtraRef(&extras, 0));
        std::vector<GsfTrack> tracks(1, track);
        GsfElectronCore core(GsfTrackRef(&tracks, 0));
        core.setSuperCluster(SuperClusterRef(&clusters, 0));
        std::vector<GsfElectronCore> cores(1, core);
        reco::GsfElectron electron(1,
                                   GsfElectron::ChargeInfo(),
                                   GsfElectronCoreRef(&cores, 0),
                                   GsfElectron::TrackClusterMatching(),
                                   GsfElectron::TrackExtrapolations(),
                                   GsfElectron::ClosestCtfTrack(),
                                   GsfElectron::FiducialFlags(),
                                   GsfElectron::ShowerShape(),
                                   GsfElectron::ConversionRejection());
        std::vector<reco::GsfElectron> electrons(1, electron);

        reco::Vertex vertex;
        std::vector<reco::Vertex> vertices(1, vertex);

        REQUIRE_NOTHROW(tester.test(std::make_pair(electronToken, std::make_unique<decltype(electrons)>(electrons)),
                                    std::make_pair(vertexToken, std::make_unique<decltype(vertices)>(vertices))));
      }
    }
  }
  SECTION("classbased robust V03") {
    const std::string baseConfig{
        R"_(from FWCore.TestProcessor.TestProcess import *
process = TestProcess()
process.toTest = cms.EDProducer("EleIdCutBasedExtProducer",
    additionalCategories = cms.bool(True),
    algorithm = cms.string('eIDCB'),
    electronIDType = cms.string('robust'),
    electronQuality = cms.string('tight'),
    electronVersion = cms.string('V03'),
    etBinning = cms.bool(True),
    reducedBarrelRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    reducedEndcapRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    src = cms.InputTag("gedGsfElectrons"),
    verticesCollection = cms.InputTag("offlinePrimaryVerticesWithBS"),
    robusttightEleIDCutsV03 = cms.PSet(
        barrel = cms.vdouble(
            0.0201, 0.0102, 0.0211, 0.00606, -1,
            -1, 2.34, 3.24, 4.51, 9999.0,
            9999.0, 9999.0, 9999.0, 9999.0, 9999.0,
            9999.0, 9999.0, 9999.0, 0.0, -9999.0,
            9999.0, 9999.0, 9999, -1, 0,
            0
        ),
        endcap = cms.vdouble(
            0.00253, 0.0291, 0.022, 0.0032, -1,
            -1, 0.826, 2.7, 0.255, 9999.0,
            9999.0, 9999.0, 9999.0, 9999.0, 9999.0,
            9999.0, 9999.0, 9999.0, 0.0, -9999.0,
            9999.0, 9999.0, 9999, -1, 0,
            0
        )
    ),
 )
process.moduleToTest(process.toTest)
)_"};

    edm::test::TestProcessor::Config config{baseConfig};
    SECTION("base configuration is OK") { REQUIRE_NOTHROW(edm::test::TestProcessor(config)); }

    SECTION("Event with dummy data") {
      auto electronToken = config.produces<std::vector<reco::GsfElectron>>("gedGsfElectrons");
      auto vertexToken = config.produces<std::vector<reco::Vertex>>("offlinePrimaryVerticesWithBS");
      {
        edm::test::TestProcessor tester(config);

        using namespace reco;
        reco::SuperCluster cluster;
        std::vector<reco::SuperCluster> clusters(1, cluster);

        TrackExtra extra;
        std::vector<TrackExtra> extras(1, extra);
        GsfTrack track;
        track.setExtra(TrackExtraRef(&extras, 0));
        std::vector<GsfTrack> tracks(1, track);
        GsfElectronCore core(GsfTrackRef(&tracks, 0));
        core.setSuperCluster(SuperClusterRef(&clusters, 0));
        std::vector<GsfElectronCore> cores(1, core);
        reco::GsfElectron electron(1,
                                   GsfElectron::ChargeInfo(),
                                   GsfElectronCoreRef(&cores, 0),
                                   GsfElectron::TrackClusterMatching(),
                                   GsfElectron::TrackExtrapolations(),
                                   GsfElectron::ClosestCtfTrack(),
                                   GsfElectron::FiducialFlags(),
                                   GsfElectron::ShowerShape(),
                                   GsfElectron::ConversionRejection());
        std::vector<reco::GsfElectron> electrons(1, electron);

        reco::Vertex vertex;
        std::vector<reco::Vertex> vertices(1, vertex);

        REQUIRE_NOTHROW(tester.test(std::make_pair(electronToken, std::make_unique<decltype(electrons)>(electrons)),
                                    std::make_pair(vertexToken, std::make_unique<decltype(vertices)>(vertices))));
      }
    }
  }
  SECTION("classbased robust V04") {
    const std::string baseConfig{
        R"_(from FWCore.TestProcessor.TestProcess import *
process = TestProcess()
process.toTest = cms.EDProducer("EleIdCutBasedExtProducer",
    additionalCategories = cms.bool(True),
    algorithm = cms.string('eIDCB'),
    electronIDType = cms.string('robust'),
    electronQuality = cms.string('tight'),
    electronVersion = cms.string('V04'),
    etBinning = cms.bool(True),
    reducedBarrelRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    reducedEndcapRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    src = cms.InputTag("gedGsfElectrons"),
    verticesCollection = cms.InputTag("offlinePrimaryVerticesWithBS"),
    robusttightEleIDCutsV04 = cms.PSet(
        barrel = cms.vdouble(
            0.0201, 0.0102, 0.0211, 0.00606, -1,
            -1, 2.34, 3.24, 4.51, 9999.0,
            9999.0, 9999.0, 9999.0, 9999.0, 9999.0,
            9999.0, 9999.0, 9999.0, 0.0, -9999.0,
            9999.0, 9999.0, 9999, -1, 0,
            0
        ),
        endcap = cms.vdouble(
            0.00253, 0.0291, 0.022, 0.0032, -1,
            -1, 0.826, 2.7, 0.255, 9999.0,
            9999.0, 9999.0, 9999.0, 9999.0, 9999.0,
            9999.0, 9999.0, 9999.0, 0.0, -9999.0,
            9999.0, 9999.0, 9999, -1, 0,
            0
        )
    ),
 )
process.moduleToTest(process.toTest)
)_"};

    edm::test::TestProcessor::Config config{baseConfig};
    SECTION("base configuration is OK") { REQUIRE_NOTHROW(edm::test::TestProcessor(config)); }

    SECTION("Event with dummy data") {
      auto electronToken = config.produces<std::vector<reco::GsfElectron>>("gedGsfElectrons");
      auto vertexToken = config.produces<std::vector<reco::Vertex>>("offlinePrimaryVerticesWithBS");
      {
        edm::test::TestProcessor tester(config);

        using namespace reco;
        reco::SuperCluster cluster;
        std::vector<reco::SuperCluster> clusters(1, cluster);

        TrackExtra extra;
        std::vector<TrackExtra> extras(1, extra);
        GsfTrack track;
        track.setExtra(TrackExtraRef(&extras, 0));
        std::vector<GsfTrack> tracks(1, track);
        GsfElectronCore core(GsfTrackRef(&tracks, 0));
        core.setSuperCluster(SuperClusterRef(&clusters, 0));
        std::vector<GsfElectronCore> cores(1, core);
        reco::GsfElectron electron(1,
                                   GsfElectron::ChargeInfo(),
                                   GsfElectronCoreRef(&cores, 0),
                                   GsfElectron::TrackClusterMatching(),
                                   GsfElectron::TrackExtrapolations(),
                                   GsfElectron::ClosestCtfTrack(),
                                   GsfElectron::FiducialFlags(),
                                   GsfElectron::ShowerShape(),
                                   GsfElectron::ConversionRejection());
        std::vector<reco::GsfElectron> electrons(1, electron);

        reco::Vertex vertex;
        std::vector<reco::Vertex> vertices(1, vertex);

        REQUIRE_NOTHROW(tester.test(std::make_pair(electronToken, std::make_unique<decltype(electrons)>(electrons)),
                                    std::make_pair(vertexToken, std::make_unique<decltype(vertices)>(vertices))));
      }
    }
  }
}

//Add additional TEST_CASEs to exercise the modules capabilities
