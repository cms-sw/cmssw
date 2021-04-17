// -*- C++ -*-
//
// Package:    METProducers
// Class:      MuonMETValueMapProducer
//
//
// Original Author:  Puneeth Kalavase
//         Created:  Sun Mar 15 11:33:20 CDT 2009
//

/*
   The meanings ofr reco::MuonMETCorrectionData::Type
   NotUsed = 0:
     The muon is not used to correct the MET by default

   CombinedTrackUsed = 1, GlobalTrackUsed = 1:

     The muon is used to correct the MET. The Global pt is used. For
     backward compatibility only

   InnerTrackUsed = 2, TrackUsed = 2:
     The muon is used to correct the MET. The tracker pt is used. For
     backward compatibility only

   OuterTrackUsed = 3, StandAloneTrackUsed = 3:
     The muon is used to correct the MET. The standalone pt is used.
     For backward compatibility only. In general, the flag should
     never be 3. You do not want to correct the MET using the pt
     measurement from the standalone system (unless you really know
     what you're doing.

   TreatedAsPion = 4:
     The muon was treated as a Pion. This is used for the tcMET
     producer

   MuonP4V4QUsed = 5, MuonCandidateValuesUsed = 5:
     The default fit is used, i.e, we get the pt from muon->pt

   (see DataFormats/MuonReco/interface/MuonMETCorrectionData.h)
*/

//____________________________________________________________________________||
#include "RecoMET/METProducers/interface/MuonMETValueMapProducer.h"

#include "RecoMET/METAlgorithms/interface/MuonMETAlgo.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//____________________________________________________________________________||
namespace cms {

  MuonMETValueMapProducer::MuonMETValueMapProducer(const edm::ParameterSet& iConfig)
      : minPt_(iConfig.getParameter<double>("minPt")),
        maxEta_(iConfig.getParameter<double>("maxEta")),
        isAlsoTkMu_(iConfig.getParameter<bool>("isAlsoTkMu")),
        maxNormChi2_(iConfig.getParameter<double>("maxNormChi2")),
        maxd0_(iConfig.getParameter<double>("maxd0")),
        minnHits_(iConfig.getParameter<int>("minnHits")),
        minnValidStaHits_(iConfig.getParameter<int>("minnValidStaHits")),
        useTrackAssociatorPositions_(iConfig.getParameter<bool>("useTrackAssociatorPositions")),
        useHO_(iConfig.getParameter<bool>("useHO")),
        towerEtThreshold_(iConfig.getParameter<double>("towerEtThreshold")),
        useRecHits_(iConfig.getParameter<bool>("useRecHits")) {
    muonToken_ = consumes<edm::View<reco::Muon>>(iConfig.getParameter<edm::InputTag>("muonInputTag"));
    beamSpotToken_ = consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamSpotInputTag"));
    magFieldToken_ = esConsumes<MagneticField, IdealMagneticFieldRecord>();

    edm::ParameterSet trackAssociatorParams = iConfig.getParameter<edm::ParameterSet>("TrackAssociatorParameters");
    edm::ConsumesCollector iC = consumesCollector();
    trackAssociatorParameters_.loadParameters(trackAssociatorParams, iC);
    trackAssociator_.useDefaultPropagator();

    produces<edm::ValueMap<reco::MuonMETCorrectionData>>("muCorrData");
  }

  //____________________________________________________________________________||
  void MuonMETValueMapProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
    edm::Handle<edm::View<reco::Muon>> muons;
    iEvent.getByToken(muonToken_, muons);

    edm::Handle<reco::BeamSpot> beamSpot;
    iEvent.getByToken(beamSpotToken_, beamSpot);

    const MagneticField& magneticField = iSetup.getData(magFieldToken_);

    double bfield = magneticField.inTesla(GlobalPoint(0., 0., 0.)).z();

    std::vector<reco::MuonMETCorrectionData> muCorrDataList;

    for (edm::View<reco::Muon>::const_iterator muon = muons->begin(); muon != muons->end(); ++muon) {
      double deltax = 0.0;
      double deltay = 0.0;
      determine_deltax_deltay(deltax, deltay, *muon, bfield, magneticField, iEvent, iSetup);

      reco::MuonMETCorrectionData::Type muCorrType = decide_correction_type(*muon, beamSpot->position());

      reco::MuonMETCorrectionData muMETCorrData(muCorrType, deltax, deltay);
      muCorrDataList.push_back(muMETCorrData);
    }

    auto valueMapMuCorrData = std::make_unique<edm::ValueMap<reco::MuonMETCorrectionData>>();

    edm::ValueMap<reco::MuonMETCorrectionData>::Filler dataFiller(*valueMapMuCorrData);

    dataFiller.insert(muons, muCorrDataList.begin(), muCorrDataList.end());
    dataFiller.fill();

    iEvent.put(std::move(valueMapMuCorrData), "muCorrData");
  }

  //____________________________________________________________________________||
  void MuonMETValueMapProducer::determine_deltax_deltay(double& deltax,
                                                        double& deltay,
                                                        const reco::Muon& muon,
                                                        double bfield,
                                                        const MagneticField& magneticField,
                                                        edm::Event& iEvent,
                                                        const edm::EventSetup& iSetup) {
    reco::TrackRef mu_track;
    if (muon.isGlobalMuon())
      mu_track = muon.globalTrack();
    else if (muon.isTrackerMuon() || muon.isRPCMuon() || muon.isGEMMuon() || muon.isME0Muon())
      mu_track = muon.innerTrack();
    else
      mu_track = muon.outerTrack();

    TrackDetMatchInfo info = trackAssociator_.associate(
        iEvent, iSetup, trackAssociator_.getFreeTrajectoryState(&magneticField, *mu_track), trackAssociatorParameters_);

    MuonMETAlgo alg;
    alg.GetMuDepDeltas(
        &muon, info, useTrackAssociatorPositions_, useRecHits_, useHO_, towerEtThreshold_, deltax, deltay, bfield);
  }

  //____________________________________________________________________________||
  reco::MuonMETCorrectionData::Type MuonMETValueMapProducer::decide_correction_type(
      const reco::Muon& muon, const math::XYZPoint& beamSpotPosition) {
    if (should_type_MuonCandidateValuesUsed(muon, beamSpotPosition))
      return reco::MuonMETCorrectionData::Type::MuonCandidateValuesUsed;

    return reco::MuonMETCorrectionData::Type::NotUsed;
  }

  //____________________________________________________________________________||
  bool MuonMETValueMapProducer::should_type_MuonCandidateValuesUsed(const reco::Muon& muon,
                                                                    const math::XYZPoint& beamSpotPosition) {
    if (!muon.isGlobalMuon())
      return false;
    if (!muon.isTrackerMuon() && isAlsoTkMu_)
      return false;
    reco::TrackRef globTk = muon.globalTrack();
    reco::TrackRef siTk = muon.innerTrack();

    if (muon.pt() < minPt_ || fabs(muon.eta()) > maxEta_)
      return false;
    if (globTk->chi2() / globTk->ndof() > maxNormChi2_)
      return false;
    if (fabs(globTk->dxy(beamSpotPosition)) > fabs(maxd0_))
      return false;
    if (siTk->numberOfValidHits() < minnHits_)
      return false;
    if (globTk->hitPattern().numberOfValidMuonHits() < minnValidStaHits_)
      return false;
    return true;
  }

  //____________________________________________________________________________||
}  // namespace cms

//____________________________________________________________________________||
