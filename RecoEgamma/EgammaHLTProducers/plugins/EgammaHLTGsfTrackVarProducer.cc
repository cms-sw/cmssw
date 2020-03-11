/** \class EgammaHLTGsfTrackVarProducer
 *
 *  \author Roberto Covarelli (CERN)
 * 
 * $Id: EgammaHLTGsfTrackVarProducer.cc,v 1.1 2012/01/23 12:56:38 sharper Exp $
 *
 */

//this class is designed to calculate dEtaIn,dPhiIn gsf track - supercluster pairs
//it can take as input std::vector<Electron> which the gsf track-sc is already done
//or it can run over the std::vector<GsfTrack> directly in which case it will pick the smallest dEta,dPhi
//the dEta, dPhi do not have to be from the same track
//it can optionally set dEta, dPhi to 0 based on the number of tracks found

#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronIsolationAssociation.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateIsolation.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronUtilities.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackingTools/GsfTools/interface/GsfPropagatorAdapter.h"
#include "TrackingTools/GsfTools/interface/MultiTrajectoryStateMode.h"
#include "TrackingTools/GsfTools/interface/MultiTrajectoryStateTransform.h"
#include "TrackingTools/PatternTools/interface/TransverseImpactPointExtrapolator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

class EgammaHLTGsfTrackVarProducer : public edm::global::EDProducer<> {
public:
  explicit EgammaHLTGsfTrackVarProducer(const edm::ParameterSet&);
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  const edm::EDGetTokenT<reco::RecoEcalCandidateCollection> recoEcalCandToken_;
  const edm::EDGetTokenT<reco::ElectronCollection> electronToken_;
  const edm::EDGetTokenT<reco::GsfTrackCollection> gsfTrackToken_;
  const edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;

  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magneticFieldToken_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerGeometryToken_;

  const int upperTrackNrToRemoveCut_;
  const int lowerTrackNrToRemoveCut_;
  const bool useDefaultValuesForBarrel_;
  const bool useDefaultValuesForEndcap_;

  const edm::EDPutTokenT<reco::RecoEcalCandidateIsolationMap> dEtaMapPutToken_;
  const edm::EDPutTokenT<reco::RecoEcalCandidateIsolationMap> dEtaSeedMapPutToken_;
  const edm::EDPutTokenT<reco::RecoEcalCandidateIsolationMap> dPhiMapPutToken_;
  const edm::EDPutTokenT<reco::RecoEcalCandidateIsolationMap> oneOverESuperMinusOneOverPMapPutToken_;
  const edm::EDPutTokenT<reco::RecoEcalCandidateIsolationMap> oneOverESeedMinusOneOverPMapPutToken_;
  const edm::EDPutTokenT<reco::RecoEcalCandidateIsolationMap> missingHitsMapPutToken_;
  const edm::EDPutTokenT<reco::RecoEcalCandidateIsolationMap> validHitsMapPutToken_;
  const edm::EDPutTokenT<reco::RecoEcalCandidateIsolationMap> chi2MapPutToken_;
};

EgammaHLTGsfTrackVarProducer::EgammaHLTGsfTrackVarProducer(const edm::ParameterSet& config)
    : recoEcalCandToken_(
          consumes<reco::RecoEcalCandidateCollection>(config.getParameter<edm::InputTag>("recoEcalCandidateProducer"))),
      electronToken_{consumes<reco::ElectronCollection>(config.getParameter<edm::InputTag>("inputCollection"))},
      gsfTrackToken_{consumes<reco::GsfTrackCollection>(config.getParameter<edm::InputTag>("inputCollection"))},
      beamSpotToken_{consumes<reco::BeamSpot>(config.getParameter<edm::InputTag>("beamSpotProducer"))},
      magneticFieldToken_{esConsumes<MagneticField, IdealMagneticFieldRecord>()},
      trackerGeometryToken_{esConsumes<TrackerGeometry, TrackerDigiGeometryRecord>()},
      upperTrackNrToRemoveCut_{config.getParameter<int>("upperTrackNrToRemoveCut")},
      lowerTrackNrToRemoveCut_{config.getParameter<int>("lowerTrackNrToRemoveCut")},
      useDefaultValuesForBarrel_{config.getParameter<bool>("useDefaultValuesForBarrel")},
      useDefaultValuesForEndcap_{config.getParameter<bool>("useDefaultValuesForEndcap")},
      dEtaMapPutToken_{produces<reco::RecoEcalCandidateIsolationMap>("Deta").setBranchAlias("deta")},
      dEtaSeedMapPutToken_{produces<reco::RecoEcalCandidateIsolationMap>("DetaSeed").setBranchAlias("detaseed")},
      dPhiMapPutToken_{produces<reco::RecoEcalCandidateIsolationMap>("Dphi").setBranchAlias("dphi")},
      oneOverESuperMinusOneOverPMapPutToken_{produces<reco::RecoEcalCandidateIsolationMap>("OneOESuperMinusOneOP")},
      oneOverESeedMinusOneOverPMapPutToken_{produces<reco::RecoEcalCandidateIsolationMap>("OneOESeedMinusOneOP")},
      missingHitsMapPutToken_{
          produces<reco::RecoEcalCandidateIsolationMap>("MissingHits").setBranchAlias("missinghits")},
      validHitsMapPutToken_{produces<reco::RecoEcalCandidateIsolationMap>("Chi2").setBranchAlias("chi2")},
      chi2MapPutToken_{produces<reco::RecoEcalCandidateIsolationMap>("ValidHits").setBranchAlias("validhits")} {}

void EgammaHLTGsfTrackVarProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>(("recoEcalCandidateProducer"), edm::InputTag("hltRecoEcalSuperClusterActivityCandidate"));
  desc.add<edm::InputTag>(("inputCollection"), edm::InputTag("hltActivityElectronGsfTracks"));
  desc.add<edm::InputTag>(("beamSpotProducer"), edm::InputTag("hltOnlineBeamSpot"));
  desc.add<int>(("upperTrackNrToRemoveCut"), 9999);
  desc.add<int>(("lowerTrackNrToRemoveCut"), -1);
  desc.add<bool>(("useDefaultValuesForBarrel"), false);
  desc.add<bool>(("useDefaultValuesForEndcap"), false);

  descriptions.add("hltEgammaHLTGsfTrackVarProducer", desc);
}
void EgammaHLTGsfTrackVarProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  // Get the HLT filtered objects
  auto recoEcalCandHandle = iEvent.getHandle(recoEcalCandToken_);

  auto const& beamSpotPosition = iEvent.get(beamSpotToken_).position();
  auto const& magneticField = iSetup.getData(magneticFieldToken_);
  auto const& trackerGeometry = iSetup.getData(trackerGeometryToken_);

  TransverseImpactPointExtrapolator extrapolator{
      GsfPropagatorAdapter{AnalyticalPropagator(&magneticField, anyDirection)}};

  reco::RecoEcalCandidateIsolationMap dEtaMap(recoEcalCandHandle);
  reco::RecoEcalCandidateIsolationMap dEtaSeedMap(recoEcalCandHandle);
  reco::RecoEcalCandidateIsolationMap dPhiMap(recoEcalCandHandle);
  reco::RecoEcalCandidateIsolationMap oneOverESuperMinusOneOverPMap(recoEcalCandHandle);
  reco::RecoEcalCandidateIsolationMap oneOverESeedMinusOneOverPMap(recoEcalCandHandle);
  reco::RecoEcalCandidateIsolationMap missingHitsMap(recoEcalCandHandle);
  reco::RecoEcalCandidateIsolationMap validHitsMap(recoEcalCandHandle);
  reco::RecoEcalCandidateIsolationMap chi2Map(recoEcalCandHandle);

  for (unsigned int iRecoEcalCand = 0; iRecoEcalCand < recoEcalCandHandle->size(); ++iRecoEcalCand) {
    reco::RecoEcalCandidateRef recoEcalCandRef(recoEcalCandHandle, iRecoEcalCand);

    const reco::SuperClusterRef scRef = recoEcalCandRef->superCluster();
    //the idea is that we can take the tracks from properly associated electrons or just take all gsf tracks with that sc as a seed
    std::vector<const reco::GsfTrack*> gsfTracks;
    if (auto electronHandle = iEvent.getHandle(electronToken_)) {
      for (auto const& ele : *electronHandle) {
        if (ele.superCluster() == scRef) {
          gsfTracks.push_back(ele.gsfTrack().get());
        }
      }
    } else {
      for (auto const& trk : iEvent.get(gsfTrackToken_)) {
        auto elseed = trk.extra()->seedRef().castTo<reco::ElectronSeedRef>();
        if (elseed->caloCluster().castTo<reco::SuperClusterRef>() == scRef) {
          gsfTracks.push_back(&trk);
        }
      }
    }

    int validHitsValue = 0;
    float chi2Value = 9999999.;
    float missingHitsValue = 9999999;
    float dEtaInValue = 999999;
    float dEtaSeedInValue = 999999;
    float dPhiInValue = 999999;
    float oneOverESuperMinusOneOverPValue = 999999;
    float oneOverESeedMinusOneOverPValue = 999999;

    const int nrTracks = gsfTracks.size();
    const bool rmCutsDueToNrTracks = nrTracks <= lowerTrackNrToRemoveCut_ || nrTracks >= upperTrackNrToRemoveCut_;
    //to use the default values, we require at least one track...
    const bool useDefaultValues = std::abs(recoEcalCandRef->eta()) < 1.479
                                      ? useDefaultValuesForBarrel_ && nrTracks >= 1
                                      : useDefaultValuesForEndcap_ && nrTracks >= 1;

    if (rmCutsDueToNrTracks || useDefaultValues) {
      dEtaInValue = 0;
      dEtaSeedInValue = 0;
      dPhiInValue = 0;
      missingHitsValue = 0;
      validHitsValue = 100;
      chi2Value = 0;
      oneOverESuperMinusOneOverPValue = 0;
      oneOverESeedMinusOneOverPValue = 0;
    } else {
      for (size_t trkNr = 0; trkNr < gsfTracks.size(); trkNr++) {
        GlobalPoint scPos(scRef->x(), scRef->y(), scRef->z());

        GlobalPoint trackExtrapToSC;
        {
          auto innTSOS =
              MultiTrajectoryStateTransform::innerStateOnSurface(*gsfTracks[trkNr], trackerGeometry, &magneticField);
          auto posTSOS = extrapolator.extrapolate(innTSOS, scPos);
          multiTrajectoryStateMode::positionFromModeCartesian(posTSOS, trackExtrapToSC);
        }

        EleRelPointPair scAtVtx(scRef->position(), trackExtrapToSC, beamSpotPosition);

        float trkP = gsfTracks[trkNr]->p();
        if (scRef->energy() != 0 && trkP != 0) {
          if (std::abs(1 / scRef->energy() - 1 / trkP) < oneOverESuperMinusOneOverPValue) {
            oneOverESuperMinusOneOverPValue = std::abs(1 / scRef->energy() - 1 / trkP);
          }
        }
        if (scRef->seed().isNonnull() && scRef->seed()->energy() != 0 && trkP != 0) {
          if (std::abs(1 / scRef->seed()->energy() - 1 / trkP) < oneOverESeedMinusOneOverPValue) {
            oneOverESeedMinusOneOverPValue = std::abs(1 / scRef->seed()->energy() - 1 / trkP);
          }
        }

        if (gsfTracks[trkNr]->hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS) < missingHitsValue) {
          missingHitsValue = gsfTracks[trkNr]->hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS);
        }

        if (gsfTracks[trkNr]->numberOfValidHits() < validHitsValue) {
          validHitsValue = gsfTracks[trkNr]->numberOfValidHits();
        }

        if (gsfTracks[trkNr]->numberOfValidHits() < chi2Value) {
          chi2Value = gsfTracks[trkNr]->normalizedChi2();
        }

        if (std::abs(scAtVtx.dEta()) < dEtaInValue) {
          //we are allowing them to come from different tracks
          dEtaInValue = std::abs(scAtVtx.dEta());
        }

        if (std::abs(scAtVtx.dEta()) < dEtaSeedInValue) {
          dEtaSeedInValue = std::abs(scAtVtx.dEta() - scRef->position().eta() + scRef->seed()->position().eta());
        }

        if (std::abs(scAtVtx.dPhi()) < dPhiInValue) {
          //we are allowing them to come from different tracks
          dPhiInValue = std::abs(scAtVtx.dPhi());
        }
      }
    }

    dEtaMap.insert(recoEcalCandRef, dEtaInValue);
    dEtaSeedMap.insert(recoEcalCandRef, dEtaSeedInValue);
    dPhiMap.insert(recoEcalCandRef, dPhiInValue);
    oneOverESuperMinusOneOverPMap.insert(recoEcalCandRef, oneOverESuperMinusOneOverPValue);
    oneOverESeedMinusOneOverPMap.insert(recoEcalCandRef, oneOverESeedMinusOneOverPValue);
    missingHitsMap.insert(recoEcalCandRef, missingHitsValue);
    validHitsMap.insert(recoEcalCandRef, validHitsValue);
    chi2Map.insert(recoEcalCandRef, chi2Value);
  }

  iEvent.emplace(dEtaMapPutToken_, dEtaMap);
  iEvent.emplace(dEtaSeedMapPutToken_, dEtaSeedMap);
  iEvent.emplace(dPhiMapPutToken_, dPhiMap);
  iEvent.emplace(oneOverESuperMinusOneOverPMapPutToken_, oneOverESuperMinusOneOverPMap);
  iEvent.emplace(oneOverESeedMinusOneOverPMapPutToken_, oneOverESeedMinusOneOverPMap);
  iEvent.emplace(missingHitsMapPutToken_, missingHitsMap);
  iEvent.emplace(validHitsMapPutToken_, validHitsMap);
  iEvent.emplace(chi2MapPutToken_, chi2Map);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(EgammaHLTGsfTrackVarProducer);
