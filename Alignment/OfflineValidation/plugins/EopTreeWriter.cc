// -*- C++ -*-
//
// Package:    EopTreeWriter
// Class:      EopTreeWriter
//
/**\class EopTreeWriter EopTreeWriter.cc Alignment/OfflineValidation/plugins/EopTreeWriter.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Holger Enderle
//         Created:  Thu Dec  4 11:22:48 CET 2008
// $Id: EopTreeWriter.cc,v 1.2 2011/11/30 07:45:28 mussgill Exp $
//
//

// framework include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

// user include files
#include "Alignment/OfflineValidation/interface/EopVariables.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "CommonTools/Utils/interface/TFileDirectory.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDefs.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/DTRecHit/interface/DTRecHitCollection.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidate.h"
#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidateFwd.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/RecoCandidate/interface/RecoCaloTowerCandidate.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"
#include "TrackingTools/GeomPropagators/interface/PropagationExceptions.h"
#include "TrackingTools/TrackAssociator/interface/TrackAssociatorParameters.h"
#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"

// ROOT includes
#include "TH1.h"
#include "TTree.h"

//
// class decleration
//

class EopTreeWriter : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit EopTreeWriter(const edm::ParameterSet&);
  ~EopTreeWriter() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;
  double getDistInCM(double eta1, double phi1, double eta2, double phi2);

  // ----------member data ---------------------------
  edm::InputTag src_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geometryToken_;

  edm::Service<TFileService> fs_;
  TTree* tree_;
  EopVariables* treeMemPtr_;
  TrackDetectorAssociator trackAssociator_;
  TrackAssociatorParameters parameters_;

  edm::EDGetTokenT<reco::TrackCollection> theTrackCollectionToken_;
  edm::EDGetTokenT<reco::IsolatedPixelTrackCandidateCollection> isoPixelTkToken_;
  edm::EDGetTokenT<EcalRecHitCollection> ecalRecHitTokenAlCaToken_;
  edm::EDGetTokenT<EcalRecHitCollection> ecalRecHitTokenEBRecoToken_;
  edm::EDGetTokenT<EcalRecHitCollection> ecalRecHitTokenEERecoToken_;
};

//
// constructors and destructor
//
EopTreeWriter::EopTreeWriter(const edm::ParameterSet& iConfig)
    : src_(iConfig.getParameter<edm::InputTag>("src")), geometryToken_(esConsumes()) {
  usesResource(TFileService::kSharedResource);
  //now do what ever initialization is needed

  // TrackAssociator parameters
  edm::ParameterSet parameters = iConfig.getParameter<edm::ParameterSet>("TrackAssociatorParameters");
  edm::ConsumesCollector iC = consumesCollector();
  parameters_.loadParameters(parameters, iC);

  tree_ = fs_->make<TTree>("EopTree", "EopTree");
  treeMemPtr_ = new EopVariables;
  tree_->Branch("EopVariables", &treeMemPtr_);  // address of pointer!

  // do the consumes
  theTrackCollectionToken_ = consumes<reco::TrackCollection>(src_);
  isoPixelTkToken_ =
      consumes<reco::IsolatedPixelTrackCandidateCollection>(edm::InputTag("IsoProd", "HcalIsolatedTrackCollection"));

  ecalRecHitTokenAlCaToken_ = consumes<EcalRecHitCollection>(edm::InputTag("IsoProd", "IsoTrackEcalRecHitCollection"));
  ecalRecHitTokenEBRecoToken_ = consumes<EcalRecHitCollection>(edm::InputTag("ecalRecHit", "EcalRecHitsEB"));
  ecalRecHitTokenEERecoToken_ = consumes<EcalRecHitCollection>(edm::InputTag("ecalRecHit", "EcalRecHitsEE"));
}

//
// member functions
//

// ------------ method called to for each event  ------------
void EopTreeWriter::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  // get geometry
  const CaloGeometry* geo = &iSetup.getData(geometryToken_);

  // temporary collection of EB+EE recHits
  std::unique_ptr<EcalRecHitCollection> tmpEcalRecHitCollection(new EcalRecHitCollection);
  bool ecalInAlca = iEvent.getHandle(ecalRecHitTokenAlCaToken_).isValid();
  bool ecalInReco =
      iEvent.getHandle(ecalRecHitTokenEBRecoToken_) && iEvent.getHandle(ecalRecHitTokenEERecoToken_).isValid();

  std::vector<edm::EDGetTokenT<EcalRecHitCollection> > ecalTokens_;
  if (ecalInAlca)
    ecalTokens_.push_back(ecalRecHitTokenAlCaToken_);
  else if (ecalInReco) {
    ecalTokens_.push_back(ecalRecHitTokenEBRecoToken_);
    ecalTokens_.push_back(ecalRecHitTokenEERecoToken_);
  } else {
    throw cms::Exception("MissingProduct", "can not find EcalRecHits");
  }

  for (const auto& i : ecalTokens_) {
    edm::Handle<EcalRecHitCollection> ec = iEvent.getHandle(i);
    for (EcalRecHitCollection::const_iterator recHit = (*ec).begin(); recHit != (*ec).end(); ++recHit) {
      tmpEcalRecHitCollection->push_back(*recHit);
    }
  }

  const auto& tracks = iEvent.get(theTrackCollectionToken_);

  edm::Handle<reco::IsolatedPixelTrackCandidateCollection> isoPixelTracks = iEvent.getHandle(isoPixelTkToken_);
  bool pixelInAlca = isoPixelTracks.isValid();

  double trackemc1;
  double trackemc3;
  double trackemc5;
  double trackhac1;
  double trackhac3;
  double trackhac5;
  double maxPNearby;
  double dist;
  double EnergyIn;
  double EnergyOut;

  parameters_.useMuon = false;

  if (pixelInAlca)
    if (isoPixelTracks->empty())
      return;

  for (const auto& track : tracks) {
    bool noChargedTracks = true;

    if (track.p() < 9.)
      continue;

    trackAssociator_.useDefaultPropagator();
    TrackDetMatchInfo info = trackAssociator_.associate(
        iEvent,
        iSetup,
        trackAssociator_.getFreeTrajectoryState(&iSetup.getData(parameters_.bFieldToken), track),
        parameters_);

    trackemc1 = info.nXnEnergy(TrackDetMatchInfo::EcalRecHits, 0);
    trackemc3 = info.nXnEnergy(TrackDetMatchInfo::EcalRecHits, 1);
    trackemc5 = info.nXnEnergy(TrackDetMatchInfo::EcalRecHits, 2);
    trackhac1 = info.nXnEnergy(TrackDetMatchInfo::HcalRecHits, 0);
    trackhac3 = info.nXnEnergy(TrackDetMatchInfo::HcalRecHits, 1);
    trackhac5 = info.nXnEnergy(TrackDetMatchInfo::HcalRecHits, 2);

    if (trackhac3 < 5.)
      continue;

    double etaecal = info.trkGlobPosAtEcal.eta();
    double phiecal = info.trkGlobPosAtEcal.phi();

    maxPNearby = -10;
    dist = 50;
    for (const auto& track1 : tracks) {
      if (&track == &track1) {
        continue;
      }
      TrackDetMatchInfo info1 = trackAssociator_.associate(iEvent, iSetup, track1, parameters_);
      double etaecal1 = info1.trkGlobPosAtEcal.eta();
      double phiecal1 = info1.trkGlobPosAtEcal.phi();

      if (etaecal1 == 0 && phiecal1 == 0)
        continue;

      double ecDist = getDistInCM(etaecal, phiecal, etaecal1, phiecal1);

      if (ecDist < 40.) {
        //calculate maximum P and sum P near seed track
        if (track1.p() > maxPNearby) {
          maxPNearby = track1.p();
          dist = ecDist;
        }

        //apply loose isolation criteria
        if (track1.p() > 5.) {
          noChargedTracks = false;
          break;
        }
      }
    }
    EnergyIn = 0;
    EnergyOut = 0;
    if (noChargedTracks) {
      for (std::vector<EcalRecHit>::const_iterator ehit = tmpEcalRecHitCollection->begin();
           ehit != tmpEcalRecHitCollection->end();
           ehit++) {
        ////////////////////// FIND ECAL CLUSTER ENERGY
        // R-scheme of ECAL CLUSTERIZATION
        const GlobalPoint& posH = geo->getPosition((*ehit).detid());
        double phihit = posH.phi();
        double etahit = posH.eta();

        double dHitCM = getDistInCM(etaecal, phiecal, etahit, phihit);

        if (dHitCM < 9.0) {
          EnergyIn += ehit->energy();
        }
        if (dHitCM > 15.0 && dHitCM < 35.0) {
          EnergyOut += ehit->energy();
        }
      }

      treeMemPtr_->fillVariables(track.charge(),
                                 track.innerOk(),
                                 track.outerRadius(),
                                 track.numberOfValidHits(),
                                 track.numberOfLostHits(),
                                 track.chi2(),
                                 track.normalizedChi2(),
                                 track.p(),
                                 track.pt(),
                                 track.ptError(),
                                 track.theta(),
                                 track.eta(),
                                 track.phi(),
                                 trackemc1,
                                 trackemc3,
                                 trackemc5,
                                 trackhac1,
                                 trackhac3,
                                 trackhac5,
                                 maxPNearby,
                                 dist,
                                 EnergyIn,
                                 EnergyOut);

      tree_->Fill();
    }
  }
}

// ------------ method called once each job just after ending the event loop  ------------
void EopTreeWriter::endJob() {
  delete treeMemPtr_;
  treeMemPtr_ = nullptr;
}

//*************************************************************
double EopTreeWriter::getDistInCM(double eta1, double phi1, double eta2, double phi2) {
  //*************************************************************

  static constexpr float EEBoundary = 1.479;  // psedo-rapidity
  static constexpr float EBRadius = 129;      // in cm
  static constexpr float EEIPdis = 317;       // in cm

  double deltaPhi = phi1 - phi2;
  while (deltaPhi > M_PI)
    deltaPhi -= 2 * M_PI;
  while (deltaPhi <= -M_PI)
    deltaPhi += 2 * M_PI;
  double dR;
  // double Rec;
  double theta1 = 2 * atan(exp(-eta1));
  double theta2 = 2 * atan(exp(-eta2));
  double cotantheta1;
  if (cos(theta1) == 0)
    cotantheta1 = 0;
  else
    cotantheta1 = 1 / tan(theta1);
  double cotantheta2;
  if (cos(theta2) == 0)
    cotantheta2 = 0;
  else
    cotantheta2 = 1 / tan(theta2);
  // if (fabs(eta1)<1.479) Rec=129; //radius of ECAL barrel
  // else Rec=317; //distance from IP to ECAL endcap
  //|vect| times tg of acos(scalar product)
  // dR=fabs((Rec/sin(theta1))*tan(acos(sin(theta1)*sin(theta2)*(sin(phi1)*sin(phi2)+cos(phi1)*cos(phi2))+cos(theta1)*cos(theta2))));

  if (fabs(eta1) < EEBoundary) {
    dR = EBRadius * sqrt((cotantheta1 - cotantheta2) * (cotantheta1 - cotantheta2) + deltaPhi * deltaPhi);
  } else {
    dR = EEIPdis *
         sqrt(tan(theta1) * tan(theta1) + tan(theta2) * tan(theta2) - 2 * tan(theta1) * tan(theta2) * cos(deltaPhi));
  }
  return dR;
}

//*************************************************************
void EopTreeWriter::fillDescriptions(edm::ConfigurationDescriptions& descriptions)
//*************************************************************
{
  edm::ParameterSetDescription desc;
  desc.setComment("Generate tree for Tracker Alignment E/p validation");
  desc.add<edm::InputTag>("src", edm::InputTag("generalTracks"));

  // track association (take defaults)
  edm::ParameterSetDescription psd0;
  TrackAssociatorParameters::fillPSetDescription(psd0);
  desc.add<edm::ParameterSetDescription>("TrackAssociatorParameters", psd0);

  descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(EopTreeWriter);
