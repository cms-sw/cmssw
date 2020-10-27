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
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

// user include files
#include <DataFormats/TrackReco/interface/Track.h>
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include <TMath.h>
#include <TH1.h>
#include "TTree.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "CommonTools/Utils/interface/TFileDirectory.h"

#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"
#include "TrackingTools/TrackAssociator/interface/TrackAssociatorParameters.h"
#include "TrackingTools/GeomPropagators/interface/PropagationExceptions.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDefs.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidateFwd.h"
#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoCaloTowerCandidate.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/DTRecHit/interface/DTRecHitCollection.h"

#include "Alignment/OfflineValidation/interface/EopVariables.h"

//
// class decleration
//

class EopTreeWriter : public edm::EDAnalyzer {
public:
  explicit EopTreeWriter(const edm::ParameterSet&);
  ~EopTreeWriter() override;

private:
  void beginJob() override;
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
EopTreeWriter::EopTreeWriter(const edm::ParameterSet& iConfig)
    : src_(iConfig.getParameter<edm::InputTag>("src")), geometryToken_(esConsumes()) {
  //now do what ever initialization is needed

  // TrackAssociator parameters
  edm::ParameterSet parameters = iConfig.getParameter<edm::ParameterSet>("TrackAssociatorParameters");
  edm::ConsumesCollector iC = consumesCollector();
  parameters_.loadParameters(parameters, iC);

  tree_ = fs_->make<TTree>("EopTree", "EopTree");
  treeMemPtr_ = new EopVariables;
  tree_->Branch("EopVariables", &treeMemPtr_);  // address of pointer!
}

EopTreeWriter::~EopTreeWriter() {
  // do anything here that needs to be done at destruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to for each event  ------------
void EopTreeWriter::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  // get geometry
  const CaloGeometry* geo = &iSetup.getData(geometryToken_);
  //    const CaloSubdetectorGeometry* towerGeometry =
  //      geo->getSubdetectorGeometry(DetId::Calo, CaloTowerDetId::SubdetId);

  // temporary collection of EB+EE recHits
  std::unique_ptr<EcalRecHitCollection> tmpEcalRecHitCollection(new EcalRecHitCollection);
  std::vector<edm::InputTag> ecalLabels_;

  edm::Handle<EcalRecHitCollection> tmpEc;
  bool ecalInAlca = iEvent.getByLabel(edm::InputTag("IsoProd", "IsoTrackEcalRecHitCollection"), tmpEc);
  bool ecalInReco = iEvent.getByLabel(edm::InputTag("ecalRecHit", "EcalRecHitsEB"), tmpEc) &&
                    iEvent.getByLabel(edm::InputTag("ecalRecHit", "EcalRecHitsEE"), tmpEc);
  if (ecalInAlca)
    ecalLabels_.push_back(edm::InputTag("IsoProd", "IsoTrackEcalRecHitCollection"));
  else if (ecalInReco) {
    ecalLabels_.push_back(edm::InputTag("ecalRecHit", "EcalRecHitsEB"));
    ecalLabels_.push_back(edm::InputTag("ecalRecHit", "EcalRecHitsEE"));
  } else
    throw cms::Exception("MissingProduct", "can not find EcalRecHits");

  std::vector<edm::InputTag>::const_iterator i;
  for (i = ecalLabels_.begin(); i != ecalLabels_.end(); i++) {
    edm::Handle<EcalRecHitCollection> ec;
    iEvent.getByLabel(*i, ec);
    for (EcalRecHitCollection::const_iterator recHit = (*ec).begin(); recHit != (*ec).end(); ++recHit) {
      tmpEcalRecHitCollection->push_back(*recHit);
    }
  }

  edm::Handle<reco::TrackCollection> tracks;
  iEvent.getByLabel(src_, tracks);

  edm::Handle<reco::IsolatedPixelTrackCandidateCollection> isoPixelTracks;
  edm::Handle<reco::IsolatedPixelTrackCandidateCollection> tmpPix;
  bool pixelInAlca = iEvent.getByLabel(edm::InputTag("IsoProd", "HcalIsolatedTrackCollection"), tmpPix);
  if (pixelInAlca)
    iEvent.getByLabel(edm::InputTag("IsoProd", "HcalIsolatedTrackCollection"), isoPixelTracks);

  Double_t trackemc1;
  Double_t trackemc3;
  Double_t trackemc5;
  Double_t trackhac1;
  Double_t trackhac3;
  Double_t trackhac5;
  Double_t maxPNearby;
  Double_t dist;
  Double_t EnergyIn;
  Double_t EnergyOut;

  parameters_.useMuon = false;

  if (pixelInAlca)
    if (isoPixelTracks->empty())
      return;

  for (reco::TrackCollection::const_iterator track = tracks->begin(); track != tracks->end(); ++track) {
    bool noChargedTracks = true;

    if (track->p() < 9.)
      continue;

    trackAssociator_.useDefaultPropagator();
    TrackDetMatchInfo info = trackAssociator_.associate(
        iEvent,
        iSetup,
        trackAssociator_.getFreeTrajectoryState(&iSetup.getData(parameters_.bFieldToken), *track),
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
    for (reco::TrackCollection::const_iterator track1 = tracks->begin(); track1 != tracks->end(); track1++) {
      if (track == track1)
        continue;
      TrackDetMatchInfo info1 = trackAssociator_.associate(iEvent, iSetup, *track1, parameters_);
      double etaecal1 = info1.trkGlobPosAtEcal.eta();
      double phiecal1 = info1.trkGlobPosAtEcal.phi();

      if (etaecal1 == 0 && phiecal1 == 0)
        continue;

      double ecDist = getDistInCM(etaecal, phiecal, etaecal1, phiecal1);

      if (ecDist < 40.) {
        //calculate maximum P and sum P near seed track
        if (track1->p() > maxPNearby) {
          maxPNearby = track1->p();
          dist = ecDist;
        }

        //apply loose isolation criteria
        if (track1->p() > 5.) {
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

      treeMemPtr_->fillVariables(track->charge(),
                                 track->innerOk(),
                                 track->outerRadius(),
                                 track->numberOfValidHits(),
                                 track->numberOfLostHits(),
                                 track->chi2(),
                                 track->normalizedChi2(),
                                 track->p(),
                                 track->pt(),
                                 track->ptError(),
                                 track->theta(),
                                 track->eta(),
                                 track->phi(),
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

// ------------ method called once each job just before starting event loop  ------------
void EopTreeWriter::beginJob() {}

// ------------ method called once each job just after ending the event loop  ------------
void EopTreeWriter::endJob() {
  delete treeMemPtr_;
  treeMemPtr_ = nullptr;
}

double EopTreeWriter::getDistInCM(double eta1, double phi1, double eta2, double phi2) {
  double deltaPhi = phi1 - phi2;
  while (deltaPhi > TMath::Pi())
    deltaPhi -= 2 * TMath::Pi();
  while (deltaPhi <= -TMath::Pi())
    deltaPhi += 2 * TMath::Pi();
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
  if (fabs(eta1) < 1.479)
    dR = 129 * sqrt((cotantheta1 - cotantheta2) * (cotantheta1 - cotantheta2) + deltaPhi * deltaPhi);
  else
    dR = 317 *
         sqrt(tan(theta1) * tan(theta1) + tan(theta2) * tan(theta2) - 2 * tan(theta1) * tan(theta2) * cos(deltaPhi));
  return dR;
}

//define this as a plug-in
DEFINE_FWK_MODULE(EopTreeWriter);
