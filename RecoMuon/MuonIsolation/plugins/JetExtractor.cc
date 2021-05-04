#include "JetExtractor.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TrackAssociator/interface/TrackAssociatorParameters.h"
#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"

#include "DataFormats/Math/interface/deltaR.h"

using namespace edm;
using namespace std;
using namespace reco;
using namespace muonisolation;
using reco::isodeposit::Direction;

JetExtractor::JetExtractor(const ParameterSet& par, edm::ConsumesCollector&& iC)
    : theJetCollectionToken(iC.consumes<CaloJetCollection>(par.getParameter<edm::InputTag>("JetCollectionLabel"))),
      thePropagatorName(par.getParameter<std::string>("PropagatorName")),
      theThreshold(par.getParameter<double>("Threshold")),
      theDR_Veto(par.getParameter<double>("DR_Veto")),
      theDR_Max(par.getParameter<double>("DR_Max")),
      theExcludeMuonVeto(par.getParameter<bool>("ExcludeMuonVeto")),
      theService(nullptr),
      theAssociator(nullptr),
      thePrintTimeReport(par.getUntrackedParameter<bool>("PrintTimeReport")) {
  ParameterSet serviceParameters = par.getParameter<ParameterSet>("ServiceParameters");
  theService = new MuonServiceProxy(serviceParameters, edm::ConsumesCollector(iC));

  //  theAssociatorParameters = new TrackAssociatorParameters(par.getParameter<edm::ParameterSet>("TrackAssociatorParameters"), iC_);
  theAssociatorParameters = new TrackAssociatorParameters();
  theAssociatorParameters->loadParameters(par.getParameter<edm::ParameterSet>("TrackAssociatorParameters"), iC);
  theAssociator = new TrackDetectorAssociator();
}

JetExtractor::~JetExtractor() {
  if (theAssociatorParameters)
    delete theAssociatorParameters;
  if (theService)
    delete theService;
  if (theAssociator)
    delete theAssociator;
}

void JetExtractor::fillVetos(const edm::Event& event, const edm::EventSetup& eventSetup, const TrackCollection& muons) {
  //   LogWarning("JetExtractor")
  //     <<"fillVetos does nothing now: IsoDeposit provides enough functionality\n"
  //     <<"to remove a deposit at/around given (eta, phi)";
}

IsoDeposit JetExtractor::deposit(const Event& event, const EventSetup& eventSetup, const Track& muon) const {
  theService->update(eventSetup);
  theAssociator->setPropagator(&*(theService->propagator(thePropagatorName)));

  typedef IsoDeposit::Veto Veto;
  IsoDeposit::Direction muonDir(muon.eta(), muon.phi());

  IsoDeposit depJet(muonDir);

  edm::ESHandle<MagneticField> bField;
  eventSetup.get<IdealMagneticFieldRecord>().get(bField);

  reco::TransientTrack tMuon(muon, &*bField);
  FreeTrajectoryState iFTS = tMuon.initialFreeState();
  TrackDetMatchInfo mInfo = theAssociator->associate(event, eventSetup, iFTS, *theAssociatorParameters);

  reco::isodeposit::Direction vetoDirection(mInfo.trkGlobPosAtHcal.eta(), mInfo.trkGlobPosAtHcal.phi());
  depJet.setVeto(Veto(vetoDirection, theDR_Veto));

  edm::Handle<CaloJetCollection> caloJetsH;
  event.getByToken(theJetCollectionToken, caloJetsH);

  //use calo towers
  CaloJetCollection::const_iterator jetCI = caloJetsH->begin();
  for (; jetCI != caloJetsH->end(); ++jetCI) {
    double deltar0 = reco::deltaR(muon, *jetCI);
    if (deltar0 > theDR_Max)
      continue;
    if (jetCI->et() < theThreshold)
      continue;

    //should I make a separate config option for this?
    std::vector<CaloTowerPtr> jetConstituents = jetCI->getCaloConstituents();

    std::vector<DetId>::const_iterator crossedCI = mInfo.crossedTowerIds.begin();
    std::vector<CaloTowerPtr>::const_iterator jetTowCI = jetConstituents.begin();

    double sumEtExcluded = 0;
    for (; jetTowCI != jetConstituents.end(); ++jetTowCI) {
      bool isExcluded = false;
      double deltaRLoc = reco::deltaR(vetoDirection, *jetCI);
      if (deltaRLoc < theDR_Veto) {
        isExcluded = true;
      }
      for (; !isExcluded && crossedCI != mInfo.crossedTowerIds.end(); ++crossedCI) {
        if (crossedCI->rawId() == (*jetTowCI)->id().rawId()) {
          isExcluded = true;
        }
      }
      if (isExcluded)
        sumEtExcluded += (*jetTowCI)->et();
    }
    if (theExcludeMuonVeto) {
      if (jetCI->et() - sumEtExcluded < theThreshold)
        continue;
    }

    double depositEt = jetCI->et();
    if (theExcludeMuonVeto)
      depositEt = depositEt - sumEtExcluded;

    reco::isodeposit::Direction jetDir(jetCI->eta(), jetCI->phi());
    depJet.addDeposit(jetDir, depositEt);
  }

  std::vector<const CaloTower*>::const_iterator crossedCI = mInfo.crossedTowers.begin();
  double muSumEt = 0;
  for (; crossedCI != mInfo.crossedTowers.end(); ++crossedCI) {
    muSumEt += (*crossedCI)->et();
  }
  depJet.addCandEnergy(muSumEt);

  return depJet;
}
