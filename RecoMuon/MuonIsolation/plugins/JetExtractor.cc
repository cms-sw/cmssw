#include "JetExtractor.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Utilities/Timing/interface/TimingReport.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/JetReco/interface/CaloJet.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TrackAssociator/interface/TrackAssociatorParameters.h"
#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"

using namespace edm;
using namespace std;
using namespace reco;
using namespace muonisolation;

JetExtractor::JetExtractor(const ParameterSet& par) :
  theJetCollectionLabel(par.getParameter<edm::InputTag>("JetCollectionLabel")),
  thePropagatorName(par.getParameter<std::string>("PropagatorName")),
  theThreshold(par.getParameter<double>("Threshold")),
  theDR_Veto(par.getParameter<double>("DR_Veto")),
  theDR_Max(par.getParameter<double>("DR_Max")),
  theExcludeMuonVeto(par.getParameter<bool>("ExcludeMuonVeto")),
  theAssociator(0),
  thePropagator(0),
  thePrintTimeReport(par.getUntrackedParameter<bool>("PrintTimeReport"))
{
  theAssociatorParameters = new TrackAssociatorParameters(par.getParameter<edm::ParameterSet>("TrackAssociatorParameters"));
  theAssociator = new TrackDetectorAssociator();
}

JetExtractor::~JetExtractor(){
  if (thePrintTimeReport) TimingReport::current()->dump(std::cout);
  if (theAssociatorParameters) delete theAssociatorParameters;
  if (theAssociator) delete theAssociator;
  if (thePropagator) delete thePropagator;
}

void JetExtractor::fillVetos(const edm::Event& event, const edm::EventSetup& eventSetup, const TrackCollection& muons)
{
//   LogWarning("JetExtractor")
//     <<"fillVetos does nothing now: MuIsoDeposit provides enough functionality\n"
//     <<"to remove a deposit at/around given (eta, phi)";

}

MuIsoDeposit JetExtractor::deposit( const Event & event, const EventSetup& eventSetup, const Track & muon) const
{
  if (thePropagator == 0){
    ESHandle<Propagator> prop;
    eventSetup.get<TrackingComponentsRecord>().get(thePropagatorName, prop);
    thePropagator = prop->clone();
    theAssociator->setPropagator(thePropagator);
  }

  typedef MuIsoDeposit::Veto Veto;
  MuIsoDeposit::Direction muonDir(muon.eta(), muon.phi());
  
  MuIsoDeposit depJet("", muonDir);

  edm::ESHandle<MagneticField> bField;
  eventSetup.get<IdealMagneticFieldRecord>().get(bField);


  reco::TransientTrack tMuon(muon, &*bField);
  FreeTrajectoryState iFTS = tMuon.initialFreeState();
  TrackDetMatchInfo mInfo = theAssociator->associate(event, eventSetup, iFTS, *theAssociatorParameters);

  Direction vetoDirection(mInfo.trkGlobPosAtHcal.eta(), mInfo.trkGlobPosAtHcal.phi());
  depJet.setVeto(Veto(vetoDirection, theDR_Veto));


  edm::Handle<CaloJetCollection> caloJetsH;
  event.getByLabel(theJetCollectionLabel, caloJetsH);

  //use calo towers    
  CaloJetCollection::const_iterator jetCI = caloJetsH->begin();
  for (; jetCI != caloJetsH->end(); ++jetCI){
    double deltar0 = deltaR(muon,*jetCI);
    if (deltar0>theDR_Max) continue;
    if (jetCI->et() < theThreshold ) continue;

    //should I make a separate config option for this?
    std::vector<CaloTowerRef> jetConstituents = jetCI->getConstituents();

    std::vector<DetId>::const_iterator crossedCI =  mInfo.crossedTowerIds.begin();
    std::vector<CaloTowerRef>::const_iterator jetTowCI = jetConstituents.begin();
    
    double sumEtExcluded = 0;
    for (;jetTowCI != jetConstituents.end(); ++ jetTowCI){
      bool isExcluded = false;
      double deltaRLoc = deltaR(vetoDirection, *jetCI);
      if (deltaRLoc < theDR_Veto){
	isExcluded = true;
      }
      for(; ! isExcluded && crossedCI != mInfo.crossedTowerIds.end(); ++crossedCI){
	if (crossedCI->rawId() == (*jetTowCI)->id().rawId()){
	  isExcluded = true;
	}
      }
      if (isExcluded) sumEtExcluded += (*jetTowCI)->et();
    }
    if (theExcludeMuonVeto){
      if (jetCI->et() - sumEtExcluded < theThreshold ) continue;
    }

    double depositEt = jetCI->et();
    if (theExcludeMuonVeto) depositEt = depositEt - sumEtExcluded;

    Direction jetDir(jetCI->eta(), jetCI->phi());
    depJet.addDeposit(jetDir, depositEt);
    
  }

  std::vector<CaloTower>::const_iterator crossedCI =  mInfo.crossedTowers.begin();
  double muSumEt = 0;
  for (; crossedCI != mInfo.crossedTowers.end(); ++crossedCI){
    muSumEt += crossedCI->et();
  }
  depJet.addMuonEnergy(muSumEt);

  return depJet;

}

double JetExtractor::PhiInRange(const double& phi) {
      double phiout = phi;

      if( phiout > 2*M_PI || phiout < -2*M_PI) {
            phiout = fmod( phiout, 2*M_PI);
      }
      if (phiout <= -M_PI) phiout += 2*M_PI;
      else if (phiout >  M_PI) phiout -= 2*M_PI;

      return phiout;
}

template <class T, class U>
double JetExtractor::deltaR(const T& t, const U& u) {
      return sqrt(pow(t.eta()-u.eta(),2) +pow(PhiInRange(t.phi()-u.phi()),2));
}

