//*****************************************************************************
// File:      EgammaRecHitExtractor.cc
// ----------------------------------------------------------------------------
// OrigAuth:  Matthias Mozer, adapted from EgammaHcalExtractor by S. Harper
// Institute: IIHE-VUB, RAL
//=============================================================================
//*****************************************************************************

#include "CommonTools/Utils/interface/StringToEnumValue.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "PhysicsTools/IsolationAlgos/interface/IsoDepositExtractor.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgoRcd.h"

#include <Math/VectorUtil.h>

#include <vector>
#include <functional>

namespace egammaisolation {

  class EgammaRecHitExtractor : public reco::isodeposit::IsoDepositExtractor {
  public:
    EgammaRecHitExtractor(const edm::ParameterSet& par, edm::ConsumesCollector&& iC) : EgammaRecHitExtractor(par, iC) {}
    EgammaRecHitExtractor(const edm::ParameterSet& par, edm::ConsumesCollector& iC);
    ~EgammaRecHitExtractor() override;
    void fillVetos(const edm::Event& ev, const edm::EventSetup& evSetup, const reco::TrackCollection& tracks) override {
    }
    reco::IsoDeposit deposit(const edm::Event& ev,
                             const edm::EventSetup& evSetup,
                             const reco::Track& track) const override {
      throw cms::Exception("Configuration Error")
          << "This extractor " << (typeid(this).name()) << " is not made for tracks";
    }
    reco::IsoDeposit deposit(const edm::Event& ev,
                             const edm::EventSetup& evSetup,
                             const reco::Candidate& c) const override;

  private:
    void collect(reco::IsoDeposit& deposit,
                 const reco::SuperClusterRef& sc,
                 const CaloSubdetectorGeometry* subdet,
                 const CaloGeometry* caloGeom,
                 const EcalRecHitCollection& hits,
                 //const EcalChannelStatus* chStatus,
                 const EcalSeverityLevelAlgo* sevLevel,
                 bool barrel) const;

    double etMin_;
    double energyMin_;
    double extRadius_;
    double intRadius_;
    double intStrip_;
    edm::InputTag barrelEcalHitsTag_;
    edm::InputTag endcapEcalHitsTag_;
    edm::EDGetTokenT<EcalRecHitCollection> barrelEcalHitsToken_;
    edm::EDGetTokenT<EcalRecHitCollection> endcapEcalHitsToken_;
    edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geometryToken_;
    edm::ESGetToken<EcalSeverityLevelAlgo, EcalSeverityLevelAlgoRcd> sevlvToken_;
    bool fakeNegativeDeposit_;
    bool tryBoth_;
    bool useEt_;
    bool vetoClustered_;
    bool sameTag_;
    //int   severityLevelCut_;
    //float severityRecHitThreshold_;
    //std::string spIdString_;
    //float spIdThreshold_;
    //EcalSeverityLevelAlgo::SpikeId spId_;
    //std::vector<int> v_chstatus_;
    std::vector<int> severitiesexclEB_;
    std::vector<int> severitiesexclEE_;
    std::vector<int> flagsexclEB_;
    std::vector<int> flagsexclEE_;
  };
}  // namespace egammaisolation

#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/IsolationAlgos/interface/IsoDepositExtractorFactory.h"
DEFINE_EDM_PLUGIN(IsoDepositExtractorFactory, egammaisolation::EgammaRecHitExtractor, "EgammaRecHitExtractor");

using namespace std;
using namespace egammaisolation;
using namespace reco::isodeposit;

EgammaRecHitExtractor::EgammaRecHitExtractor(const edm::ParameterSet& par, edm::ConsumesCollector& iC)
    : etMin_(par.getParameter<double>("etMin")),
      energyMin_(par.getParameter<double>("energyMin")),
      extRadius_(par.getParameter<double>("extRadius")),
      intRadius_(par.getParameter<double>("intRadius")),
      intStrip_(par.getParameter<double>("intStrip")),
      barrelEcalHitsTag_(par.getParameter<edm::InputTag>("barrelEcalHits")),
      endcapEcalHitsTag_(par.getParameter<edm::InputTag>("endcapEcalHits")),
      barrelEcalHitsToken_(iC.consumes<EcalRecHitCollection>(barrelEcalHitsTag_)),
      endcapEcalHitsToken_(iC.consumes<EcalRecHitCollection>(endcapEcalHitsTag_)),
      geometryToken_(iC.esConsumes()),
      sevlvToken_(iC.esConsumes()),
      fakeNegativeDeposit_(par.getParameter<bool>("subtractSuperClusterEnergy")),
      tryBoth_(par.getParameter<bool>("tryBoth")),
      vetoClustered_(par.getParameter<bool>("vetoClustered")),
      sameTag_(false)
//severityLevelCut_(par.getParameter<int>("severityLevelCut"))
//severityRecHitThreshold_(par.getParameter<double>("severityRecHitThreshold")),
//spIdString_(par.getParameter<std::string>("spikeIdString")),
//spIdThreshold_(par.getParameter<double>("spikeIdThreshold")),
{
  const std::vector<std::string> flagnamesEB = par.getParameter<std::vector<std::string> >("RecHitFlagToBeExcludedEB");

  const std::vector<std::string> flagnamesEE = par.getParameter<std::vector<std::string> >("RecHitFlagToBeExcludedEE");

  flagsexclEB_ = StringToEnumValue<EcalRecHit::Flags>(flagnamesEB);

  flagsexclEE_ = StringToEnumValue<EcalRecHit::Flags>(flagnamesEE);

  const std::vector<std::string> severitynamesEB =
      par.getParameter<std::vector<std::string> >("RecHitSeverityToBeExcludedEB");

  severitiesexclEB_ = StringToEnumValue<EcalSeverityLevel::SeverityLevel>(severitynamesEB);

  const std::vector<std::string> severitynamesEE =
      par.getParameter<std::vector<std::string> >("RecHitSeverityToBeExcludedEE");

  severitiesexclEE_ = StringToEnumValue<EcalSeverityLevel::SeverityLevel>(severitynamesEE);

  if ((intRadius_ != 0.0) && (fakeNegativeDeposit_)) {
    throw cms::Exception("Configuration Error") << "EgammaRecHitExtractor: "
                                                << "If you use 'subtractSuperClusterEnergy', you *must* set "
                                                   "'intRadius' to ZERO; it does not make sense, otherwise.";
  }
  std::string isoVariable = par.getParameter<std::string>("isolationVariable");
  if (isoVariable == "et") {
    useEt_ = true;
  } else if (isoVariable == "energy") {
    useEt_ = false;
  } else {
    throw cms::Exception("Configuration Error")
        << "EgammaRecHitExtractor: isolationVariable '" << isoVariable << "' not known. "
        << " Supported values are 'et', 'energy'. ";
  }
  if (endcapEcalHitsTag_.encode() == barrelEcalHitsTag_.encode()) {
    sameTag_ = true;
    if (tryBoth_) {
      edm::LogWarning("EgammaRecHitExtractor")
          << "If you have configured 'barrelRecHits' == 'endcapRecHits', so I'm switching 'tryBoth' to FALSE.";
      tryBoth_ = false;
    }
  }
}

EgammaRecHitExtractor::~EgammaRecHitExtractor() {}

reco::IsoDeposit EgammaRecHitExtractor::deposit(const edm::Event& iEvent,
                                                const edm::EventSetup& iSetup,
                                                const reco::Candidate& emObject) const {
  //Get the channel status from the db
  //edm::ESHandle<EcalChannelStatus> chStatus;
  //iSetup.get<EcalChannelStatusRcd>().get(chStatus);

  const EcalSeverityLevelAlgo* sevLevel = &iSetup.getData(sevlvToken_);

  const CaloGeometry* caloGeom = &iSetup.getData(geometryToken_);
  const CaloSubdetectorGeometry* barrelgeom = caloGeom->getSubdetectorGeometry(DetId::Ecal, EcalBarrel);
  const CaloSubdetectorGeometry* endcapgeom = caloGeom->getSubdetectorGeometry(DetId::Ecal, EcalEndcap);

  static const std::string metname = "EgammaIsolationAlgos|EgammaRecHitExtractor";

  //define isodeposit starting from candidate
  reco::SuperClusterRef sc = emObject.get<reco::SuperClusterRef>();
  math::XYZPoint caloPosition = sc->position();

  Direction candDir(caloPosition.eta(), caloPosition.phi());
  reco::IsoDeposit deposit(candDir);
  deposit.setVeto(reco::IsoDeposit::Veto(candDir, intRadius_));
  double sinTheta = sin(2 * atan(exp(-sc->eta())));
  deposit.addCandEnergy(sc->energy() * (useEt_ ? sinTheta : 1.0));

  // subtract supercluster if desired
  double fakeEnergy = -sc->rawEnergy();
  if (fakeNegativeDeposit_) {
    deposit.addDeposit(candDir, fakeEnergy * (useEt_ ? sinTheta : 1.0));  // not exactly clean...
  }

  // fill rechits
  bool inBarrel = sameTag_ || (abs(sc->eta()) < 1.479);  //check for barrel. If only one collection is used, use barrel
  if (inBarrel || tryBoth_) {
    collect(deposit, sc, barrelgeom, caloGeom, iEvent.get(barrelEcalHitsToken_), sevLevel, true);
  }

  if ((!inBarrel) || tryBoth_) {
    collect(deposit, sc, endcapgeom, caloGeom, iEvent.get(endcapEcalHitsToken_), sevLevel, false);
  }

  return deposit;
}

void EgammaRecHitExtractor::collect(reco::IsoDeposit& deposit,
                                    const reco::SuperClusterRef& sc,
                                    const CaloSubdetectorGeometry* subdet,
                                    const CaloGeometry* caloGeom,
                                    const EcalRecHitCollection& hits,
                                    //const EcalChannelStatus* chStatus,
                                    const EcalSeverityLevelAlgo* sevLevel,
                                    bool barrel) const {
  GlobalPoint caloPosition(sc->position().x(), sc->position().y(), sc->position().z());
  CaloSubdetectorGeometry::DetIdSet chosen = subdet->getCells(caloPosition, extRadius_);
  EcalRecHitCollection::const_iterator j = hits.end();
  double caloeta = caloPosition.eta();
  double calophi = caloPosition.phi();
  double r2 = intRadius_ * intRadius_;

  std::vector<std::pair<DetId, float> >::const_iterator rhIt;

  for (CaloSubdetectorGeometry::DetIdSet::const_iterator i = chosen.begin(), end = chosen.end(); i != end; ++i) {
    j = hits.find(*i);
    if (j != hits.end()) {
      const GlobalPoint& position = caloGeom->getPosition(*i);
      double eta = position.eta();
      double phi = position.phi();
      double energy = j->energy();
      double et = energy * position.perp() / position.mag();
      double phiDiff = reco::deltaPhi(phi, calophi);

      //check if we are supposed to veto clustered and then do so
      if (vetoClustered_) {
        //Loop over basic clusters:
        bool isClustered = false;
        for (auto bcIt = sc->clustersBegin(); bcIt != sc->clustersEnd(); ++bcIt) {
          for (rhIt = (*bcIt)->hitsAndFractions().begin(); rhIt != (*bcIt)->hitsAndFractions().end(); ++rhIt) {
            if (rhIt->first == *i)
              isClustered = true;
            if (isClustered)
              break;
          }
          if (isClustered)
            break;
        }  //end loop over basic clusters

        if (isClustered)
          continue;
      }  //end if removeClustered

      std::vector<int>::const_iterator sit;
      int severityFlag = sevLevel->severityLevel(j->detid(), hits);
      if (barrel) {
        sit = std::find(severitiesexclEB_.begin(), severitiesexclEB_.end(), severityFlag);
        if (sit != severitiesexclEB_.end())
          continue;
      } else {
        sit = std::find(severitiesexclEE_.begin(), severitiesexclEE_.end(), severityFlag);
        if (sit != severitiesexclEE_.end())
          continue;
      }

      if (barrel) {
        // new rechit flag checks
        if (!j->checkFlag(EcalRecHit::kGood)) {
          if (j->checkFlags(flagsexclEB_)) {
            continue;
          }
        }
      } else {
        // new rechit flag checks
        if (!j->checkFlag(EcalRecHit::kGood)) {
          if (j->checkFlags(flagsexclEE_)) {
            continue;
          }
        }
      }

      if (et > etMin_ && energy > energyMin_  //Changed to fabs - then changed back to energy
          && fabs(eta - caloeta) > intStrip_ && (eta - caloeta) * (eta - caloeta) + phiDiff * phiDiff > r2) {
        deposit.addDeposit(Direction(eta, phi), (useEt_ ? et : energy));
      }
    }
  }
}
