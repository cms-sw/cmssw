#ifndef EgammaIsolationProducers_EgammaRecHitExtractor_h
#define EgammaIsolationProducers_EgammaRecHitExtractor_h
//*****************************************************************************
// File:      EgammaRecHitExtractor.h
// ----------------------------------------------------------------------------
// OrigAuth:  Matthias Mozer, adapted from EgammaHcalExtractor by S. Harper
// Institute: IIHE-VUB, RAL
//=============================================================================
//*****************************************************************************

//C++ includes
#include <vector>
#include <functional>

//CMSSW includes
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "PhysicsTools/IsolationAlgos/interface/IsoDepositExtractor.h"
#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"

#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgoRcd.h"

namespace egammaisolation {

  class EgammaRecHitExtractor : public reco::isodeposit::IsoDepositExtractor {
  public:
    EgammaRecHitExtractor(const edm::ParameterSet& par, edm::ConsumesCollector && iC) :
      EgammaRecHitExtractor(par, iC) {}
    EgammaRecHitExtractor(const edm::ParameterSet& par, edm::ConsumesCollector & iC);
    virtual ~EgammaRecHitExtractor() ;
    virtual void fillVetos(const edm::Event & ev, const edm::EventSetup & evSetup, const reco::TrackCollection & tracks) { }
    virtual reco::IsoDeposit deposit(const edm::Event & ev, const edm::EventSetup & evSetup, const reco::Track & track) const {
      throw cms::Exception("Configuration Error") << "This extractor " << (typeid(this).name()) << " is not made for tracks";
    }
    virtual reco::IsoDeposit deposit(const edm::Event & ev, const edm::EventSetup & evSetup, const reco::Candidate & c) const ;

  private:
    void collect(reco::IsoDeposit &deposit,
		 const reco::SuperClusterRef& sc, const CaloSubdetectorGeometry* subdet,
		 const CaloGeometry* caloGeom,
		 const EcalRecHitCollection &hits,
		 //const EcalChannelStatus* chStatus,
		 const EcalSeverityLevelAlgo* sevLevel,
		 bool barrel) const;

    double etMin_ ;
    double energyMin_ ;
    double extRadius_ ;
    double intRadius_ ;
    double intStrip_ ;
    edm::InputTag barrelEcalHitsTag_;
    edm::InputTag endcapEcalHitsTag_;
    edm::EDGetTokenT<EcalRecHitCollection> barrelEcalHitsToken_;
    edm::EDGetTokenT<EcalRecHitCollection> endcapEcalHitsToken_;
    bool fakeNegativeDeposit_;
    bool  tryBoth_;
    bool  useEt_;
    bool  vetoClustered_;
    bool  sameTag_;
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
}
#endif
