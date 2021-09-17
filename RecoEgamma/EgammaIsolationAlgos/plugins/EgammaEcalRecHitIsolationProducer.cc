//*****************************************************************************
// File:      EgammaEcalRecHitIsolationProducer.cc
// ----------------------------------------------------------------------------
// OrigAuth:  Matthias Mozer, adapted from EgammaHcalIsolationProducer by S. Harper
// Institute: IIHE-VUB, RAL
//=============================================================================
//*****************************************************************************

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandAssociation.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaRecHitIsolation.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgoRcd.h"

class EgammaEcalRecHitIsolationProducer : public edm::global::EDProducer<> {
public:
  explicit EgammaEcalRecHitIsolationProducer(const edm::ParameterSet&);

  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  const edm::EDGetTokenT<edm::View<reco::Candidate>> emObjectProducer_;
  const edm::EDGetTokenT<EcalRecHitCollection> ecalBarrelRecHitCollection_;
  const edm::EDGetTokenT<EcalRecHitCollection> ecalEndcapRecHitCollection_;

  double egIsoPtMinBarrel_;       //minimum Et noise cut
  double egIsoEMinBarrel_;        //minimum E noise cut
  double egIsoPtMinEndcap_;       //minimum Et noise cut
  double egIsoEMinEndcap_;        //minimum E noise cut
  double egIsoConeSizeOut_;       //outer cone size
  double egIsoConeSizeInBarrel_;  //inner cone size
  double egIsoConeSizeInEndcap_;  //inner cone size
  double egIsoJurassicWidth_;     // exclusion strip width for jurassic veto

  bool useIsolEt_;  //switch for isolEt rather than isolE
  bool tryBoth_;    // use rechits from barrel + endcap
  bool subtract_;   // subtract SC energy (allows veto cone of zero size)

  bool useNumCrystals_;  // veto on number of crystals
  bool vetoClustered_;   // veto all clusterd rechits

  edm::ESGetToken<EcalSeverityLevelAlgo, EcalSeverityLevelAlgoRcd> sevLvToken_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> caloGeometrytoken_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(EgammaEcalRecHitIsolationProducer);

EgammaEcalRecHitIsolationProducer::EgammaEcalRecHitIsolationProducer(const edm::ParameterSet& config)
    //inputs
    : emObjectProducer_{consumes(config.getParameter<edm::InputTag>("emObjectProducer"))},
      ecalBarrelRecHitCollection_{consumes(config.getParameter<edm::InputTag>("ecalBarrelRecHitCollection"))},
      ecalEndcapRecHitCollection_{consumes(config.getParameter<edm::InputTag>("ecalEndcapRecHitCollection"))} {
  //vetos
  egIsoPtMinBarrel_ = config.getParameter<double>("etMinBarrel");
  egIsoEMinBarrel_ = config.getParameter<double>("eMinBarrel");
  egIsoPtMinEndcap_ = config.getParameter<double>("etMinEndcap");
  egIsoEMinEndcap_ = config.getParameter<double>("eMinEndcap");
  egIsoConeSizeInBarrel_ = config.getParameter<double>("intRadiusBarrel");
  egIsoConeSizeInEndcap_ = config.getParameter<double>("intRadiusEndcap");
  egIsoConeSizeOut_ = config.getParameter<double>("extRadius");
  egIsoJurassicWidth_ = config.getParameter<double>("jurassicWidth");

  // options
  useIsolEt_ = config.getParameter<bool>("useIsolEt");
  tryBoth_ = config.getParameter<bool>("tryBoth");
  subtract_ = config.getParameter<bool>("subtract");
  useNumCrystals_ = config.getParameter<bool>("useNumCrystals");
  vetoClustered_ = config.getParameter<bool>("vetoClustered");

  //EventSetup Tokens
  sevLvToken_ = esConsumes();
  caloGeometrytoken_ = esConsumes();

  //register your products
  produces<edm::ValueMap<double>>();
}

// ------------ method called to produce the data  ------------
void EgammaEcalRecHitIsolationProducer::produce(edm::StreamID,
                                                edm::Event& iEvent,
                                                const edm::EventSetup& iSetup) const {
  // Get the  filtered objects
  auto emObjectHandle = iEvent.getHandle(emObjectProducer_);

  // Next get Ecal hits barrel
  auto ecalBarrelRecHitHandle = iEvent.getHandle(ecalBarrelRecHitCollection_);

  // Next get Ecal hits endcap
  auto ecalEndcapRecHitHandle = iEvent.getHandle(ecalEndcapRecHitCollection_);

  edm::ESHandle<EcalSeverityLevelAlgo> sevlv = iSetup.getHandle(sevLvToken_);
  const EcalSeverityLevelAlgo* sevLevel = sevlv.product();

  //Get Calo Geometry
  edm::ESHandle<CaloGeometry> pG = iSetup.getHandle(caloGeometrytoken_);
  const CaloGeometry* caloGeom = pG.product();

  //reco::CandViewDoubleAssociations* isoMap = new reco::CandViewDoubleAssociations( reco::CandidateBaseRefProd( emObjectHandle ) );
  auto isoMap = std::make_unique<edm::ValueMap<double>>();
  edm::ValueMap<double>::Filler filler(*isoMap);
  std::vector<double> retV(emObjectHandle->size(), 0);

  EgammaRecHitIsolation ecalBarrelIsol(egIsoConeSizeOut_,
                                       egIsoConeSizeInBarrel_,
                                       egIsoJurassicWidth_,
                                       egIsoPtMinBarrel_,
                                       egIsoEMinBarrel_,
                                       caloGeom,
                                       *ecalBarrelRecHitHandle,
                                       sevLevel,
                                       DetId::Ecal);
  ecalBarrelIsol.setUseNumCrystals(useNumCrystals_);
  ecalBarrelIsol.setVetoClustered(vetoClustered_);

  EgammaRecHitIsolation ecalEndcapIsol(egIsoConeSizeOut_,
                                       egIsoConeSizeInEndcap_,
                                       egIsoJurassicWidth_,
                                       egIsoPtMinEndcap_,
                                       egIsoEMinEndcap_,
                                       caloGeom,
                                       *ecalEndcapRecHitHandle,
                                       sevLevel,
                                       DetId::Ecal);
  ecalEndcapIsol.setUseNumCrystals(useNumCrystals_);
  ecalEndcapIsol.setVetoClustered(vetoClustered_);

  for (size_t i = 0; i < emObjectHandle->size(); ++i) {
    //i need to know if its in the barrel/endcap so I get the supercluster handle to find out the detector eta
    //this might not be the best way, are we guaranteed that eta<1.5 is barrel
    //this can be safely replaced by another method which determines where the emobject is
    //then we either get the isolation Et or isolation Energy depending on user selection
    double isoValue = 0.;

    reco::SuperClusterRef superClus = emObjectHandle->at(i).get<reco::SuperClusterRef>();

    if (tryBoth_) {  //barrel + endcap
      if (useIsolEt_)
        isoValue =
            ecalBarrelIsol.getEtSum(&(emObjectHandle->at(i))) + ecalEndcapIsol.getEtSum(&(emObjectHandle->at(i)));
      else
        isoValue = ecalBarrelIsol.getEnergySum(&(emObjectHandle->at(i))) +
                   ecalEndcapIsol.getEnergySum(&(emObjectHandle->at(i)));
    } else if (fabs(superClus->eta()) < 1.479) {  //barrel
      if (useIsolEt_)
        isoValue = ecalBarrelIsol.getEtSum(&(emObjectHandle->at(i)));
      else
        isoValue = ecalBarrelIsol.getEnergySum(&(emObjectHandle->at(i)));
    } else {  //endcap
      if (useIsolEt_)
        isoValue = ecalEndcapIsol.getEtSum(&(emObjectHandle->at(i)));
      else
        isoValue = ecalEndcapIsol.getEnergySum(&(emObjectHandle->at(i)));
    }

    //we subtract off the electron energy here as well
    double subtractVal = 0;

    if (useIsolEt_)
      subtractVal = superClus.get()->rawEnergy() * sin(2 * atan(exp(-superClus.get()->eta())));
    else
      subtractVal = superClus.get()->rawEnergy();

    if (subtract_)
      isoValue -= subtractVal;

    retV[i] = isoValue;
    //all done, isolation is now in the map

  }  //end of loop over em objects

  filler.insert(emObjectHandle, retV.begin(), retV.end());
  filler.fill();

  iEvent.put(std::move(isoMap));
}

//define this as a plug-in
//DEFINE_FWK_MODULE(EgammaRecHitIsolation,Producer);
