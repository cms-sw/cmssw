#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/Muon.h"

#include "L1Trigger/L1TCalorimeter/interface/CaloTools.h"
#include "HLTrigger/HLTcore/interface/defaultModuleLabel.h"

/**************************************************************
/ purpose: enable filtering of calo objects in eta/phi or deltaR
/          regions around generic objects 
/
/ operation : accepts all objects with
/             (dEta <dEtaMax  && dPhi < dPhiMax) || dR < dRMax
/             so the OR of a rectangular region and cone region
****************************************************************/

//this is a struct which contains all the eta/phi regions
//from which to filter the calo objs
class EtaPhiRegion {
private:
  float centreEta_;
  float centrePhi_;
  float maxDeltaR2_;
  float maxDEta_;
  float maxDPhi_;

public:
  EtaPhiRegion(float iEta, float iPhi, float iDR, float iDEta, float iDPhi)
      : centreEta_(iEta), centrePhi_(iPhi), maxDeltaR2_(iDR * iDR), maxDEta_(iDEta), maxDPhi_(iDPhi) {}
  ~EtaPhiRegion() {}
  bool operator()(float eta, float phi) const {
    return reco::deltaR2(eta, phi, centreEta_, centrePhi_) < maxDeltaR2_ ||
           (std::abs(eta - centreEta_) < maxDEta_ && std::abs(reco::deltaPhi(phi, centrePhi_)) < maxDPhi_);
  }
};

class EtaPhiRegionDataBase {
public:
  EtaPhiRegionDataBase() {}
  virtual ~EtaPhiRegionDataBase() = default;
  virtual void getEtaPhiRegions(const edm::Event&, std::vector<EtaPhiRegion>&) const = 0;
};

//this class stores the tokens to access the objects around which we wish to filter
//it makes a vector of EtaPhiRegions which are then used to filter the CaloObjs
template <typename T1>
class EtaPhiRegionData : public EtaPhiRegionDataBase {
private:
  float minEt_;
  float maxEt_;
  float maxDeltaR_;
  float maxDEta_;
  float maxDPhi_;
  edm::EDGetTokenT<T1> token_;

public:
  EtaPhiRegionData(const edm::ParameterSet& para, edm::ConsumesCollector& consumesColl)
      : minEt_(para.getParameter<double>("minEt")),
        maxEt_(para.getParameter<double>("maxEt")),
        maxDeltaR_(para.getParameter<double>("maxDeltaR")),
        maxDEta_(para.getParameter<double>("maxDEta")),
        maxDPhi_(para.getParameter<double>("maxDPhi")),
        token_(consumesColl.consumes<T1>(para.getParameter<edm::InputTag>("inputColl"))) {}

  void getEtaPhiRegions(const edm::Event&, std::vector<EtaPhiRegion>&) const override;
};

template <typename CaloObjType, typename CaloObjCollType = edm::SortedCollection<CaloObjType>>
class HLTCaloObjInRegionsProducer : public edm::stream::EDProducer<> {
public:
  HLTCaloObjInRegionsProducer(const edm::ParameterSet& ps);
  ~HLTCaloObjInRegionsProducer() override {}

  void produce(edm::Event&, const edm::EventSetup&) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  EtaPhiRegionDataBase* createEtaPhiRegionData(const std::string&,
                                               const edm::ParameterSet&,
                                               edm::ConsumesCollector&&);  //calling function owns this
  static std::unique_ptr<CaloObjCollType> makeFilteredColl(const edm::Handle<CaloObjCollType>& inputColl,
                                                           const edm::ESHandle<CaloGeometry>& caloGeomHandle,
                                                           const std::vector<EtaPhiRegion>& regions);
  static bool validIDForGeom(const DetId& id);
  std::vector<std::string> outputProductNames_;
  std::vector<edm::InputTag> inputCollTags_;
  std::vector<edm::EDGetTokenT<CaloObjCollType>> inputTokens_;
  std::vector<std::unique_ptr<EtaPhiRegionDataBase>> etaPhiRegionData_;
};

template <typename CaloObjType, typename CaloObjCollType>
HLTCaloObjInRegionsProducer<CaloObjType, CaloObjCollType>::HLTCaloObjInRegionsProducer(const edm::ParameterSet& para) {
  const std::vector<edm::ParameterSet> etaPhiRegions =
      para.getParameter<std::vector<edm::ParameterSet>>("etaPhiRegions");
  for (auto& pset : etaPhiRegions) {
    const std::string type = pset.getParameter<std::string>("type");
    etaPhiRegionData_.emplace_back(createEtaPhiRegionData(
        type,
        pset,
        consumesCollector()));  //meh I was going to use a factory but it was going to be overly complex for my needs
  }

  outputProductNames_ = para.getParameter<std::vector<std::string>>("outputProductNames");
  inputCollTags_ = para.getParameter<std::vector<edm::InputTag>>("inputCollTags");
  if (outputProductNames_.size() != inputCollTags_.size()) {
    throw cms::Exception("InvalidConfiguration")
        << " error outputProductNames and inputCollTags must be the same size, they are " << outputProductNames_.size()
        << " vs " << inputCollTags_.size();
  }
  for (unsigned int collNr = 0; collNr < inputCollTags_.size(); collNr++) {
    inputTokens_.push_back(consumes<CaloObjCollType>(inputCollTags_[collNr]));
    produces<CaloObjCollType>(outputProductNames_[collNr]);
  }
}

template <typename CaloObjType, typename CaloObjCollType>
void HLTCaloObjInRegionsProducer<CaloObjType, CaloObjCollType>::fillDescriptions(
    edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  std::vector<std::string> outputProductNames;
  outputProductNames.push_back("EcalRegionalRecHitsEB");
  desc.add<std::vector<std::string>>("outputProductNames", outputProductNames);
  std::vector<edm::InputTag> inputColls;
  inputColls.push_back(edm::InputTag("hltHcalDigis"));
  desc.add<std::vector<edm::InputTag>>("inputCollTags", inputColls);
  std::vector<edm::ParameterSet> etaPhiRegions;

  edm::ParameterSet ecalCandPSet;
  ecalCandPSet.addParameter<std::string>("type", "RecoEcalCandidate");
  ecalCandPSet.addParameter<double>("minEt", -1);
  ecalCandPSet.addParameter<double>("maxEt", -1);
  ecalCandPSet.addParameter<double>("maxDeltaR", 0.5);
  ecalCandPSet.addParameter<double>("maxDEta", 0.);
  ecalCandPSet.addParameter<double>("maxDPhi", 0.);
  ecalCandPSet.addParameter<edm::InputTag>("inputColl", edm::InputTag("hltEgammaCandidates"));
  etaPhiRegions.push_back(ecalCandPSet);

  edm::ParameterSetDescription etaPhiRegionDesc;
  etaPhiRegionDesc.add<std::string>("type");
  etaPhiRegionDesc.add<double>("minEt");
  etaPhiRegionDesc.add<double>("maxEt");
  etaPhiRegionDesc.add<double>("maxDeltaR");
  etaPhiRegionDesc.add<double>("maxDEta");
  etaPhiRegionDesc.add<double>("maxDPhi");
  etaPhiRegionDesc.add<edm::InputTag>("inputColl");
  desc.addVPSet("etaPhiRegions", etaPhiRegionDesc, etaPhiRegions);

  descriptions.add(defaultModuleLabel<HLTCaloObjInRegionsProducer<CaloObjType, CaloObjCollType>>(), desc);
}

template <typename CaloObjType, typename CaloObjCollType>
void HLTCaloObjInRegionsProducer<CaloObjType, CaloObjCollType>::produce(edm::Event& event,
                                                                        const edm::EventSetup& setup) {
  // get the collection geometry:
  edm::ESHandle<CaloGeometry> caloGeomHandle;
  setup.get<CaloGeometryRecord>().get(caloGeomHandle);

  std::vector<EtaPhiRegion> regions;
  std::for_each(etaPhiRegionData_.begin(),
                etaPhiRegionData_.end(),
                [&event, &regions](const std::unique_ptr<EtaPhiRegionDataBase>& input) {
                  input->getEtaPhiRegions(event, regions);
                });

  for (size_t inputCollNr = 0; inputCollNr < inputTokens_.size(); inputCollNr++) {
    edm::Handle<CaloObjCollType> inputColl;
    event.getByToken(inputTokens_[inputCollNr], inputColl);

    if (!(inputColl.isValid())) {
      edm::LogError("ProductNotFound") << "could not get a handle on the " << typeid(CaloObjCollType).name()
                                       << " named " << inputCollTags_[inputCollNr].encode() << std::endl;
      continue;
    }
    auto outputColl = makeFilteredColl(inputColl, caloGeomHandle, regions);
    event.put(std::move(outputColl), outputProductNames_[inputCollNr]);
  }
}

template <typename CaloObjType, typename CaloObjCollType>
std::unique_ptr<CaloObjCollType> HLTCaloObjInRegionsProducer<CaloObjType, CaloObjCollType>::makeFilteredColl(
    const edm::Handle<CaloObjCollType>& inputColl,
    const edm::ESHandle<CaloGeometry>& caloGeomHandle,
    const std::vector<EtaPhiRegion>& regions) {
  auto outputColl = std::make_unique<CaloObjCollType>();
  if (!inputColl->empty()) {
    const CaloSubdetectorGeometry* subDetGeom = caloGeomHandle->getSubdetectorGeometry(inputColl->begin()->id());
    if (!regions.empty()) {
      for (const CaloObjType& obj : *inputColl) {
        auto objGeom = subDetGeom->getGeometry(obj.id());
        if (objGeom == nullptr) {
          //wondering what to do here
          //something is very very wrong
          //given HLT should never crash or throw, decided to log an error
          //update: so turns out HCAL can pass through calibration channels in QIE11 so for that module, its an expected behaviour
          //so we check if the ID is valid
          if (validIDForGeom(obj.id())) {
            edm::LogError("HLTCaloObjInRegionsProducer")
                << "for an object of type " << typeid(CaloObjType).name() << " the geometry returned null for id "
                << DetId(obj.id()).rawId() << " with initial ID " << DetId(inputColl->begin()->id()).rawId()
                << " in HLTCaloObjsInRegion, this shouldnt be possible and something has gone wrong, auto accepting "
                   "hit";
          }
          outputColl->push_back(obj);
          continue;
        }
        float eta = objGeom->getPosition().eta();
        float phi = objGeom->getPosition().phi();

        for (const auto& region : regions) {
          if (region(eta, phi)) {
            outputColl->push_back(obj);
            break;
          }
        }
      }
    }  //end check of empty regions
  }    //end check of empty rec-hits
  return outputColl;
}

//tells us if an ID should have a valid geometry
//it assumes that all IDs do except those specifically mentioned
//HCAL for example have laser calibs in the digi collection so
//so we have to ensure that HCAL is HB,HE or HO
template <typename CaloObjType, typename CaloObjCollType>
bool HLTCaloObjInRegionsProducer<CaloObjType, CaloObjCollType>::validIDForGeom(const DetId& id) {
  if (id.det() == DetId::Hcal) {
    if (id.subdetId() == HcalSubdetector::HcalEmpty || id.subdetId() == HcalSubdetector::HcalOther) {
      return false;
    }
  }
  return true;
}

template <typename CaloObjType, typename CaloObjCollType>
EtaPhiRegionDataBase* HLTCaloObjInRegionsProducer<CaloObjType, CaloObjCollType>::createEtaPhiRegionData(
    const std::string& type, const edm::ParameterSet& para, edm::ConsumesCollector&& consumesColl) {
  if (type == "L1EGamma") {
    return new EtaPhiRegionData<l1t::EGammaBxCollection>(para, consumesColl);
  } else if (type == "L1Jet") {
    return new EtaPhiRegionData<l1t::JetBxCollection>(para, consumesColl);
  } else if (type == "L1Muon") {
    return new EtaPhiRegionData<l1t::MuonBxCollection>(para, consumesColl);
  } else if (type == "L1Tau") {
    return new EtaPhiRegionData<l1t::TauBxCollection>(para, consumesColl);
  } else if (type == "RecoEcalCandidate") {
    return new EtaPhiRegionData<reco::RecoEcalCandidateCollection>(para, consumesColl);
  } else if (type == "RecoChargedCandidate") {
    return new EtaPhiRegionData<reco::RecoChargedCandidateCollection>(para, consumesColl);
  } else if (type == "Electron") {
    return new EtaPhiRegionData<reco::Electron>(para, consumesColl);
  } else {
    //this is a major issue and could lead to rather subtle efficiency losses, so if its incorrectly configured, we're aborting the job!
    throw cms::Exception("InvalidConfig")
        << " type " << type
        << " is not recognised, this means the rec-hit you think you are keeping may not be and you should fix this "
           "error as it can lead to hard to find efficiency loses"
        << std::endl;
  }
}

template <typename CandCollType>
void EtaPhiRegionData<CandCollType>::getEtaPhiRegions(const edm::Event& event,
                                                      std::vector<EtaPhiRegion>& regions) const {
  edm::Handle<CandCollType> cands;
  event.getByToken(token_, cands);

  for (auto const& cand : *cands) {
    if (cand.et() >= minEt_ && (maxEt_ < 0 || cand.et() < maxEt_)) {
      regions.push_back(EtaPhiRegion(cand.eta(), cand.phi(), maxDeltaR_, maxDEta_, maxDPhi_));
    }
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/HcalDigi/interface/HBHEDataFrame.h"
using HLTHcalDigisInRegionsProducer = HLTCaloObjInRegionsProducer<HBHEDataFrame>;
DEFINE_FWK_MODULE(HLTHcalDigisInRegionsProducer);

#include "DataFormats/HcalDigi/interface/QIE11DataFrame.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
using HLTHcalQIE11DigisInRegionsProducer = HLTCaloObjInRegionsProducer<QIE11DataFrame, QIE11DigiCollection>;
DEFINE_FWK_MODULE(HLTHcalQIE11DigisInRegionsProducer);

#include "DataFormats/HcalDigi/interface/QIE10DataFrame.h"
using HLTHcalQIE10DigisInRegionsProducer = HLTCaloObjInRegionsProducer<QIE10DataFrame, QIE10DigiCollection>;
DEFINE_FWK_MODULE(HLTHcalQIE10DigisInRegionsProducer);

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
using HLTEcalEBDigisInRegionsProducer = HLTCaloObjInRegionsProducer<EBDataFrame, EBDigiCollection>;
DEFINE_FWK_MODULE(HLTEcalEBDigisInRegionsProducer);
using HLTEcalEEDigisInRegionsProducer = HLTCaloObjInRegionsProducer<EEDataFrame, EEDigiCollection>;
DEFINE_FWK_MODULE(HLTEcalEEDigisInRegionsProducer);
using HLTEcalESDigisInRegionsProducer = HLTCaloObjInRegionsProducer<ESDataFrame, ESDigiCollection>;
DEFINE_FWK_MODULE(HLTEcalESDigisInRegionsProducer);

//these two classes are intended to ultimately replace the EcalRecHit and EcalUncalibratedRecHit
//instances of HLTRecHitInAllL1RegionsProducer, particulary as we're free of legacy / stage-1 L1 now
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
using HLTEcalRecHitsInRegionsProducer = HLTCaloObjInRegionsProducer<EcalRecHit>;
DEFINE_FWK_MODULE(HLTEcalRecHitsInRegionsProducer);
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
using HLTEcalUnCalibRecHitsInRegionsProducer = HLTCaloObjInRegionsProducer<EcalUncalibratedRecHit>;
DEFINE_FWK_MODULE(HLTEcalUnCalibRecHitsInRegionsProducer);
