#include "CalibCalorimetry/CaloMiscalibTools/interface/HcalRecHitRecalib.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/MiscalibReaderFromXMLHcal.h"

HcalRecHitRecalib::HcalRecHitRecalib(const edm::ParameterSet& iConfig)
    : tok_hbhe_(consumes<HBHERecHitCollection>(iConfig.getParameter<edm::InputTag>("hbheInput"))),
      tok_ho_(consumes<HORecHitCollection>(iConfig.getParameter<edm::InputTag>("hoInput"))),
      tok_hf_(consumes<HFRecHitCollection>(iConfig.getParameter<edm::InputTag>("hfInput"))),
      topologyToken_(esConsumes<edm::Transition::BeginRun>()),
      recalibHBHEHits_(iConfig.getParameter<std::string>("RecalibHBHEHitCollection")),
      recalibHFHits_(iConfig.getParameter<std::string>("RecalibHFHitCollection")),
      recalibHOHits_(iConfig.getParameter<std::string>("RecalibHOHitCollection")),
      hcalfileinpath_(iConfig.getUntrackedParameter<std::string>("fileNameHcal", "")),
      refactor_(iConfig.getUntrackedParameter<double>("Refactor", (double)1)),
      refactor_mean_(iConfig.getUntrackedParameter<double>("Refactor_mean", (double)1)) {
  //register your products
  produces<HBHERecHitCollection>(recalibHBHEHits_);
  produces<HFRecHitCollection>(recalibHFHits_);
  produces<HORecHitCollection>(recalibHOHits_);

  // here read them from xml (particular to HCAL)
  edm::FileInPath hcalfiletmp("CalibCalorimetry/CaloMiscalibTools/data/" + hcalfileinpath_);
  hcalfile_ = hcalfiletmp.fullPath();
}

HcalRecHitRecalib::~HcalRecHitRecalib() {}

void HcalRecHitRecalib::beginRun(const edm::Run&, const edm::EventSetup& iSetup) {
  const HcalTopology& topology = iSetup.getData(topologyToken_);

  mapHcal_.prefillMap(topology);

  MiscalibReaderFromXMLHcal hcalreader_(mapHcal_);
  if (!hcalfile_.empty())
    hcalreader_.parseXMLMiscalibFile(hcalfile_);
  mapHcal_.print();
}

// ------------ method called to produce the data  ------------
void HcalRecHitRecalib::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace std;

  Handle<HBHERecHitCollection> HBHERecHitsHandle;
  Handle<HFRecHitCollection> HFRecHitsHandle;
  Handle<HORecHitCollection> HORecHitsHandle;

  const HBHERecHitCollection* HBHERecHits = nullptr;
  const HFRecHitCollection* HFRecHits = nullptr;
  const HORecHitCollection* HORecHits = nullptr;

  iEvent.getByToken(tok_hbhe_, HBHERecHitsHandle);
  if (!HBHERecHitsHandle.isValid()) {
    LogDebug("") << "HcalREcHitRecalib: Error! can't get product!" << std::endl;
  } else {
    HBHERecHits = HBHERecHitsHandle.product();  // get a ptr to the product
  }

  iEvent.getByToken(tok_ho_, HORecHitsHandle);
  if (!HORecHitsHandle.isValid()) {
    LogDebug("") << "HcalREcHitRecalib: Error! can't get product!" << std::endl;
  } else {
    HORecHits = HORecHitsHandle.product();  // get a ptr to the product
  }

  iEvent.getByToken(tok_hf_, HFRecHitsHandle);
  if (!HFRecHitsHandle.isValid()) {
    LogDebug("") << "HcalREcHitRecalib: Error! can't get product!" << std::endl;
  } else {
    HFRecHits = HFRecHitsHandle.product();  // get a ptr to the product
  }

  //Create empty output collections
  auto RecalibHBHERecHitCollection = std::make_unique<HBHERecHitCollection>();
  auto RecalibHFRecHitCollection = std::make_unique<HFRecHitCollection>();
  auto RecalibHORecHitCollection = std::make_unique<HORecHitCollection>();

  if (HBHERecHits) {
    HBHERecHitCollection::const_iterator itHBHE;
    for (itHBHE = HBHERecHits->begin(); itHBHE != HBHERecHits->end(); ++itHBHE) {
      // make the rechit with rescaled energy and put in the output collection
      float icalconst = (mapHcal_.get().find(itHBHE->id().rawId()))->second;
      icalconst = refactor_mean_ +
                  (icalconst - refactor_mean_) * refactor_;  //apply additional scaling factor (works if gaussian)
      HBHERecHit aHit(itHBHE->id(), itHBHE->energy() * icalconst, itHBHE->time());

      RecalibHBHERecHitCollection->push_back(aHit);
    }
  }

  if (HFRecHits) {
    HFRecHitCollection::const_iterator itHF;
    for (itHF = HFRecHits->begin(); itHF != HFRecHits->end(); ++itHF) {
      // make the rechit with rescaled energy and put in the output collection
      float icalconst = (mapHcal_.get().find(itHF->id().rawId()))->second;
      icalconst = refactor_mean_ +
                  (icalconst - refactor_mean_) * refactor_;  //apply additional scaling factor (works if gaussian)
      HFRecHit aHit(itHF->id(), itHF->energy() * icalconst, itHF->time());

      RecalibHFRecHitCollection->push_back(aHit);
    }
  }

  if (HORecHits) {
    HORecHitCollection::const_iterator itHO;
    for (itHO = HORecHits->begin(); itHO != HORecHits->end(); ++itHO) {
      // make the rechit with rescaled energy and put in the output collection
      float icalconst = (mapHcal_.get().find(itHO->id().rawId()))->second;
      icalconst = refactor_mean_ +
                  (icalconst - refactor_mean_) * refactor_;  //apply additional scaling factor (works if gaussian)
      HORecHit aHit(itHO->id(), itHO->energy() * icalconst, itHO->time());

      RecalibHORecHitCollection->push_back(aHit);
    }
  }

  //Put Recalibrated rechit in the event
  iEvent.put(std::move(RecalibHBHERecHitCollection), recalibHBHEHits_);
  iEvent.put(std::move(RecalibHFRecHitCollection), recalibHFHits_);
  iEvent.put(std::move(RecalibHORecHitCollection), recalibHOHits_);
}
