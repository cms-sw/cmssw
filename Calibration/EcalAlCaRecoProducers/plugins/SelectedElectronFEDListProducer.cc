#include "Calibration/EcalAlCaRecoProducers/plugins/SelectedElectronFEDListProducer.h"

#include <fstream>
#include "HLTrigger/HLTcore/interface/defaultModuleLabel.h"

#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

//#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "FWCore/Utilities/interface/InputTag.h"
// raw data
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
//#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

// Geometry
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
// strip geometry
#include "CalibFormats/SiStripObjects/interface/SiStripRegionCabling.h"
#include "CalibTracker/Records/interface/SiStripRegionCablingRcd.h"

// egamma objects
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"

// Hcal objects
#include "CondFormats/HcalObjects/interface/HcalElectronicsMap.h"

// Strip and pixel
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"

// detector id
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalElectronicsId.h"
// Hcal rec hit
#include "DataFormats/CaloRecHit/interface/CaloRecHit.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
// Geometry
#include "Geometry/Records/interface/CaloTopologyRecord.h"
#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"
#include "Geometry/EcalAlgo/interface/EcalPreshowerGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
// strip geometry
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
// Message logger
#include "FWCore/MessageLogger/interface/MessageLogger.h"
// Strip and pixel
#include "CondFormats/SiStripObjects/interface/FedChannelConnection.h"
#include "CondFormats/DataRecord/interface/SiPixelFedCablingMapRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingTree.h"

// Hcal objects
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"

using namespace std;

/// Producer constructor
template <typename TEle, typename TCand>
SelectedElectronFEDListProducer<TEle, TCand>::SelectedElectronFEDListProducer(const edm::ParameterSet& iConfig) {
  // input electron collection Tag
  if (iConfig.existsAs<std::vector<edm::InputTag>>("electronTags")) {
    electronTags_ = iConfig.getParameter<std::vector<edm::InputTag>>("electronTags");
    if (electronTags_.empty())
      throw cms::Exception("Configuration")
          << "[SelectedElectronFEDListProducer] empty electron collection is given --> at least one \n";
  } else
    throw cms::Exception("Configuration")
        << "[SelectedElectronFEDListProducer] no electron collection are given --> need at least one \n";

  // Consumes for the electron collection
  LogDebug("SelectedElectronFEDListProducer") << " Electron Collections" << std::endl;
  for (std::vector<edm::InputTag>::const_iterator itEleTag = electronTags_.begin(); itEleTag != electronTags_.end();
       ++itEleTag) {
    electronToken_.push_back(consumes<TEleColl>(*itEleTag));
    LogDebug("SelectedElectronFEDListProducer") << " Ele collection: " << *(itEleTag) << std::endl;
  }

  // input RecoEcalCandidate collection Tag
  if (iConfig.existsAs<std::vector<edm::InputTag>>("recoEcalCandidateTags")) {
    recoEcalCandidateTags_ = iConfig.getParameter<std::vector<edm::InputTag>>("recoEcalCandidateTags");
    if (recoEcalCandidateTags_.empty())
      throw cms::Exception("Configuration") << "[SelectedElectronFEDListProducer] empty ecal candidate collections "
                                               "collection is given --> at least one \n";
  } else
    throw cms::Exception("Configuration") << "[SelectedElectronFEDListProducer] no electron reco ecal candidate "
                                             "collection are given --> need at least one \n";

  // Consumes for the recoEcal candidate collection
  for (std::vector<edm::InputTag>::const_iterator itEcalCandTag = recoEcalCandidateTags_.begin();
       itEcalCandTag != recoEcalCandidateTags_.end();
       ++itEcalCandTag) {
    recoEcalCandidateToken_.push_back(consumes<trigger::TriggerFilterObjectWithRefs>(*itEcalCandTag));
    LogDebug("SelectedElectronFEDListProducer") << " Reco ecal candidate collection: " << *(itEcalCandTag) << std::endl;
  }

  // list of gsf collections
  if (iConfig.existsAs<std::vector<int>>("isGsfElectronCollection")) {
    isGsfElectronCollection_ = iConfig.getParameter<std::vector<int>>("isGsfElectronCollection");
    if (isGsfElectronCollection_.empty())
      throw cms::Exception("Configuration")
          << "[SelectedElectronFEDListProducer] empty electron flag collection --> at least one \n";
  } else
    throw cms::Exception("Configuration")
        << "[SelectedElectronFEDListProducer] no electron flag are given --> need at least one \n";

  if (isGsfElectronCollection_.size() != electronTags_.size() or
      isGsfElectronCollection_.size() != recoEcalCandidateTags_.size())
    throw cms::Exception("Configuration") << "[SelectedElectronFEDListProducer] electron flag , electron collection "
                                             "and reco ecal cand collection must have the same size ! \n";

  // take the beam spot Tag
  if (iConfig.existsAs<edm::InputTag>("beamSpot"))
    beamSpotTag_ = iConfig.getParameter<edm::InputTag>("beamSpot");
  else
    beamSpotTag_ = edm::InputTag("hltOnlineBeamSpot");

  if (!(beamSpotTag_ == edm::InputTag("")))
    beamSpotToken_ = consumes<reco::BeamSpot>(beamSpotTag_);

  LogDebug("SelectedElectronFEDListProducer") << " Beam Spot Tag " << beamSpotTag_ << std::endl;

  // take the HBHE recHit Tag
  if (iConfig.existsAs<edm::InputTag>("HBHERecHitTag"))
    HBHERecHitTag_ = iConfig.getParameter<edm::InputTag>("HBHERecHitTag");
  else
    HBHERecHitTag_ = edm::InputTag("hltHbhereco");

  if (!(HBHERecHitTag_ == edm::InputTag("")))
    hbheRecHitToken_ = consumes<HBHERecHitCollection>(HBHERecHitTag_);

  // raw data collector label
  if (iConfig.existsAs<edm::InputTag>("rawDataTag"))
    rawDataTag_ = iConfig.getParameter<edm::InputTag>("rawDataTag");
  else
    rawDataTag_ = edm::InputTag("rawDataCollector");

  if (!(rawDataTag_ == edm::InputTag("")))
    rawDataToken_ = consumes<FEDRawDataCollection>(rawDataTag_);

  LogDebug("SelectedElectronFEDListProducer") << " RawDataInput " << rawDataTag_ << std::endl;

  // add a set of selected feds
  if (iConfig.existsAs<std::vector<int>>("addThisSelectedFEDs")) {
    addThisSelectedFEDs_ = iConfig.getParameter<std::vector<int>>("addThisSelectedFEDs");
    if (addThisSelectedFEDs_.empty())
      addThisSelectedFEDs_.push_back(-1);
  } else
    addThisSelectedFEDs_.push_back(-1);

  std::vector<int>::const_iterator AddFed = addThisSelectedFEDs_.begin();
  for (; AddFed != addThisSelectedFEDs_.end(); ++AddFed)
    LogDebug("SelectedElectronFEDListProducer") << " Additional FED: " << *(AddFed) << std::endl;

  // ES look up table path
  if (iConfig.existsAs<std::string>("ESLookupTable"))
    ESLookupTable_ = iConfig.getParameter<edm::FileInPath>("ESLookupTable");
  else
    ESLookupTable_ = edm::FileInPath("EventFilter/ESDigiToRaw/data/ES_lookup_table.dat");

  // output model label
  if (iConfig.existsAs<std::string>("outputLabelModule"))
    outputLabelModule_ = iConfig.getParameter<std::string>("outputLabelModule");
  else
    outputLabelModule_ = "streamElectronRawData";

  LogDebug("SelectedElectronFEDListProducer") << " Output Label " << outputLabelModule_ << std::endl;

  // dR for the strip region
  if (iConfig.existsAs<double>("dRStripRegion"))
    dRStripRegion_ = iConfig.getParameter<double>("dRStripRegion");
  else
    dRStripRegion_ = 0.5;

  LogDebug("SelectedElectronFEDListProducer") << " dRStripRegion " << dRStripRegion_ << std::endl;

  // dR for the hcal region
  if (iConfig.existsAs<double>("dRHcalRegion"))
    dRHcalRegion_ = iConfig.getParameter<double>("dRHcalRegion");
  else
    dRHcalRegion_ = 0.5;

  // dPhi, dEta and maxZ for pixel dump
  if (iConfig.existsAs<double>("dPhiPixelRegion"))
    dPhiPixelRegion_ = iConfig.getParameter<double>("dPhiPixelRegion");
  else
    dPhiPixelRegion_ = 0.5;

  if (iConfig.existsAs<double>("dEtaPixelRegion"))
    dEtaPixelRegion_ = iConfig.getParameter<double>("dEtaPixelRegion");
  else
    dEtaPixelRegion_ = 0.5;

  if (iConfig.existsAs<double>("maxZPixelRegion"))
    maxZPixelRegion_ = iConfig.getParameter<double>("maxZPixelRegion");
  else
    maxZPixelRegion_ = 24.;

  LogDebug("SelectedElectronFEDListProducer")
      << " dPhiPixelRegion " << dPhiPixelRegion_ << " dEtaPixelRegion " << dEtaPixelRegion_ << " MaxZPixelRegion "
      << maxZPixelRegion_ << std::endl;

  // bool
  if (iConfig.existsAs<bool>("dumpSelectedEcalFed"))
    dumpSelectedEcalFed_ = iConfig.getParameter<bool>("dumpSelectedEcalFed");
  else
    dumpSelectedEcalFed_ = true;

  if (iConfig.existsAs<bool>("dumpSelectedSiStripFed"))
    dumpSelectedSiStripFed_ = iConfig.getParameter<bool>("dumpSelectedSiStripFed");
  else
    dumpSelectedSiStripFed_ = true;

  if (iConfig.existsAs<bool>("dumpSelectedSiPixelFed"))
    dumpSelectedSiPixelFed_ = iConfig.getParameter<bool>("dumpSelectedSiPixelFed");
  else
    dumpSelectedSiPixelFed_ = true;

  if (iConfig.existsAs<bool>("dumpSelectedHCALFed"))
    dumpSelectedHCALFed_ = iConfig.getParameter<bool>("dumpSelectedHCALFed");
  else
    dumpSelectedHCALFed_ = true;

  LogDebug("SelectedElectronFEDListProducer")
      << " DumpEcalFedList set to " << dumpSelectedEcalFed_ << " DumpSelectedSiStripFed " << dumpSelectedSiStripFed_
      << " DumpSelectedSiPixelFed " << dumpSelectedSiPixelFed_ << std::endl;

  if (iConfig.existsAs<bool>("dumpAllEcalFed"))
    dumpAllEcalFed_ = iConfig.getParameter<bool>("dumpAllEcalFed");
  else
    dumpAllEcalFed_ = false;

  if (iConfig.existsAs<bool>("dumpAllTrackerFed"))
    dumpAllTrackerFed_ = iConfig.getParameter<bool>("dumpAllTrackerFed");
  else
    dumpAllTrackerFed_ = false;

  if (iConfig.existsAs<bool>("dumpAllHCALFed"))
    dumpAllHCALFed_ = iConfig.getParameter<bool>("dumpAllHCALFed");
  else
    dumpAllHCALFed_ = false;

  LogDebug("SelectedElectronFEDListProducer")
      << " DumpAllEcalFed " << dumpAllEcalFed_ << " DumpAllTrackerFed " << dumpAllTrackerFed_ << " Dump all HCAL fed "
      << dumpAllHCALFed_ << std::endl;

  // initialize pre-shower fed id --> look up table
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
      for (int k = 0; k < 40; ++k)
        for (int m = 0; m < 40; m++)
          ES_fedId_[i][j][k][m] = -1;

  // read in look-up table
  int nLines, iz, ip, ix, iy, fed, kchip, pace, bundle, fiber, optorx;
  std::ifstream ES_file;
  ES_file.open(ESLookupTable_.fullPath().c_str());
  LogDebug("SelectedElectronFEDListProducer")
      << " Look Up table for ES " << ESLookupTable_.fullPath().c_str() << std::endl;
  if (ES_file.is_open()) {
    ES_file >> nLines;
    for (int i = 0; i < nLines; ++i) {
      ES_file >> iz >> ip >> ix >> iy >> fed >> kchip >> pace >> bundle >> fiber >> optorx;
      ES_fedId_[(3 - iz) / 2 - 1][ip - 1][ix - 1][iy - 1] = fed;
    }
  } else
    LogDebug("SelectedElectronFEDListProducer")
        << " Look up table file can not be found in " << ESLookupTable_.fullPath().c_str() << std::endl;
  ES_file.close();

  // produce the final collection
  produces<FEDRawDataCollection>(outputLabelModule_);  // produce exit collection
}

template <typename TEle, typename TCand>
SelectedElectronFEDListProducer<TEle, TCand>::~SelectedElectronFEDListProducer() {
  if (!electronTags_.empty())
    electronTags_.clear();
  if (!recoEcalCandidateTags_.empty())
    recoEcalCandidateTags_.clear();
  if (!recoEcalCandidateToken_.empty())
    recoEcalCandidateToken_.clear();
  if (!electronToken_.empty())
    electronToken_.clear();
  if (!fedList_.empty())
    fedList_.clear();
  if (!pixelModuleVector_.empty())
    pixelModuleVector_.clear();
}

template <typename TEle, typename TCand>
void SelectedElectronFEDListProducer<TEle, TCand>::beginJob() {
  LogDebug("SelectedElectronFEDListProducer") << " Begin of the Job " << std::endl;
}

template <typename TEle, typename TCand>
void SelectedElectronFEDListProducer<TEle, TCand>::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // get the hcal electronics map
  edm::ESHandle<HcalDbService> pSetup;
  iSetup.get<HcalDbRecord>().get(pSetup);
  HcalReadoutMap_ = pSetup->getHcalMapping();

  // get the ecal electronics map
  edm::ESHandle<EcalElectronicsMapping> ecalmapping;
  iSetup.get<EcalMappingRcd>().get(ecalmapping);
  EcalMapping_ = ecalmapping.product();

  // get the calo geometry
  edm::ESHandle<CaloGeometry> caloGeometry;
  iSetup.get<CaloGeometryRecord>().get(caloGeometry);
  GeometryCalo_ = caloGeometry.product();

  //ES geometry
  GeometryES_ = caloGeometry->getSubdetectorGeometry(DetId::Ecal, EcalPreshower);

  // pixel tracker cabling map
  edm::ESTransientHandle<SiPixelFedCablingMap> pixelCablingMap;
  iSetup.get<SiPixelFedCablingMapRcd>().get(pixelCablingMap);
  PixelCabling_.reset();
  PixelCabling_ = pixelCablingMap->cablingTree();

  edm::ESHandle<TrackerGeometry> trackerGeometry;
  iSetup.get<TrackerDigiGeometryRecord>().get(trackerGeometry);

  if (pixelModuleVector_.empty()) {
    // build the tracker pixel module map
    std::vector<const GeomDet*>::const_iterator itTracker = trackerGeometry->dets().begin();
    for (; itTracker != trackerGeometry->dets().end(); ++itTracker) {
      int subdet = (*itTracker)->geographicalId().subdetId();
      if (!(subdet == PixelSubdetector::PixelBarrel || subdet == PixelSubdetector::PixelEndcap))
        continue;
      PixelModule module;
      module.x = (*itTracker)->position().x();
      module.y = (*itTracker)->position().y();
      module.z = (*itTracker)->position().z();
      module.Phi = (*itTracker)->position().phi();
      module.Eta = (*itTracker)->position().eta();
      module.DetId = (*itTracker)->geographicalId().rawId();
      const std::vector<sipixelobjects::CablingPathToDetUnit> path2det = PixelCabling_->pathToDetUnit(module.DetId);
      module.Fed = path2det[0].fed;

      pixelModuleVector_.push_back(module);
    }
    std::sort(pixelModuleVector_.begin(), pixelModuleVector_.end());
  }

  edm::ESHandle<SiStripRegionCabling> SiStripCablingHandle;
  iSetup.get<SiStripRegionCablingRcd>().get(SiStripCablingHandle);
  StripRegionCabling_ = SiStripCablingHandle.product();

  SiStripRegionCabling::Cabling SiStripCabling;
  SiStripCabling = StripRegionCabling_->getRegionCabling();
  regionDimension_ = StripRegionCabling_->regionDimensions();

  // event by event analysis
  // Get event raw data
  edm::Handle<FEDRawDataCollection> rawdata;
  if (!(rawDataTag_ == edm::InputTag("")))
    iEvent.getByToken(rawDataToken_, rawdata);

  // take the beam spot position
  edm::Handle<reco::BeamSpot> beamSpot;
  if (!(beamSpotTag_ == edm::InputTag("")))
    iEvent.getByToken(beamSpotToken_, beamSpot);
  if (!beamSpot.failedToGet())
    beamSpotPosition_ = beamSpot->position();
  else
    beamSpotPosition_.SetXYZ(0, 0, 0);

  // take the calo tower collection
  edm::Handle<HBHERecHitCollection> hbheRecHitHandle;
  if (!(HBHERecHitTag_ == edm::InputTag("")))
    iEvent.getByToken(hbheRecHitToken_, hbheRecHitHandle);
  const HBHERecHitCollection* hcalRecHitCollection = nullptr;
  if (!hbheRecHitHandle.failedToGet())
    hcalRecHitCollection = hbheRecHitHandle.product();

  double radTodeg = 180. / Geom::pi();

  if (dumpAllEcalFed_) {
    for (uint32_t iEcalFed = FEDNumbering::MINECALFEDID; iEcalFed <= FEDNumbering::MAXECALFEDID; iEcalFed++)
      fedList_.push_back(iEcalFed);
    for (uint32_t iESFed = FEDNumbering::MINPreShowerFEDID; iESFed <= FEDNumbering::MAXPreShowerFEDID; iESFed++)
      fedList_.push_back(iESFed);
  }

  if (dumpAllTrackerFed_) {
    for (uint32_t iPixelFed = FEDNumbering::MINSiPixelFEDID; iPixelFed <= FEDNumbering::MAXSiPixelFEDID; iPixelFed++)
      fedList_.push_back(iPixelFed);
    for (uint32_t iStripFed = FEDNumbering::MINSiStripFEDID; iStripFed <= FEDNumbering::MAXSiStripFEDID; iStripFed++)
      fedList_.push_back(iStripFed);
  }

  if (dumpAllHCALFed_) {
    for (uint32_t iHcalFed = FEDNumbering::MINHCALFEDID; iHcalFed <= FEDNumbering::MAXHCALFEDID; iHcalFed++)
      fedList_.push_back(iHcalFed);
  }

  // loop on the input electron collection vector
  TEle electron;
  edm::Ref<TCandColl> recoEcalCand;
  edm::Handle<TEleColl> electrons;
  edm::Handle<trigger::TriggerFilterObjectWithRefs> triggerRecoEcalCandidateCollection;
  std::vector<edm::Ref<TCandColl>> recoEcalCandColl;

  // iterator to electron and ecal candidate collections
  typename std::vector<edm::EDGetTokenT<TEleColl>>::const_iterator itElectronColl = electronToken_.begin();
  std::vector<int>::const_iterator itElectronCollFlag = isGsfElectronCollection_.begin();
  std::vector<edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs>>::const_iterator itRecoEcalCandColl =
      recoEcalCandidateToken_.begin();

  // if you want to dump just FED related to the triggering electron/s
  if (!dumpAllTrackerFed_ || !dumpAllEcalFed_) {
    // loop on the same time on ecal candidate and elctron collection and boolean for Gsf ones
    for (; itRecoEcalCandColl != recoEcalCandidateToken_.end() and itElectronColl != electronToken_.end() and
           itElectronCollFlag != isGsfElectronCollection_.end();
         ++itElectronColl, ++itElectronCollFlag, ++itRecoEcalCandColl) {
      // get ecal candidate collection
      iEvent.getByToken(*itRecoEcalCandColl, triggerRecoEcalCandidateCollection);
      if (triggerRecoEcalCandidateCollection.failedToGet())
        continue;

      // get gsf electron collection
      iEvent.getByToken(*itElectronColl, electrons);
      if (electrons.failedToGet())
        continue;

      triggerRecoEcalCandidateCollection->getObjects(trigger::TriggerCluster, recoEcalCandColl);
      if (recoEcalCandColl.empty())
        triggerRecoEcalCandidateCollection->getObjects(trigger::TriggerPhoton, recoEcalCandColl);
      if (recoEcalCandColl.empty())
        triggerRecoEcalCandidateCollection->getObjects(trigger::TriggerElectron, recoEcalCandColl);

      typename std::vector<edm::Ref<TCandColl>>::const_iterator itRecoEcalCand =
          recoEcalCandColl.begin();  // loop on recoEcalCandidate objects

      // loop on the recoEcalCandidates
      for (; itRecoEcalCand != recoEcalCandColl.end(); ++itRecoEcalCand) {
        recoEcalCand = (*itRecoEcalCand);
        reco::SuperClusterRef scRefRecoEcalCand =
            recoEcalCand->superCluster();  // take the supercluster in order to match with electron objects

        typename TEleColl::const_iterator itEle = electrons->begin();
        for (; itEle != electrons->end(); ++itEle) {  // loop on all the electrons inside a collection
          // get electron supercluster and the associated hit -> detID
          electron = (*itEle);
          reco::SuperClusterRef scRef = electron.superCluster();
          if (scRefRecoEcalCand != scRef)
            continue;  // mathching

          const std::vector<std::pair<DetId, float>>& hits = scRef->hitsAndFractions();
          // start in dump the ecal FED associated to the electron
          std::vector<std::pair<DetId, float>>::const_iterator itSChits = hits.begin();
          if (!dumpAllEcalFed_) {
            for (; itSChits != hits.end(); ++itSChits) {
              if ((*itSChits).first.subdetId() == EcalBarrel) {  // barrel part
                EBDetId idEBRaw((*itSChits).first);
                GlobalPoint point = GeometryCalo_->getPosition(idEBRaw);
                int hitFED = FEDNumbering::MINECALFEDID +
                             EcalMapping_->GetFED(double(point.eta()), double(point.phi()) * radTodeg);
                if (hitFED < FEDNumbering::MINECALFEDID || hitFED > FEDNumbering::MAXECALFEDID)
                  continue;

                LogDebug("SelectedElectronFEDListProducer")
                    << " electron hit detID Barrel " << (*itSChits).first.rawId() << " eta " << double(point.eta())
                    << " phi " << double(point.phi()) * radTodeg << " FED " << hitFED << std::endl;

                if (dumpSelectedEcalFed_) {
                  if (!fedList_.empty()) {
                    if (std::find(fedList_.begin(), fedList_.end(), hitFED) == fedList_.end())
                      fedList_.push_back(hitFED);  // in order not to duplicate info
                  } else
                    fedList_.push_back(hitFED);
                }
              } else if ((*itSChits).first.subdetId() == EcalEndcap) {  // endcap one
                EEDetId idEERaw((*itSChits).first);
                GlobalPoint point = GeometryCalo_->getPosition(idEERaw);
                int hitFED = FEDNumbering::MINECALFEDID +
                             EcalMapping_->GetFED(double(point.eta()), double(point.phi()) * radTodeg);
                if (hitFED < FEDNumbering::MINECALFEDID || hitFED > FEDNumbering::MAXECALFEDID)
                  continue;

                LogDebug("SelectedElectronFEDListProducer")
                    << " electron hit detID Endcap " << (*itSChits).first.rawId() << " eta " << double(point.eta())
                    << " phi " << double(point.phi()) * radTodeg << " FED " << hitFED << std::endl;
                if (dumpSelectedEcalFed_) {
                  if (!fedList_.empty()) {
                    if (std::find(fedList_.begin(), fedList_.end(), hitFED) == fedList_.end())
                      fedList_.push_back(hitFED);
                  } else
                    fedList_.push_back(hitFED);

                  // preshower hit for each ecal endcap hit
                  DetId tmpX =
                      (dynamic_cast<const EcalPreshowerGeometry*>(GeometryES_))->getClosestCellInPlane(point, 1);
                  ESDetId stripX = (tmpX == DetId(0)) ? ESDetId(0) : ESDetId(tmpX);
                  int hitFED =
                      ES_fedId_[(3 - stripX.zside()) / 2 - 1][stripX.plane() - 1][stripX.six() - 1][stripX.siy() - 1];
                  LogDebug("SelectedElectronFEDListProducer")
                      << " ES hit plane X (deiID) " << stripX.rawId() << " six " << stripX.six() << " siy "
                      << stripX.siy() << " plane " << stripX.plane() << " FED ID " << hitFED << std::endl;
                  if (hitFED < FEDNumbering::MINPreShowerFEDID || hitFED > FEDNumbering::MAXPreShowerFEDID)
                    continue;
                  if (hitFED < 0)
                    continue;
                  if (!fedList_.empty()) {
                    if (std::find(fedList_.begin(), fedList_.end(), hitFED) == fedList_.end())
                      fedList_.push_back(hitFED);
                  } else
                    fedList_.push_back(hitFED);

                  DetId tmpY =
                      (dynamic_cast<const EcalPreshowerGeometry*>(GeometryES_))->getClosestCellInPlane(point, 2);
                  ESDetId stripY = (tmpY == DetId(0)) ? ESDetId(0) : ESDetId(tmpY);
                  hitFED =
                      ES_fedId_[(3 - stripY.zside()) / 2 - 1][stripY.plane() - 1][stripY.six() - 1][stripY.siy() - 1];
                  if (hitFED < FEDNumbering::MINPreShowerFEDID || hitFED > FEDNumbering::MAXPreShowerFEDID)
                    continue;
                  LogDebug("SelectedElectronFEDListProducer")
                      << " ES hit plane Y (deiID) " << stripY.rawId() << " six " << stripY.six() << " siy "
                      << stripY.siy() << " plane " << stripY.plane() << " FED ID " << hitFED << std::endl;
                  if (hitFED < 0)
                    continue;
                  if (!fedList_.empty()) {
                    if (std::find(fedList_.begin(), fedList_.end(), hitFED) == fedList_.end())
                      fedList_.push_back(hitFED);
                  } else
                    fedList_.push_back(hitFED);
                }
              }  // end endcap
            }    // end loop on SC hit

            // check HCAL behind each hit
            if (dumpSelectedHCALFed_) {
              HBHERecHitCollection::const_iterator itHcalRecHit = hcalRecHitCollection->begin();
              for (; itHcalRecHit != hcalRecHitCollection->end(); ++itHcalRecHit) {
                HcalDetId recHitId(itHcalRecHit->id());
                const HcalGeometry* cellGeometry =
                    static_cast<const HcalGeometry*>(GeometryCalo_->getSubdetectorGeometry(recHitId));
                float dR = reco::deltaR(scRef->eta(),
                                        scRef->phi(),
                                        cellGeometry->getPosition(recHitId).eta(),
                                        cellGeometry->getPosition(recHitId).phi());
                if (dR <= dRHcalRegion_) {
                  const HcalElectronicsId electronicId = HcalReadoutMap_->lookup(recHitId);
                  int hitFED = electronicId.dccid() + FEDNumbering::MINHCALFEDID;
                  LogDebug("SelectedElectronFEDListProducer")
                      << " matched hcal recHit : HcalDetId " << recHitId << " HcalElectronicsId " << electronicId
                      << " dcc id " << electronicId.dccid() << " spigot " << electronicId.spigot() << " fiber channel "
                      << electronicId.fiberChanId() << " fiber index " << electronicId.fiberIndex() << std::endl;
                  if (hitFED < FEDNumbering::MINHCALFEDID || hitFED > FEDNumbering::MAXHCALFEDID)
                    continue;  //first eighteen feds are for HBHE
                  if (hitFED < 0)
                    continue;
                  if (!fedList_.empty()) {
                    if (std::find(fedList_.begin(), fedList_.end(), hitFED) == fedList_.end())
                      fedList_.push_back(hitFED);
                  } else
                    fedList_.push_back(hitFED);
                }
              }
            }  // End Hcal
          }    // End Ecal

          // get the electron track
          if (!dumpAllTrackerFed_) {
            //loop on the region
            if (dumpSelectedSiStripFed_) {
              double eta;
              double phi;
              if (*itElectronCollFlag) {
                eta = electron.gsfTrack()->eta();
                phi = electron.gsfTrack()->phi();
              } else {
                eta = electron.track()->eta();
                phi = electron.track()->phi();
              }
              for (uint32_t iCabling = 0; iCabling < SiStripCabling.size(); iCabling++) {
                SiStripRegionCabling::Position pos = StripRegionCabling_->position(iCabling);
                double dphi = fabs(pos.second - phi);
                if (dphi > acos(-1))
                  dphi = 2 * acos(-1) - dphi;
                double R = sqrt(pow(pos.first - eta, 2) + dphi * dphi);
                if (R - sqrt(pow(regionDimension_.first / 2, 2) + pow(regionDimension_.second / 2, 2)) > dRStripRegion_)
                  continue;
                //get vector of subdets within region
                const SiStripRegionCabling::RegionCabling regSubdets = SiStripCabling[iCabling];
                //cycle on subdets
                for (uint32_t idet = 0; idet < SiStripRegionCabling::ALLSUBDETS; idet++) {  //cicle between 1 and 4
                  //get vector of layers whin subdet of region
                  const SiStripRegionCabling::WedgeCabling regSubdetLayers = regSubdets[idet];  // at most 10 layers
                  for (uint32_t ilayer = 0; ilayer < SiStripRegionCabling::ALLLAYERS; ilayer++) {
                    //get map of vectors of feds withing the layer of subdet of region
                    const SiStripRegionCabling::ElementCabling fedVectorMap =
                        regSubdetLayers[ilayer];  // vector of the fed
                    SiStripRegionCabling::ElementCabling::const_iterator itFedMap = fedVectorMap.begin();
                    for (; itFedMap != fedVectorMap.end(); itFedMap++) {
                      for (uint32_t op = 0; op < (itFedMap->second).size(); op++) {
                        int hitFED = (itFedMap->second)[op].fedId();
                        if (hitFED < FEDNumbering::MINSiStripFEDID || hitFED > FEDNumbering::MAXSiStripFEDID)
                          continue;
                        LogDebug("SelectedElectronFEDListProducer") << " SiStrip (FedID) " << hitFED << std::endl;
                        if (!fedList_.empty()) {
                          if (std::find(fedList_.begin(), fedList_.end(), hitFED) == fedList_.end())
                            fedList_.push_back(hitFED);
                        } else
                          fedList_.push_back(hitFED);
                      }
                    }
                  }
                }
              }
            }  // end si strip
            if (dumpSelectedSiPixelFed_) {
              math::XYZVector momentum;
              if (*itElectronCollFlag)
                momentum = electron.gsfTrack()->momentum();
              else
                momentum = electron.track()->momentum();
              PixelRegion region(momentum, dPhiPixelRegion_, dEtaPixelRegion_, maxZPixelRegion_);
              PixelModule lowerBound(region.vector.phi() - region.dPhi, region.vector.eta() - region.dEta);
              PixelModule upperBound(region.vector.phi() + region.dPhi, region.vector.eta() + region.dEta);

              std::vector<PixelModule>::const_iterator itUp, itDn;
              if (lowerBound.Phi >= -M_PI && upperBound.Phi <= M_PI) {
                itDn = std::lower_bound(pixelModuleVector_.begin(), pixelModuleVector_.end(), lowerBound);
                itUp = std::upper_bound(pixelModuleVector_.begin(), pixelModuleVector_.end(), upperBound);
                pixelFedDump(itDn, itUp, region);
              } else {
                if (lowerBound.Phi < -M_PI)
                  lowerBound.Phi = lowerBound.Phi + 2 * M_PI;
                PixelModule phi_p(M_PI, region.vector.eta() - region.dEta);
                itDn = std::lower_bound(pixelModuleVector_.begin(), pixelModuleVector_.end(), lowerBound);
                itUp = std::upper_bound(pixelModuleVector_.begin(), pixelModuleVector_.end(), phi_p);
                pixelFedDump(itDn, itUp, region);

                if (upperBound.Phi < -M_PI)
                  upperBound.Phi = upperBound.Phi - 2 * M_PI;
                PixelModule phi_m(-M_PI, region.vector.eta() - region.dEta);
                itDn = std::lower_bound(pixelModuleVector_.begin(), pixelModuleVector_.end(), phi_m);
                itUp = std::upper_bound(pixelModuleVector_.begin(), pixelModuleVector_.end(), upperBound);
                pixelFedDump(itDn, itUp, region);
              }
            }
          }  // end tracker analysis
        }    // end loop on the electron candidate
      }      // end loop on the electron collection collection
    }        // end loop on the recoEcal candidate
  }          // end loop on the recoEcal candidate collection
  // add a set of chosen FED
  for (unsigned int iFed = 0; iFed < addThisSelectedFEDs_.size(); iFed++) {
    if (addThisSelectedFEDs_.at(iFed) == -1)
      continue;
    fedList_.push_back(addThisSelectedFEDs_.at(iFed));
  }

  // make the final raw data collection
  auto streamFEDRawProduct = std::make_unique<FEDRawDataCollection>();
  std::sort(fedList_.begin(), fedList_.end());
  std::vector<uint32_t>::const_iterator itfedList = fedList_.begin();
  for (; itfedList != fedList_.end(); ++itfedList) {
    LogDebug("SelectedElectronFEDListProducer") << " fed point " << *itfedList << "  ";
    const FEDRawData& data = rawdata->FEDData(*itfedList);
    if (data.size() > 0) {
      FEDRawData& fedData = streamFEDRawProduct->FEDData(*itfedList);
      fedData.resize(data.size());
      memcpy(fedData.data(), data.data(), data.size());
    }
  }

  iEvent.put(std::move(streamFEDRawProduct), outputLabelModule_);

  if (!fedList_.empty())
    fedList_.clear();
}

template <typename TEle, typename TCand>
void SelectedElectronFEDListProducer<TEle, TCand>::endJob() {
  LogDebug("SelectedElectronFEDListProducer") << " End of the Job " << std::endl;
}

template <typename TEle, typename TCand>
void SelectedElectronFEDListProducer<TEle, TCand>::pixelFedDump(std::vector<PixelModule>::const_iterator& itDn,
                                                                std::vector<PixelModule>::const_iterator& itUp,
                                                                const PixelRegion& region) {
  for (; itDn != itUp; ++itDn) {
    float zmodule = itDn->z - ((itDn->x - beamSpotPosition_.x()) * region.cosphi +
                               (itDn->y - beamSpotPosition_.y()) * region.sinphi) *
                                  region.atantheta;
    if (std::abs(zmodule) > region.maxZ)
      continue;
    int hitFED = itDn->Fed;
    if (hitFED < FEDNumbering::MINSiPixelFEDID || hitFED > FEDNumbering::MAXSiPixelFEDID)
      continue;
    LogDebug("SelectedElectronFEDListProducer")
        << " electron pixel hit " << itDn->DetId << " hitFED " << hitFED << std::endl;
    if (!fedList_.empty()) {
      if (std::find(fedList_.begin(), fedList_.end(), hitFED) == fedList_.end())
        fedList_.push_back(hitFED);
    } else
      fedList_.push_back(hitFED);
  }

  return;
}

template <typename TEle, typename TCand>
void SelectedElectronFEDListProducer<TEle, TCand>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<vector<edm::InputTag>>("electronTags", {edm::InputTag("hltEgammaGsfElectrons")});
  desc.add<vector<edm::InputTag>>("recoEcalCandidateTags", {edm::InputTag("hltL1EG25Ele27WP85GsfTrackIsoFilter")});
  desc.add<edm::FileInPath>("ESLookupTable", edm::FileInPath("EventFilter/ESDigiToRaw/data/ES_lookup_table.dat"));
  desc.add<edm::InputTag>("HBHERecHitTag", edm::InputTag("hltHbhereco"));
  desc.add<edm::InputTag>("beamSpotTag", edm::InputTag("hltOnlineBeamSpot"));
  desc.add<edm::InputTag>("rawDataTag", edm::InputTag("rawDataCollector"));
  desc.add<vector<int>>("addThisSelectedFEDs", {812, 813});
  desc.add<vector<int>>("isGsfElectronCollection", {true});
  desc.add<std::string>("outputLabelModule", "StreamElectronRawFed");
  desc.add<bool>("dumpSelectedSiPixelFed", true);
  desc.add<bool>("dumpSelectedSiStripFed", true);
  desc.add<bool>("dumpSelectedEcalFed", true);
  desc.add<bool>("dumpSelectedHCALFed", true);
  desc.add<double>("dPhiPixelRegion", 0.3);
  desc.add<double>("dEtaPixelRegion", 0.3);
  desc.add<double>("dRStripRegion", 0.3);
  desc.add<double>("dRHcalRegion", 0.3);
  desc.add<double>("maxZPixelRegion", 24);
  desc.add<bool>("dumpAllTrackerFed", false);
  desc.add<bool>("dumpAllEcalFed", false);
  desc.add<bool>("dumpAllHcalFed", false);

  descriptions.add(defaultModuleLabel<SelectedElectronFEDListProducer<TEle, TCand>>(), desc);
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
typedef SelectedElectronFEDListProducer<reco::Electron, reco::RecoEcalCandidate> SelectedElectronFEDListProducerGsf;
DEFINE_FWK_MODULE(SelectedElectronFEDListProducerGsf);
