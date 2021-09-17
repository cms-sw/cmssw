/** \class EcalTPSkimmer
 *   produce a subset of TP information
 *
 *  \author Federico Ferri, CEA/Saclay Irfu/SPP
 *
 **/

#include "RecoLocalCalo/EcalRecProducers/plugins/EcalTPSkimmer.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"

EcalTPSkimmer::EcalTPSkimmer(const edm::ParameterSet& ps) {
  skipModule_ = ps.getParameter<bool>("skipModule");

  doBarrel_ = ps.getParameter<bool>("doBarrel");
  doEndcap_ = ps.getParameter<bool>("doEndcap");

  chStatusToSelectTP_ = ps.getParameter<std::vector<uint32_t> >("chStatusToSelectTP");

  tpOutputCollection_ = ps.getParameter<std::string>("tpOutputCollection");
  tpInputToken_ = consumes<EcalTrigPrimDigiCollection>(ps.getParameter<edm::InputTag>("tpInputCollection"));
  ttMapToken_ = esConsumes<EcalTrigTowerConstituentsMap, IdealGeometryRecord>();
  if (not skipModule_) {
    chStatusToken_ = esConsumes<EcalChannelStatus, EcalChannelStatusRcd>();
  }
  produces<EcalTrigPrimDigiCollection>(tpOutputCollection_);
}

EcalTPSkimmer::~EcalTPSkimmer() {}

void EcalTPSkimmer::produce(edm::Event& evt, const edm::EventSetup& es) {
  insertedTP_.clear();

  using namespace edm;

  ttMap_ = es.getHandle(ttMapToken_);

  // collection of rechits to put in the event
  auto tpOut = std::make_unique<EcalTrigPrimDigiCollection>();

  if (skipModule_) {
    evt.put(std::move(tpOut), tpOutputCollection_);
    return;
  }

  edm::ESHandle<EcalChannelStatus> chStatus = es.getHandle(chStatusToken_);

  edm::Handle<EcalTrigPrimDigiCollection> tpIn;
  evt.getByToken(tpInputToken_, tpIn);

  if (doBarrel_) {
    EcalChannelStatusMap::const_iterator chit;
    uint16_t code = 0;
    for (int i = 0; i < EBDetId::kSizeForDenseIndexing; ++i) {
      if (!EBDetId::validDenseIndex(i))
        continue;
      EBDetId id = EBDetId::detIdFromDenseIndex(i);
      chit = chStatus->find(id);
      // check if the channel status means TP to be kept
      if (chit != chStatus->end()) {
        code = (*chit).getStatusCode();
        if (std::find(chStatusToSelectTP_.begin(), chStatusToSelectTP_.end(), code) != chStatusToSelectTP_.end()) {
          // retrieve the TP DetId
          EcalTrigTowerDetId ttDetId(((EBDetId)id).tower());
          // insert the TP if not done already
          if (!alreadyInserted(ttDetId))
            insertTP(ttDetId, tpIn, *tpOut);
        }
      } else {
        edm::LogError("EcalDetIdToBeRecoveredProducer") << "No channel status found for xtal " << id.rawId()
                                                        << "! something wrong with EcalChannelStatus in your DB? ";
      }
    }
  }

  if (doEndcap_) {
    EcalChannelStatusMap::const_iterator chit;
    uint16_t code = 0;
    for (int i = 0; i < EEDetId::kSizeForDenseIndexing; ++i) {
      if (!EEDetId::validDenseIndex(i))
        continue;
      EEDetId id = EEDetId::detIdFromDenseIndex(i);
      chit = chStatus->find(id);
      // check if the channel status means TP to be kept
      if (chit != chStatus->end()) {
        code = (*chit).getStatusCode();
        if (std::find(chStatusToSelectTP_.begin(), chStatusToSelectTP_.end(), code) != chStatusToSelectTP_.end()) {
          // retrieve the TP DetId
          EcalTrigTowerDetId ttDetId = ttMap_->towerOf(id);
          // insert the TP if not done already
          if (!alreadyInserted(ttDetId))
            insertTP(ttDetId, tpIn, *tpOut);
        }
      } else {
        edm::LogError("EcalDetIdToBeRecoveredProducer") << "No channel status found for xtal " << id.rawId()
                                                        << "! something wrong with EcalChannelStatus in your DB? ";
      }
    }
  }

  // put the collection of reconstructed hits in the event
  LogInfo("EcalTPSkimmer") << "total # of TP inserted: " << tpOut->size();

  evt.put(std::move(tpOut), tpOutputCollection_);
}

bool EcalTPSkimmer::alreadyInserted(EcalTrigTowerDetId ttId) { return (insertedTP_.find(ttId) != insertedTP_.end()); }

void EcalTPSkimmer::insertTP(EcalTrigTowerDetId ttId,
                             edm::Handle<EcalTrigPrimDigiCollection>& tpIn,
                             EcalTrigPrimDigiCollection& tpOut) {
  EcalTrigPrimDigiCollection::const_iterator tpIt = tpIn->find(ttId);
  if (tpIt != tpIn->end()) {
    tpOut.push_back(*tpIt);
    insertedTP_.insert(ttId);
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(EcalTPSkimmer);
