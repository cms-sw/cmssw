#include "EventFilter/EcalRawToDigi/interface/DCCEBTCCBlock.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "EventFilter/EcalRawToDigi/interface/EcalElectronicsMapper.h"
#include "EventFilter/EcalRawToDigi/interface/DCCDataUnpacker.h"

DCCEBTCCBlock::DCCEBTCCBlock(DCCDataUnpacker* u, EcalElectronicsMapper* m, DCCEventBlock* e, bool unpack)
    : DCCTCCBlock(u, m, e, unpack) {
  blockLength_ = mapper_->getEBTCCBlockLength();
  expNumbTTs_ = TCC_EB_NUMBTTS;
}

void DCCEBTCCBlock::updateCollectors() { tps_ = unpacker_->ecalTpsCollection(); }

bool DCCEBTCCBlock::checkTccIdAndNumbTTs() {
  expTccId_ = mapper_->getActiveSM() + TCCID_SMID_SHIFT_EB;

  if (tccId_ != expTccId_) {
    if (!DCCDataUnpacker::silentMode_) {
      edm::LogWarning("IncorrectBlock") << "Error on event " << event_->l1A() << " with bx " << event_->bx()
                                        << " in fed " << mapper_->getActiveDCC() << "\n TCC id is " << tccId_
                                        << " while expected is " << expTccId_ << "\n TCC Block Skipped ...";
      //todo : add this to error colection
    }
    return false;
  }

  //Check number of TT Flags
  if (nTTs_ != expNumbTTs_) {
    if (!DCCDataUnpacker::silentMode_) {
      edm::LogWarning("IncorrectBlock") << "Unable to unpack TCC block for event " << event_->l1A() << " in fed "
                                        << mapper_->getActiveDCC() << "\n Number of TTs " << nTTs_
                                        << " is different from expected " << expNumbTTs_ << "\n TCC Block Skipped ...";
      //todo : add this to error colection
    }
    return false;
  }
  return true;
}

void DCCEBTCCBlock::addTriggerPrimitivesToCollection() {
  //point to trigger data
  data_++;

  unsigned int towersInPhi = EcalElectronicsMapper::kTowersInPhi;

  const uint16_t* tccP_ = reinterpret_cast<const uint16_t*>(data_);

  for (unsigned int i = 1; i <= expNumbTTs_; i++) {
    unsigned int theTT = i;

    if (NUMB_SM_EB_PLU_MIN <= mapper_->getActiveSM() && mapper_->getActiveSM() <= NUMB_SM_EB_PLU_MAX) {
      unsigned int u = (i - 1) % towersInPhi;
      u = towersInPhi - u;
      theTT = ((i - 1) / towersInPhi) * towersInPhi + u;
    }

    pTP_ = mapper_->getTPPointer(tccId_, theTT);
    for (unsigned int ns = 0; ns < nTSamples_; ns++, tccP_++) {
      pTP_->setSampleValue(ns, (*tccP_));
      (*tps_)->push_back(*pTP_);
    }
  }
}
