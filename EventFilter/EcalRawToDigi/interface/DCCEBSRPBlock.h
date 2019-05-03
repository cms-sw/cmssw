#ifndef DCCEBSRPBLOCK_HH
#define DCCEBSRPBLOCK_HH

/*
 *\ Class DCCEBSRPBlock
 *
 * Class responsible for the SR flag unpacking in the EB.
 *
 * \file DCCEBSRPBlock.h
 *
 *
 * \author N. Almeida
 *
 */

#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "DCCSRPBlock.h"

#include <DataFormats/EcalDetId/interface/EcalDetIdCollections.h>
#include <DataFormats/EcalDigi/interface/EcalDigiCollections.h>
#include <DataFormats/EcalRawData/interface/EcalRawDataCollections.h>

class DCCEBSRPBlock : public DCCSRPBlock {

public:
  DCCEBSRPBlock(DCCDataUnpacker *u, EcalElectronicsMapper *m, DCCEventBlock *e,
                bool unpack);

  void updateCollectors() override;

protected:
  void addSRFlagToCollection() override;

  bool checkSrpIdAndNumbSRFlags() override;

  std::unique_ptr<EBSrFlagCollection> *ebSrFlagsDigis_;

  EcalTrigTowerDetId *pTTDetId_;
};

#endif
