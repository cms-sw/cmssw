#ifndef DCCEESRPBLOCK_HH
#define DCCEESRPBLOCK_HH

/*
 *\ Class DCCEESRPBlock
 *
 * Class responsible for the SR flag unpacking in the EE.
 *
 * \file DCCEESRPBlock.h
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

class DCCEESRPBlock : public DCCSRPBlock {

public:
  DCCEESRPBlock(DCCDataUnpacker *u, EcalElectronicsMapper *m, DCCEventBlock *e,
                bool unpack);

  void updateCollectors() override;

protected:
  void addSRFlagToCollection() override;

  bool checkSrpIdAndNumbSRFlags() override;

  std::unique_ptr<EESrFlagCollection> *eeSrFlagsDigis_;

  EcalScDetId *pSCDetId_;
};

#endif
