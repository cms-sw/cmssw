#ifndef DCCEBEVENTBLOCK_HH
#define DCCEBEVENTBLOCK_HH

/*
 *\ Class DCCEBEventBlock
 *
 * Specialization of the DCCEventBlock for the EB
 * The class instintes the DCCTowerBlock, DCCEBTCBlock and DCCEBSRPBlock
 *unpacking classes
 *
 * \file DCCEBEventBlock.h
 *
 *
 * \author N. Almeida
 *
 *
 */

#include "DCCEventBlock.h"
#include "DCCRawDataDefinitions.h"
#include <DataFormats/EcalDigi/interface/EcalDigiCollections.h>
#include <DataFormats/EcalRawData/interface/EcalRawDataCollections.h>

class DCCEBEventBlock : public DCCEventBlock {

public:
  DCCEBEventBlock(DCCDataUnpacker *u, EcalElectronicsMapper *m, bool hU,
                  bool srpU, bool tccU, bool feU, bool memU,
                  bool forceToKeepFRdata);

  void unpack(const uint64_t *buffer, size_t bufferSize,
              unsigned int expFedId) override;

protected:
  int unpackTCCBlocks() override;
};

#endif
