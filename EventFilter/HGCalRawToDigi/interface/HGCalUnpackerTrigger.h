/****************************************************************************
 *
 * A top level class dispatching unpacking of HGCal raw trigger data 
 * to specialized classes.
 * 
 * Authors: Jeremi Niedziela, Lovisa Rygaard, Yulun Miao
 *   
 *
 ****************************************************************************/

#ifndef EventFilter_HGCalRawToDigi_HGCalUnpackerTrigger_h
#define EventFilter_HGCalRawToDigi_HGCalUnpackerTrigger_h

#include "DataFormats/FEDRawData/interface/RawDataBuffer.h"
#include "CondFormats/HGCalObjects/interface/HGCalTriggerConfiguration.h"
#include "DataFormats/HGCalDigi/interface/HGCalTriggerDefinitions.h"
#include "DataFormats/HGCalDigi/interface/HGCalDigiTriggerHost.h"
#include "DataFormats/HGCalDigi/interface/HGCalECONTPacketInfoHost.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingModuleIndexerTrigger.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/HGCalDigi/interface/HGCalFEDTriggerPacketInfoHost.h"

class HGCalUnpackerTrigger {
public:
  HGCalUnpackerTrigger() {}
  bool parseFEDData(unsigned fedId,
                    const RawFragmentWrapper& fed_data,
                    const HGCalTriggerConfiguration& config,
                    const HGCalMappingModuleIndexerTrigger& moduleIndexer,
                    hgcaldigi::HGCalDigiTriggerHost& digisTrigger,
                    hgcaldigi::HGCalFEDTriggerPacketInfoHost& fedTriggerPacketInfo,
                    hgcaldigi::HGCalECONTPacketInfoHost& econtPacketInfo);
  bool parseTDAQBlock();

private:
  // std::unique_ptr<HGCalUnpackerTriggerSpecialization> unpackerSpecialization_;
};

#endif
