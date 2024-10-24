#ifndef ZDCQIE10Task_h
#define ZDCQIE10Task_h

/*
 *	file:			ZDCQIE10Task.h
 *	Author:			Quan Wang
 *	Description:
 *		Task for ZDC Read out
 */

#include "DQM/HcalCommon/interface/ElectronicsMap.h"
#include "DQM/HcalCommon/interface/DQTask.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/L1TObjects/interface/CaloParams.h"
#include "CondFormats/DataRecord/interface/L1TCaloParamsRcd.h"

#include "DataFormats/L1TCalorimeter/interface/CaloTower.h"
#include "DataFormats/L1TCalorimeter/interface/CaloCluster.h"
#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"
#include "CondFormats/HcalObjects/interface/HcalLongRecoParams.h"
#include "CondFormats/HcalObjects/interface/HcalLongRecoParam.h"

class ZDCQIE10Task : public hcaldqm::DQTask {
public:
  ZDCQIE10Task(edm::ParameterSet const &);
  ~ZDCQIE10Task() override {}

  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

protected:
  void _process(edm::Event const &, edm::EventSetup const &) override;

  //	tags
  edm::InputTag _tagQIE10;
  edm::InputTag sumTag;
  edm::EDGetTokenT<QIE10DigiCollection> _tokQIE10;
  edm::ESGetToken<HcalDbService, HcalDbRecord> hcalDbServiceToken_;
  edm::EDGetToken sumToken_;
  edm::ESGetToken<HcalTopology, HcalRecNumberingRecord> htopoToken_;
  edm::ESGetToken<HcalLongRecoParams, HcalLongRecoParamsRcd> paramsToken_;

  //emap
  hcaldqm::electronicsmap::ElectronicsMap _ehashmap;  // online only

  //	hcaldqm::Containers
  std::map<uint32_t, MonitorElement *> _cADC_EChannel;
  std::map<uint32_t, MonitorElement *> _cADC_vs_TS_EChannel;
  std::map<uint32_t, MonitorElement *> _cDigiSize_Crate;
  std::map<uint32_t, MonitorElement *> _cDigiSize_FED;
  std::map<uint32_t, MonitorElement *> _cADC_PM;
  std::map<uint32_t, MonitorElement *> _cADC_vs_TS_PM;
  std::map<uint32_t, MonitorElement *> _cOccupancy_FEDuTCA;
  std::map<uint32_t, MonitorElement *> _cOccupancy_ElectronicsuTCA;
  std::map<uint32_t, MonitorElement *> _cOccupancy_Crate;
  std::map<uint32_t, MonitorElement *> _cOccupancy_CrateSlot;
  std::map<uint32_t, MonitorElement *> _cZDC_SUMS;
  std::map<uint32_t, MonitorElement *> _cZDC_BXSUMS;
  std::map<uint32_t, MonitorElement *> _cZDC_BX_EmuSUMS;
  std::map<uint32_t, MonitorElement *> _cZDC_CapIDS;
  std::map<uint32_t, MonitorElement *> _cfC_EChannel;
  std::map<uint32_t, MonitorElement *> _cTDC_EChannel;
  std::map<uint32_t, MonitorElement *> _cfC_vs_TS_EChannel;
  std::map<uint32_t, MonitorElement *> _cZDC_HAD_TM;
  std::map<uint32_t, MonitorElement *> _cZDC_EM_TM;

  std::unique_ptr<HcalLongRecoParams> longRecoParams_;
};

#endif
