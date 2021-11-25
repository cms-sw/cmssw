#ifndef TPTask_h
#define TPTask_h

/**
 *	file:
 *	Author:
 *	Description:
 */

#include "DQM/HcalCommon/interface/DQTask.h"
#include "DQM/HcalCommon/interface/Utilities.h"
#include "DQM/HcalCommon/interface/HashFilter.h"
#include "DQM/HcalCommon/interface/Container1D.h"
#include "DQM/HcalCommon/interface/Container2D.h"
#include "DQM/HcalCommon/interface/ContainerProf1D.h"
#include "DQM/HcalCommon/interface/ContainerProf2D.h"
#include "DQM/HcalCommon/interface/ContainerSingle1D.h"
#include "DQM/HcalCommon/interface/ContainerSingle2D.h"
#include "DQM/HcalCommon/interface/ContainerSingleProf2D.h"
#include "DQM/HcalCommon/interface/ElectronicsMap.h"

class TPTask : public hcaldqm::DQTask {
public:
  TPTask(edm::ParameterSet const &);
  ~TPTask() override {}

  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  std::shared_ptr<hcaldqm::Cache> globalBeginLuminosityBlock(edm::LuminosityBlock const &,
                                                             edm::EventSetup const &) const override;
  void globalEndLuminosityBlock(edm::LuminosityBlock const &, edm::EventSetup const &) override;

protected:
  void _process(edm::Event const &, edm::EventSetup const &) override;
  void _resetMonitors(hcaldqm::UpdateFreq) override;

  edm::InputTag _tagData;
  edm::InputTag _tagDataL1Rec;
  edm::InputTag _tagEmul;
  edm::InputTag _tagEmulNoTDCCut;
  edm::EDGetTokenT<HcalTrigPrimDigiCollection> _tokData;
  edm::EDGetTokenT<HcalTrigPrimDigiCollection> _tokDataL1Rec;
  edm::EDGetTokenT<HcalTrigPrimDigiCollection> _tokEmul;
  edm::EDGetTokenT<HcalTrigPrimDigiCollection> _tokEmulNoTDCCut;
  edm::ESGetToken<HcalDbService, HcalDbRecord> hcalDbServiceToken_;

  //	flag vector
  std::vector<hcaldqm::flag::Flag> _vflags;
  enum TPFlag { fEtMsm = 0, fDataMsn = 1, fEmulMsn = 2, fUnknownIds = 3, fSentRecL1Msm = 4, nTPFlag = 5 };

  //	switches/cuts/etc...
  bool _skip1x1;
  int _cutEt;
  double _thresh_EtMsmRate_high, _thresh_EtMsmRate_low, _thresh_FGMsmRate_high, _thresh_FGMsmRate_low, _thresh_DataMsn,
      _thresh_EmulMsn;
  std::vector<bool> _vFGBitsReady;

  //	hashes/FEDs vectors
  std::vector<uint32_t> _vhashFEDs;

  //	emap
  hcaldqm::electronicsmap::ElectronicsMap _ehashmap;

  //	Filters
  hcaldqm::filter::HashFilter _filter_VME;
  hcaldqm::filter::HashFilter _filter_uTCA;
  hcaldqm::filter::HashFilter _filter_depth0;

  //	Et/FG
  hcaldqm::Container1D _cEtData_TTSubdet;
  hcaldqm::Container1D _cEtEmul_TTSubdet;
  hcaldqm::Container2D _cEtCorr_TTSubdet;
  hcaldqm::Container2D _cSOIEtCorr_TTSubdet;
  hcaldqm::Container2D _cSOIEtCorrEmulL1_TTSubdet;
  hcaldqm::Container2D _cEtCorr2x3_TTSubdet;  //	online only
  hcaldqm::Container2D _cFGCorr_TTSubdet[hcaldqm::constants::NUM_FGBITS];
  hcaldqm::ContainerProf1D _cEtCutDatavsLS_TTSubdet;  // online only!
  hcaldqm::ContainerProf1D _cEtCutEmulvsLS_TTSubdet;  // online only!
  hcaldqm::Container2D _cEtCutDatavsBX_TTSubdet;      // online only!
  hcaldqm::ContainerProf1D _cEtCutEmulvsBX_TTSubdet;  // online only!

  hcaldqm::ContainerProf2D _cEtData_ElectronicsuTCA;
  hcaldqm::ContainerProf2D _cEtEmul_ElectronicsuTCA;

  //	depth like
  hcaldqm::ContainerSingleProf2D _cEtData_depthlike;
  hcaldqm::ContainerSingleProf2D _cEtEmul_depthlike;
  hcaldqm::ContainerSingleProf2D _cEtCutData_depthlike;
  hcaldqm::ContainerSingleProf2D _cEtCutEmul_depthlike;

  //	Et Correlation Ratio
  hcaldqm::ContainerProf2D _cEtCorrRatio_ElectronicsuTCA;
  hcaldqm::ContainerSingleProf2D _cEtCorrRatio_depthlike;
  hcaldqm::ContainerProf1D _cEtCorrRatiovsLS_TTSubdet;  // online only!
  hcaldqm::ContainerProf1D _cEtCorrRatiovsBX_TTSubdet;  // online only!

  //	Occupancies
  hcaldqm::Container2D _cOccupancyData_ElectronicsuTCA;
  hcaldqm::Container2D _cOccupancyEmul_ElectronicsuTCA;

  hcaldqm::Container2D _cOccupancyCutData_ElectronicsuTCA;
  hcaldqm::Container2D _cOccupancyCutEmul_ElectronicsuTCA;

  //	depth like
  hcaldqm::ContainerSingle2D _cOccupancyData_depthlike;
  hcaldqm::ContainerSingle2D _cOccupancyEmul_depthlike;
  hcaldqm::ContainerSingle2D _cOccupancyCutData_depthlike;
  hcaldqm::ContainerSingle2D _cOccupancyCutEmul_depthlike;

  //	2x3 occupancies just in case
  hcaldqm::ContainerSingle2D _cOccupancyData2x3_depthlike;  // online only!
  hcaldqm::ContainerSingle2D _cOccupancyEmul2x3_depthlike;  // online only!

  //	Mismatches: Et and FG
  hcaldqm::Container2D _cEtMsm_ElectronicsuTCA;
  hcaldqm::Container2D _cFGMsm_ElectronicsuTCA;
  hcaldqm::ContainerSingle2D _cEtMsm_depthlike;
  hcaldqm::ContainerSingle2D _cFGMsm_depthlike;
  hcaldqm::ContainerProf1D _cEtMsmvsLS_TTSubdet;       // online only
  hcaldqm::ContainerProf1D _cEtMsmRatiovsLS_TTSubdet;  // online only
  hcaldqm::ContainerProf1D _cEtMsmvsBX_TTSubdet;       // online only
  hcaldqm::ContainerProf1D _cEtMsmRatiovsBX_TTSubdet;  // online only

  // Mismatches: data sent vs received
  hcaldqm::ContainerSingle2D _cEtMsm_uHTR_L1T_depthlike;
  hcaldqm::ContainerSingle1D _cEtMsm_uHTR_L1T_LS;

  //	Missing Data w.r.t. Emulator
  hcaldqm::Container2D _cMsnData_ElectronicsuTCA;
  hcaldqm::ContainerSingle2D _cMsnData_depthlike;
  hcaldqm::ContainerProf1D _cMsnDatavsLS_TTSubdet;     //	online only
  hcaldqm::ContainerProf1D _cMsnCutDatavsLS_TTSubdet;  // online only
  hcaldqm::ContainerProf1D _cMsnDatavsBX_TTSubdet;     // online only
  hcaldqm::ContainerProf1D _cMsnCutDatavsBX_TTSubdet;  // online only

  //	Missing Emulator w.r.t. Data
  hcaldqm::Container2D _cMsnEmul_ElectronicsuTCA;
  hcaldqm::ContainerSingle2D _cMsnEmul_depthlike;
  hcaldqm::ContainerProf1D _cMsnEmulvsLS_TTSubdet;     // online only
  hcaldqm::ContainerProf1D _cMsnCutEmulvsLS_TTSubdet;  //	online only
  hcaldqm::ContainerProf1D _cMsnEmulvsBX_TTSubdet;     // online only
  hcaldqm::ContainerProf1D _cMsnCutEmulvsBX_TTSubdet;  // online only

  //	Occupancy vs BX and LS
  hcaldqm::ContainerProf1D _cOccupancyDatavsBX_TTSubdet;     // online only
  hcaldqm::ContainerProf1D _cOccupancyEmulvsBX_TTSubdet;     // online only
  hcaldqm::ContainerProf1D _cOccupancyCutDatavsBX_TTSubdet;  // online only
  hcaldqm::ContainerProf1D _cOccupancyCutEmulvsBX_TTSubdet;  // online only
  hcaldqm::ContainerProf1D _cOccupancyDatavsLS_TTSubdet;     // online only
  hcaldqm::ContainerProf1D _cOccupancyEmulvsLS_TTSubdet;     // online only
  hcaldqm::ContainerProf1D _cOccupancyCutDatavsLS_TTSubdet;  // online only
  hcaldqm::ContainerProf1D _cOccupancyCutEmulvsLS_TTSubdet;  // online only

  //	track unknown ids
  MonitorElement *meUnknownIds1LS;
  bool _unknownIdsPresent;

  hcaldqm::Container2D _cSummaryvsLS_FED;    // online only
  hcaldqm::ContainerSingle2D _cSummaryvsLS;  // online only
  hcaldqm::ContainerXXX<uint32_t> _xEtMsm, _xFGMsm, _xNumCorr, _xDataMsn, _xDataTotal, _xEmulMsn, _xEmulTotal,
      _xSentRecL1Msm;

  // Temporary storage for occupancy with and without HF TDC cut
  hcaldqm::ContainerSingle2D _cOccupancy_HF_depth, _cOccupancyNoTDC_HF_depth;
  hcaldqm::ContainerSingle1D _cOccupancy_HF_ieta, _cOccupancyNoTDC_HF_ieta;

  // Container storing matched sent-received TPs
  std::vector<std::pair<HcalTriggerPrimitiveDigi, HcalTriggerPrimitiveDigi> > _vEmulTPDigis_SentRec;
  std::vector<std::pair<HcalTriggerPrimitiveDigi, HcalTriggerPrimitiveDigi> > _vTPDigis_SentRec;
};

#endif
