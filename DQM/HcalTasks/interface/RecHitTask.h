#ifndef RecHitTask_h
#define RecHitTask_h

/**
 *	module:			RecHitTask.h
 *	Author:			VK
 *	Description:	
 *		HCAL RECO Data Tier Evaluation
 *
 *	Online:
 *	Offline:
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

class RecHitTask : public hcaldqm::DQTask {
public:
  RecHitTask(edm::ParameterSet const &);
  ~RecHitTask() override {}

  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  std::shared_ptr<hcaldqm::Cache> globalBeginLuminosityBlock(edm::LuminosityBlock const &,
                                                             edm::EventSetup const &) const override;
  void globalEndLuminosityBlock(edm::LuminosityBlock const &, edm::EventSetup const &) override;

protected:
  void _process(edm::Event const &, edm::EventSetup const &) override;
  void _resetMonitors(hcaldqm::UpdateFreq) override;

  edm::InputTag _tagHBHE;
  edm::InputTag _tagHO;
  edm::InputTag _tagHF;
  edm::InputTag _tagPreHF;
  bool _hfPreRecHitsAvailable;
  edm::EDGetTokenT<HBHERecHitCollection> _tokHBHE;
  edm::EDGetTokenT<HORecHitCollection> _tokHO;
  edm::EDGetTokenT<HFRecHitCollection> _tokHF;
  edm::EDGetTokenT<HFPreRecHitCollection> _tokPreHF;
  edm::ESGetToken<HcalDbService, HcalDbRecord> hcalDbServiceToken_;

  double _cutE_HBHE, _cutE_HO, _cutE_HF;
  double _thresh_unihf;

  //	hashes/FED vectors
  std::vector<uint32_t> _vhashFEDs;

  //	flag vectors
  std::vector<hcaldqm::flag::Flag> _vflags;
  enum RecoFlag { fUni = 0, fTCDS = 1, fUnknownIds = 2, nRecoFlag = 3 };

  //	emap
  hcaldqm::electronicsmap::ElectronicsMap _ehashmap;

  //	Filters
  hcaldqm::filter::HashFilter _filter_VME;
  hcaldqm::filter::HashFilter _filter_uTCA;
  hcaldqm::filter::HashFilter _filter_FEDsVME;
  hcaldqm::filter::HashFilter _filter_FEDsuTCA;
  hcaldqm::filter::HashFilter _filter_HF;

  //	Energy. Just filling. No Summary Generation
  hcaldqm::Container1D _cEnergy_Subdet;
  hcaldqm::ContainerProf1D _cEnergyvsieta_Subdet;    //	online only!
  hcaldqm::ContainerProf1D _cEnergyvsiphi_SubdetPM;  // online only!
  hcaldqm::ContainerProf2D _cEnergy_depth;
  hcaldqm::ContainerProf1D _cEnergyvsLS_SubdetPM;  // online only!
  hcaldqm::ContainerProf1D _cEnergyvsBX_SubdetPM;  // online only

  //	Timing vs Energy. No Summary Generation
  hcaldqm::Container2D _cTimingvsEnergy_SubdetPM;

  //	Timing. HBHE Partition is used for TCDS shift monitoring
  hcaldqm::Container1D _cTimingCut_SubdetPM;
  hcaldqm::Container1D _cTimingCut_HBHEPartition;
  hcaldqm::ContainerProf2D _cTimingCut_FEDuTCA;
  hcaldqm::ContainerProf2D _cTimingCut_ElectronicsuTCA;
  hcaldqm::ContainerProf2D _cTimingCut_depth;
  hcaldqm::ContainerProf1D _cTimingCutvsLS_FED;
  hcaldqm::ContainerProf1D _cTimingCutvsLS_SubdetPM;
  hcaldqm::ContainerProf1D _cTimingCutvsieta_Subdet;    //	online only
  hcaldqm::ContainerProf1D _cTimingCutvsiphi_SubdetPM;  //	online only
  hcaldqm::ContainerProf1D _cTimingCutvsBX_SubdetPM;    // online only

  //	Occupancy w/o a cut. Used for checking missing channels
  hcaldqm::Container2D _cOccupancy_depth;
  hcaldqm::Container2D _cOccupancy_FEDuTCA;
  hcaldqm::Container2D _cOccupancy_ElectronicsuTCA;
  hcaldqm::ContainerProf1D _cOccupancyvsLS_Subdet;
  hcaldqm::Container1D _cOccupancyvsiphi_SubdetPM;  // online only
  hcaldqm::Container1D _cOccupancyvsieta_Subdet;    //	online only

  //	Occupancy w/ a Cut.
  hcaldqm::Container2D _cOccupancyCut_FEDuTCA;
  hcaldqm::Container2D _cOccupancyCut_ElectronicsuTCA;
  hcaldqm::ContainerProf1D _cOccupancyCutvsLS_Subdet;  // online only
  hcaldqm::Container2D _cOccupancyCut_depth;
  hcaldqm::Container1D _cOccupancyCutvsiphi_SubdetPM;      // online only
  hcaldqm::Container1D _cOccupancyCutvsieta_Subdet;        // online only
  hcaldqm::ContainerProf1D _cOccupancyCutvsBX_Subdet;      // online only!
  hcaldqm::Container2D _cOccupancyCutvsiphivsLS_SubdetPM;  // online only
  hcaldqm::ContainerXXX<uint32_t> _xUniHF, _xUni;

  // QIE10 dual anode histograms
  hcaldqm::Container2D _cDAAsymmetryVsCharge_SubdetPM;
  hcaldqm::ContainerProf2D _cDAAsymmetryMean_cut_depth;
  hcaldqm::Container1D _cDAAsymmetry_cut_SubdetPM;

  //	tracks the unknown ids
  MonitorElement *meUnknownIds1LS;
  bool _unknownIdsPresent;

  std::vector<HcalGenericDetId> _gids;     // online only
  hcaldqm::Container2D _cSummaryvsLS_FED;  // online only!
  hcaldqm::ContainerSingle2D _cSummaryvsLS;
};

#endif
