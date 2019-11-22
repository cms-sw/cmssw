#ifndef DQM_HcalTasks_UMNioTask_h
#define DQM_HcalTasks_UMNioTask_h

/*
 *	file:			LaserTask.h
 *	Author:			Viktor Khristenko
 *	Date:			16.10.2015
 */

#include "DQM/HcalCommon/interface/DQTask.h"
#include "DQM/HcalCommon/interface/Utilities.h"
#include "DQM/HcalCommon/interface/ElectronicsMap.h"
#include "DQM/HcalCommon/interface/ContainerXXX.h"
#include "DQM/HcalCommon/interface/Container1D.h"
#include "DQM/HcalCommon/interface/Container2D.h"
#include "DQM/HcalCommon/interface/ContainerSingle1D.h"
#include "DQM/HcalCommon/interface/ContainerSingle2D.h"
#include "DQM/HcalCommon/interface/ContainerSingleProf2D.h"
#include "DQM/HcalCommon/interface/ContainerProf1D.h"
#include "DQM/HcalCommon/interface/ContainerProf2D.h"
#include "FWCore/Framework/interface/Run.h"

class UMNioTask : public hcaldqm::DQTask {
public:
  UMNioTask(edm::ParameterSet const&);
  ~UMNioTask() override {}

  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void dqmEndRun(edm::Run const& r, edm::EventSetup const&) override {
    if (_ptype == hcaldqm::fLocal) {
      if (r.runAuxiliary().run() == 1)
        return;
    }
  }
  void dqmEndLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

protected:
  //	funcs
  void _process(edm::Event const&, edm::EventSetup const&) override;

  // Get index of a particular OrbitGapType in the vector, which is used as the value for filling the histogram
  int getOrbitGapIndex(uint8_t eventType, uint32_t laserType);

  std::vector<uint32_t> _eventtypes;

  //	tags and tokens
  edm::InputTag _taguMN;
  edm::InputTag _tagHBHE;
  edm::InputTag _tagHO;
  edm::InputTag _tagHF;
  edm::EDGetTokenT<HBHEDigiCollection> _tokHBHE;
  edm::EDGetTokenT<HODigiCollection> _tokHO;
  edm::EDGetTokenT<HFDigiCollection> _tokHF;
  edm::EDGetTokenT<HcalUMNioDigi> _tokuMN;

  //	cuts
  double _lowHBHE, _lowHO, _lowHF;

  //	emap
  hcaldqm::electronicsmap::ElectronicsMap _ehashmap;
  hcaldqm::filter::HashFilter _filter_uTCA;
  hcaldqm::filter::HashFilter _filter_VME;

  //	1D
  hcaldqm::ContainerSingle2D _cEventType;
  hcaldqm::ContainerSingle2D _cTotalCharge;
  hcaldqm::ContainerSingleProf2D _cTotalChargeProfile;
};
#endif
