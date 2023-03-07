#ifndef DQM_HcalTasks_HFRaddamTask_h
#define DQM_HcalTasks_HFRaddamTask_h

/*
 *	file:			RadDamTask.h
 *	Author:			Viktor Khristenko
 *	Date:			16.10.2015
 */

#include "DQM/HcalCommon/interface/DQTask.h"
#include "DQM/HcalCommon/interface/Utilities.h"
#include "DQM/HcalCommon/interface/Container1D.h"
#include "DQM/HcalCommon/interface/Container2D.h"
#include "DQM/HcalCommon/interface/ContainerProf1D.h"
#include "DQM/HcalCommon/interface/ContainerProf2D.h"
#include "DQM/HcalCommon/interface/ContainerSingle1D.h"

class HFRaddamTask : public hcaldqm::DQTask {
public:
  HFRaddamTask(edm::ParameterSet const&);
  ~HFRaddamTask() override {}

  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

protected:
  //	funcs
  void _process(edm::Event const&, edm::EventSetup const&) override;
  bool _isApplicable(edm::Event const&) override;

  //	Tags and Tokens
  edm::InputTag _tagHF;
  edm::InputTag _taguMN;
  edm::EDGetTokenT<QIE10DigiCollection> _tokHF;
  edm::EDGetTokenT<HcalUMNioDigi> _tokuMN;

  //	vector of Detector Ids for RadDam
  std::vector<HcalDetId> _vDetIds;

  //	Cuts

  //	Compact

  //	1D
  std::vector<hcaldqm::ContainerSingle1D> _vcShape;
};

#endif
