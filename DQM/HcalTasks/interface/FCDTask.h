#ifndef FCDTask_h
#define FCDTask_h

/*
 *	file:			FCDTask.h
 *	Author:			Quan Wang
 *	Description:
 *		Task for ZDC Read out
 */

#include "DQM/HcalCommon/interface/ElectronicsMap.h"

class FCDTask : public DQMEDAnalyzer {
public:
  struct FCDChannel {
    int crate;
    int slot;
    int fiber;
    int fiberChannel;
  };

public:
  FCDTask(edm::ParameterSet const&);
  ~FCDTask() override {}

  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

protected:
  void analyze(edm::Event const&, edm::EventSetup const&) override;

  //	tags
  edm::InputTag _tagQIE10;
  edm::EDGetTokenT<QIE10DigiCollection> _tokQIE10;
  edm::ESGetToken<HcalDbService, HcalDbRecord> hcalDbServiceToken_;

  //	hcaldqm::Containers
  std::map<HcalElectronicsId, MonitorElement*> _cADC;
  std::map<HcalElectronicsId, MonitorElement*> _cADC_vs_TS;
  std::map<HcalElectronicsId, MonitorElement*> _cTDC;
  std::map<HcalElectronicsId, MonitorElement*> _cTDCTime;

  std::vector<HcalElectronicsId> _fcd_eids;
  std::vector<FCDChannel> _channels;
  HcalElectronicsMap const* _emap;
  hcaldqm::electronicsmap::ElectronicsMap _ehashmap;
};

#endif
