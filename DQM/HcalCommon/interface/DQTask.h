#ifndef DQTask_h
#define DQTask_h

/*
 *	file:		DQTask.h
 *	Author:		VK
 *	Date:		13.10.2015
 */

#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"
#include "CondFormats/DataRecord/interface/HcalChannelQualityRcd.h"
#include "CondFormats/RunInfo/interface/RunInfo.h"
#include "CondFormats/DataRecord/interface/RunSummaryRcd.h"

#include "DQM/HcalCommon/interface/ContainerI.h"
#include "DQM/HcalCommon/interface/ContainerS.h"
#include "DQM/HcalCommon/interface/ContainerXXX.h"
#include "DQM/HcalCommon/interface/DQModule.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

namespace hcaldqm {
  enum UpdateFreq { fEvent = 0, f1LS = 1, f10LS = 2, f50LS = 3, f100LS = 4, nUpdateFreq = 5 };
  class DQTask : public DQMOneLumiEDAnalyzer<>, public DQModule {
  public:
    //	constructor
    DQTask(edm::ParameterSet const &);
    ~DQTask() override {}

    //	base inheritance to override from DQMEDAnalyzer
    void analyze(edm::Event const &, edm::EventSetup const &) override;
    void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
    void dqmBeginRun(edm::Run const &, edm::EventSetup const &) override;
    void dqmBeginLuminosityBlock(edm::LuminosityBlock const &, edm::EventSetup const &) override;
    void dqmEndLuminosityBlock(edm::LuminosityBlock const &, edm::EventSetup const &) override;

  protected:
    // protected funcs
    virtual void _resetMonitors(UpdateFreq);
    virtual void _process(edm::Event const &, edm::EventSetup const &) = 0;
    virtual bool _isApplicable(edm::Event const &) { return true; }
    virtual int _getCalibType(edm::Event const &);

    //	protected vars
    ContainerI _cEvsTotal;
    ContainerI _cEvsPerLS;
    ContainerI _cRunKeyVal;
    ContainerS _cRunKeyName;
    ContainerS _cProcessingTypeName;

    //	counters
    int _procLSs;

    //	container of quality masks from conddb
    ContainerXXX<uint32_t> _xQuality;
    //	vector of Electronics raw Ids of HCAL FEDs
    //	registered at cDAQ for the Run
    std::vector<uint32_t> _vcdaqEids;

    //	Tags and corresponding Tokens
    edm::InputTag _tagRaw;
    edm::EDGetTokenT<FEDRawDataCollection> _tokRaw;

    // Conditions and emap
    edm::ESGetToken<HcalDbService, HcalDbRecord> hcalDbServiceToken_;
    edm::ESGetToken<RunInfo, RunInfoRcd> runInfoToken_;
    edm::ESGetToken<HcalChannelQuality, HcalChannelQualityRcd> hcalChannelQualityToken_;

    edm::ESHandle<HcalDbService> _dbService;
    HcalElectronicsMap const *_emap = nullptr;
  };
}  // namespace hcaldqm

#endif
