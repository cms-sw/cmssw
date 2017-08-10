#ifndef DQM_L1TMonitorClient_L1TStage2RatioClient_H
#define DQM_L1TMonitorClient_L1TStage2RatioClient_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"

class L1TStage2RatioClient: public DQMEDHarvester
{
  public:

    L1TStage2RatioClient(const edm::ParameterSet&);
    virtual ~L1TStage2RatioClient();
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  protected:

    virtual void dqmEndJob(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter)override;
    virtual void dqmEndLuminosityBlock(DQMStore::IBooker& ibooker,DQMStore::IGetter& igetter,const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& c) override;

  private:

    void book(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter);
    void processHistograms(DQMStore::IGetter& igetter);

    std::string monitorDir_;
    std::string inputNum_;
    std::string inputDen_;
    std::string ratioName_;
    std::string ratioTitle_;
    std::string yAxisTitle_;
    bool binomialErr_;
    std::vector<int> ignoreBin_;

    MonitorElement* ratioME_;
};

#endif

