/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors:
*   Jan Kašpar (jan.kaspar@gmail.com)
*   Rafał Leszko (rafal.leszko@gmail.com)
*
****************************************************************************/

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/TotemDigi/interface/TotemFEDInfo.h"
#include "DataFormats/TotemDigi/interface/TotemTriggerCounters.h"

#include <string>

//----------------------------------------------------------------------------------------------------
 
class TotemDAQTriggerDQMSource: public DQMEDAnalyzer
{
  public:
    TotemDAQTriggerDQMSource(const edm::ParameterSet& ps);
    virtual ~TotemDAQTriggerDQMSource();
  
  protected:
    void dqmBeginRun(edm::Run const &, edm::EventSetup const &) override;
    void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
    void analyze(edm::Event const& e, edm::EventSetup const& eSetup);
    void beginLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& eSetup);
    void endLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& eSetup);
    void endRun(edm::Run const& run, edm::EventSetup const& eSetup);

  private:
    unsigned int verbosity;

    edm::EDGetTokenT<std::vector<TotemFEDInfo>> tokenFEDInfo;
    edm::EDGetTokenT<TotemTriggerCounters> tokenTriggerCounters;
    
    MonitorElement *daq_bx_diff;
    MonitorElement *daq_event_bx_diff;
    MonitorElement *daq_event_bx_diff_vs_fed;
    MonitorElement *daq_trigger_bx_diff;

    MonitorElement *trigger_type;
    MonitorElement *trigger_event_num;
    MonitorElement *trigger_bunch_num;
    MonitorElement *trigger_src_id;
    MonitorElement *trigger_orbit_num;
    MonitorElement *trigger_revision_num;
    MonitorElement *trigger_run_num;
    MonitorElement *trigger_trigger_num;
    MonitorElement *trigger_inhibited_triggers_num;
    MonitorElement *trigger_input_status_bits;
};

//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------

using namespace std;
using namespace edm;

//----------------------------------------------------------------------------------------------------

TotemDAQTriggerDQMSource::TotemDAQTriggerDQMSource(const edm::ParameterSet& ps) :
  verbosity(ps.getUntrackedParameter<unsigned int>("verbosity", 0))
{
  tokenFEDInfo = consumes<vector<TotemFEDInfo>>(ps.getParameter<edm::InputTag>("tagFEDInfo"));
  tokenTriggerCounters = consumes<TotemTriggerCounters>(ps.getParameter<edm::InputTag>("tagTriggerCounters"));
}

//----------------------------------------------------------------------------------------------------

TotemDAQTriggerDQMSource::~TotemDAQTriggerDQMSource()
{
}

//----------------------------------------------------------------------------------------------------

void TotemDAQTriggerDQMSource::dqmBeginRun(edm::Run const &, edm::EventSetup const &)
{
}

//----------------------------------------------------------------------------------------------------

void TotemDAQTriggerDQMSource::bookHistograms(DQMStore::IBooker &ibooker, edm::Run const &, edm::EventSetup const &)
{
  ibooker.cd();
  
  ibooker.setCurrentFolder("CTPPS/DAQ/");

  daq_bx_diff = ibooker.book1D("bx_diff", ";OptoRx_{i}.BX - OptoRx_{j}.BX", 100, 0., 0.);
  daq_event_bx_diff = ibooker.book1D("daq_event_bx_diff", ";OptoRx_{i}.BX - Event.BX", 100, 0., 0.);
  daq_event_bx_diff_vs_fed = ibooker.book2D("daq_event_bx_diff_vs_fed", ";OptoRx.ID;OptoRx.BX - Event.BX", 10, 0., 0., 10., 0., 0.);

  daq_trigger_bx_diff = ibooker.book1D("trigger_bx_diff", ";OptoRx_{i}.BX - LoneG.BX", 100, 0., 0.);

  ibooker.setCurrentFolder("CTPPS/Trigger/");

  trigger_type = ibooker.book1D("type", ";type", 100, 0., 0.);
  trigger_event_num = ibooker.book1D("event_num", ";event_num", 100, 0., 0.);
  trigger_bunch_num = ibooker.book1D("bunch_num", ";bunch_num", 100, 0., 0.);
  trigger_src_id = ibooker.book1D("src_id", ";src_id", 100, 0., 0.);
  trigger_orbit_num = ibooker.book1D("orbit_num", ";orbit_num", 100, 0., 0.);
  trigger_revision_num = ibooker.book1D("revision_num", ";revision_num", 100, 0., 0.);
  trigger_run_num = ibooker.book1D("run_num", ";run_num", 100, 0., 0.);
  trigger_trigger_num = ibooker.book1D("trigger_num", ";trigger_num", 100, 0., 0.);
  trigger_inhibited_triggers_num = ibooker.book1D("inhibited_triggers_num", ";inhibited_triggers_num", 100, 0., 0.);
  trigger_input_status_bits = ibooker.book1D("input_status_bits", ";input_status_bits", 100, 0., 0.);
}

//----------------------------------------------------------------------------------------------------

void TotemDAQTriggerDQMSource::beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& context) 
{
}

//----------------------------------------------------------------------------------------------------

void TotemDAQTriggerDQMSource::analyze(edm::Event const& event, edm::EventSetup const& eventSetup)
{
  // get input
  Handle<vector<TotemFEDInfo>> fedInfo;
  event.getByToken(tokenFEDInfo, fedInfo);

  Handle<TotemTriggerCounters> triggerCounters;
  event.getByToken(tokenTriggerCounters, triggerCounters);

  // check validity
  bool daqValid = fedInfo.isValid();
  bool triggerValid = triggerCounters.isValid();

  if (!daqValid || !triggerValid)
  {
    if (verbosity)
    {
      LogPrint("TotemDAQTriggerDQMSource") <<
        "WARNING in TotemDAQTriggerDQMSource::analyze > some of the inputs are not valid.\n"
        << "    fedInfo.isValid = " << fedInfo.isValid() << "\n"
        << "    triggerCounters.isValid = " << triggerCounters.isValid();
    }
  }

  // DAQ plots
  if (daqValid)
  {
    for (auto &it1 : *fedInfo)
    {
      daq_event_bx_diff->Fill(it1.getBX() - event.bunchCrossing());
      daq_event_bx_diff_vs_fed->Fill(it1.getFEDId(), it1.getBX() - event.bunchCrossing());
  
      for (auto &it2 : *fedInfo)
      {
        if (it2.getFEDId() <= it1.getFEDId())
          continue;
  
        daq_bx_diff->Fill(it2.getBX() - it1.getBX());
      }
    }
  }

  // trigger plots
  if (triggerValid)
  {
    trigger_type->Fill(triggerCounters->type);
    trigger_event_num->Fill(triggerCounters->event_num);
    trigger_bunch_num->Fill(triggerCounters->bunch_num);
    trigger_src_id->Fill(triggerCounters->src_id);
    trigger_orbit_num->Fill(triggerCounters->orbit_num);
    trigger_revision_num->Fill(triggerCounters->revision_num);
    trigger_run_num->Fill(triggerCounters->run_num);
    trigger_trigger_num->Fill(triggerCounters->trigger_num);
    trigger_inhibited_triggers_num->Fill(triggerCounters->inhibited_triggers_num);
    trigger_input_status_bits->Fill(triggerCounters->input_status_bits);
  }

  // combined DAQ + trigger plots
  if (daqValid && triggerValid)
  {
    for (auto &it : *fedInfo)
      daq_trigger_bx_diff->Fill(it.getBX() - triggerCounters->orbit_num);
  }
}

//----------------------------------------------------------------------------------------------------

void TotemDAQTriggerDQMSource::endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& eSetup) 
{
}

//----------------------------------------------------------------------------------------------------

void TotemDAQTriggerDQMSource::endRun(edm::Run const& run, edm::EventSetup const& eSetup)
{
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(TotemDAQTriggerDQMSource);
