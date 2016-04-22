/****************************************************************************
*
* This is a part of TotemDQM and TOTEM offline software.
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
    edm::EDGetTokenT<std::vector<TotemFEDInfo>> tokenFEDInfo;
    edm::EDGetTokenT<TotemTriggerCounters> tokenTriggerCounters;
    
    MonitorElement *daq_bx_diff;
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

TotemDAQTriggerDQMSource::TotemDAQTriggerDQMSource(const edm::ParameterSet& ps)
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
  
  ibooker.setCurrentFolder("Totem/DAQ/");

  daq_bx_diff = ibooker.book1D("bx_diff", "bx_diff", 100, 0., 0.);

  daq_trigger_bx_diff = ibooker.book1D("trigger_bx_diff", "trigger_bx_diff", 100, 0., 0.);

  ibooker.setCurrentFolder("Totem/Trigger/");

  trigger_type = ibooker.book1D("type", "type", 100, 0., 0.);
  trigger_event_num = ibooker.book1D("event_num", "event_num", 100, 0., 0.);
  trigger_bunch_num = ibooker.book1D("bunch_num", "bunch_num", 100, 0., 0.);
  trigger_src_id = ibooker.book1D("src_id", "src_id", 100, 0., 0.);
  trigger_orbit_num = ibooker.book1D("orbit_num", "orbit_num", 100, 0., 0.);
  trigger_revision_num = ibooker.book1D("revision_num", "revision_num", 100, 0., 0.);
  trigger_run_num = ibooker.book1D("run_num", "run_num", 100, 0., 0.);
  trigger_trigger_num = ibooker.book1D("trigger_num", "trigger_num", 100, 0., 0.);
  trigger_inhibited_triggers_num = ibooker.book1D("inhibited_triggers_num", "inhibited_triggers_num", 100, 0., 0.);
  trigger_input_status_bits = ibooker.book1D("input_status_bits", "input_status_bits", 100, 0., 0.);
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
  bool valid = true;
  valid &= fedInfo.isValid();
  valid &= triggerCounters.isValid();

  if (!valid)
  {
    printf("ERROR in TotemDAQTriggerDQMSource::analyze > some of the required inputs are not valid. Skipping this event.\n");
    printf("\tfedInfo.isValid = %i\n", fedInfo.isValid());
    printf("\ttriggerCounters.isValid = %i\n", triggerCounters.isValid());

    return;
  }

  // DAQ plots
  for (auto &it1 : *fedInfo)
  {
    for (auto &it2 : *fedInfo)
    {
      if (it2.getFEDId() <= it1.getFEDId())
        continue;

      daq_bx_diff->Fill(it2.getBX() - it1.getBX());
    }
  }

  // trigger plots
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

  // combined DAQ + trigger plots
  for (auto &it : *fedInfo)
    daq_trigger_bx_diff->Fill(it.getBX() - triggerCounters->orbit_num);
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
