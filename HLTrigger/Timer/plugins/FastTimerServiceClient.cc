// C++ headers
#include <string>
#include <cstring>

// boost headers
#include <boost/regex.hpp>

// Root headers
#include <TH1D.h>
#include <TH1F.h>

// CMSSW headers
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/Provenance/interface/ProcessHistory.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"

struct MEPSet {
  std::string folder;
  std::string name;
  int nbins;
  double xmin;
  double xmax;
};

class FastTimerServiceClient : public DQMEDHarvester {
public:
  explicit FastTimerServiceClient(edm::ParameterSet const&);
  ~FastTimerServiceClient() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  static void fillLumiMePSetDescription(edm::ParameterSetDescription& pset);
  static void fillPUMePSetDescription(edm::ParameterSetDescription& pset);

private:
  void dqmEndLuminosityBlock(DQMStore::IBooker& booker,
                             DQMStore::IGetter& getter,
                             edm::LuminosityBlock const&,
                             edm::EventSetup const&) override;
  void dqmEndJob(DQMStore::IBooker& booker, DQMStore::IGetter& getter) override;

  void fillSummaryPlots(DQMStore::IBooker& booker, DQMStore::IGetter& getter);
  void fillProcessSummaryPlots(DQMStore::IBooker& booker, DQMStore::IGetter& getter, std::string const& path);
  void fillPathSummaryPlots(DQMStore::IBooker& booker,
                            DQMStore::IGetter& getter,
                            double events,
                            std::string const& path);
  void fillPlotsVsLumi(DQMStore::IBooker& booker,
                       DQMStore::IGetter& getter,
                       std::string const& current_path,
                       std::string const& suffix,
                       MEPSet const& pset);

  static MEPSet getHistoPSet(const edm::ParameterSet& pset);

  std::string const dqm_path_;

  bool const doPlotsVsOnlineLumi_;
  bool const doPlotsVsPixelLumi_;
  bool const doPlotsVsPU_;

  MEPSet const onlineLumiMEPSet_;
  MEPSet const pixelLumiMEPSet_;
  MEPSet const puMEPSet_;

  bool const fillEveryLumiSection_;
};

FastTimerServiceClient::FastTimerServiceClient(edm::ParameterSet const& config)
    : dqm_path_(config.getUntrackedParameter<std::string>("dqmPath")),
      doPlotsVsOnlineLumi_(config.getParameter<bool>("doPlotsVsOnlineLumi")),
      doPlotsVsPixelLumi_(config.getParameter<bool>("doPlotsVsPixelLumi")),
      doPlotsVsPU_(config.getParameter<bool>("doPlotsVsPU")),
      onlineLumiMEPSet_(doPlotsVsOnlineLumi_ ? getHistoPSet(config.getParameter<edm::ParameterSet>("onlineLumiME"))
                                             : MEPSet{}),
      pixelLumiMEPSet_(doPlotsVsPixelLumi_ ? getHistoPSet(config.getParameter<edm::ParameterSet>("pixelLumiME"))
                                           : MEPSet{}),
      puMEPSet_(doPlotsVsPU_ ? getHistoPSet(config.getParameter<edm::ParameterSet>("puME")) : MEPSet{}),
      fillEveryLumiSection_(config.getParameter<bool>("fillEveryLumiSection")) {}

void FastTimerServiceClient::dqmEndJob(DQMStore::IBooker& booker, DQMStore::IGetter& getter) {
  fillSummaryPlots(booker, getter);
}

void FastTimerServiceClient::dqmEndLuminosityBlock(DQMStore::IBooker& booker,
                                                   DQMStore::IGetter& getter,
                                                   edm::LuminosityBlock const& lumi,
                                                   edm::EventSetup const& setup) {
  if (fillEveryLumiSection_)
    fillSummaryPlots(booker, getter);
}

void FastTimerServiceClient::fillSummaryPlots(DQMStore::IBooker& booker, DQMStore::IGetter& getter) {
  if (getter.get(dqm_path_ + "/event time_real")) {
    // the plots are directly in the configured folder
    fillProcessSummaryPlots(booker, getter, dqm_path_);
  } else {
    static const boost::regex running_n_processes(".*/Running .*");

    booker.setCurrentFolder(dqm_path_);
    std::vector<std::string> subdirs = getter.getSubdirs();
    for (auto const& subdir : subdirs) {
      // the plots are in a per-number-of-processes folder
      if (boost::regex_match(subdir, running_n_processes)) {
        booker.setCurrentFolder(subdir);
        if (getter.get(subdir + "/event time_real"))
          fillProcessSummaryPlots(booker, getter, subdir);

        std::vector<std::string> subsubdirs = getter.getSubdirs();
        for (auto const& subsubdir : subsubdirs) {
          if (getter.get(subsubdir + "/event time_real"))
            fillProcessSummaryPlots(booker, getter, subsubdir);
        }
      }
    }  // loop on subdirs
  }
}

void FastTimerServiceClient::fillProcessSummaryPlots(DQMStore::IBooker& booker,
                                                     DQMStore::IGetter& getter,
                                                     std::string const& current_path) {
  MonitorElement* me = getter.get(current_path + "/event time_real");
  if (me == nullptr)
    // no FastTimerService DQM information
    return;

  if (doPlotsVsOnlineLumi_)
    fillPlotsVsLumi(booker, getter, current_path, "vs_lumi", onlineLumiMEPSet_);
  if (doPlotsVsPixelLumi_)
    fillPlotsVsLumi(booker, getter, current_path, "vs_pixelLumi", pixelLumiMEPSet_);
  if (doPlotsVsPU_)
    fillPlotsVsLumi(booker, getter, current_path, "vs_pileup", puMEPSet_);

  //  getter.setCurrentFolder(current_path);

  double events = me->getTH1F()->GetEntries();

  // look for per-process directories
  static const boost::regex process_name(".*/process .*");

  booker.setCurrentFolder(current_path);  // ?!?!?
  std::vector<std::string> subdirs = getter.getSubdirs();
  for (auto const& subdir : subdirs) {
    if (boost::regex_match(subdir, process_name)) {
      getter.setCurrentFolder(subdir);
      // look for per-path plots inside each per-process directory
      std::vector<std::string> subsubdirs = getter.getSubdirs();
      for (auto const& subsubdir : subsubdirs) {
        if (getter.get(subsubdir + "/path time_real")) {
          fillPathSummaryPlots(booker, getter, events, subdir);
          break;
        }
      }
    }
  }  // loop on subdir
}

void FastTimerServiceClient::fillPathSummaryPlots(DQMStore::IBooker& booker,
                                                  DQMStore::IGetter& getter,
                                                  double events,
                                                  std::string const& current_path) {
  // note: the following checks need to be kept separate, as any of these histograms might be missing

  booker.setCurrentFolder(current_path);
  std::vector<std::string> subsubdirs = getter.getSubdirs();
  size_t npaths = subsubdirs.size();

  MonitorElement* paths_time =
      booker.book1D("paths_time_real", "Total (real) time spent in each path", npaths, -0.5, double(npaths) - 0.5);
  MonitorElement* paths_thread =
      booker.book1D("paths_time_thread", "Total (thread) time spent in each path", npaths, -0.5, double(npaths) - 0.5);
  MonitorElement* paths_allocated =
      booker.book1D("paths_allocated", "Total allocated memory in each path", npaths, -0.5, double(npaths) - 0.5);
  MonitorElement* paths_deallocated =
      booker.book1D("paths_deallocated", "Total deallocated in each path", npaths, -0.5, double(npaths) - 0.5);

  MonitorElement* me;
  double mean = -1.;

  // extract the list of Paths and EndPaths from the summary plots
  int ibin = 1;
  for (auto const& subsubdir : subsubdirs) {
    std::string test = "/path ";
    if (subsubdir.find(test) == std::string::npos)
      continue;

    static const boost::regex prefix(current_path + "/path ");
    std::string path = boost::regex_replace(subsubdir, prefix, "");

    paths_time->setBinLabel(ibin, path);
    paths_thread->setBinLabel(ibin, path);
    paths_allocated->setBinLabel(ibin, path);
    paths_deallocated->setBinLabel(ibin, path);

    if ((me = getter.get(subsubdir + "/path time_real"))) {
      mean = me->getMean();
      paths_time->setBinContent(ibin, mean);
    }
    if ((me = getter.get(subsubdir + "/path time_thread"))) {
      mean = me->getMean();
      paths_thread->setBinContent(ibin, mean);
    }
    if ((me = getter.get(subsubdir + "/path allocated"))) {
      mean = me->getMean();
      paths_allocated->setBinContent(ibin, mean);
    }

    if ((me = getter.get(subsubdir + "/path deallocated"))) {
      mean = me->getMean();
      paths_deallocated->setBinContent(ibin, mean);
    }

    ibin++;
  }

  for (auto const& subsubdir : subsubdirs) {
    // for each path, fill histograms with
    //  - the average time spent in each module (total time spent in that module, averaged over all events)
    //  - the running time spent in each module (total time spent in that module, averaged over the events where that module actually ran)
    //  - the "efficiency" of each module (number of time a module succeded divided by the number of times the has run)

    getter.setCurrentFolder(subsubdir);
    std::vector<std::string> allmenames = getter.getMEs();
    if (allmenames.empty())
      continue;

    MonitorElement* me_counter = getter.get(subsubdir + "/module_counter");
    MonitorElement* me_real_total = getter.get(subsubdir + "/module_time_real_total");
    MonitorElement* me_thread_total = getter.get(subsubdir + "/module_time_thread_total");

    if (me_counter == nullptr or me_real_total == nullptr)
      continue;

    TH1D* counter = me_counter->getTH1D();
    TH1D* real_total = me_real_total->getTH1D();
    TH1D* thread_total = me_thread_total->getTH1D();
    uint32_t bins = counter->GetXaxis()->GetNbins() - 1;
    double min = counter->GetXaxis()->GetXmin();
    double max = counter->GetXaxis()->GetXmax() - 1;

    TH1F* real_average;
    TH1F* real_running;
    TH1F* thread_average;
    TH1F* thread_running;
    TH1F* efficiency;
    MonitorElement* me;

    booker.setCurrentFolder(subsubdir);
    me = getter.get(subsubdir + "/module_time_real_average");
    if (me) {
      real_average = me->getTH1F();
      assert(me->getTH1F()->GetXaxis()->GetXmin() == min);
      assert(me->getTH1F()->GetXaxis()->GetXmax() == max);
      real_average->Reset();
    } else {
      real_average = booker.book1D("module_time_real_average", "module real average timing", bins, min, max)->getTH1F();
      real_average->SetYTitle("average processing (real) time [ms]");
      for (uint32_t i = 1; i <= bins; ++i) {
        const char* module = counter->GetXaxis()->GetBinLabel(i);
        real_average->GetXaxis()->SetBinLabel(i, module);
      }
    }

    me = getter.get(subsubdir + "/module_time_thread_average");
    if (me) {
      thread_average = me->getTH1F();
      assert(me->getTH1F()->GetXaxis()->GetXmin() == min);
      assert(me->getTH1F()->GetXaxis()->GetXmax() == max);
      thread_average->Reset();
    } else {
      thread_average =
          booker.book1D("module_time_thread_average", "module thread average timing", bins, min, max)->getTH1F();
      thread_average->SetYTitle("average processing (thread) time [ms]");
      for (uint32_t i = 1; i <= bins; ++i) {
        const char* module = counter->GetXaxis()->GetBinLabel(i);
        thread_average->GetXaxis()->SetBinLabel(i, module);
      }
    }

    me = getter.get(subsubdir + "/module_time_real_running");
    if (me) {
      real_running = me->getTH1F();
      assert(me->getTH1F()->GetXaxis()->GetXmin() == min);
      assert(me->getTH1F()->GetXaxis()->GetXmax() == max);
      real_running->Reset();
    } else {
      real_running = booker.book1D("module_time_real_running", "module real running timing", bins, min, max)->getTH1F();
      real_running->SetYTitle("running processing (real) time [ms]");
      for (uint32_t i = 1; i <= bins; ++i) {
        const char* module = counter->GetXaxis()->GetBinLabel(i);
        real_running->GetXaxis()->SetBinLabel(i, module);
      }
    }

    me = getter.get(subsubdir + "/module_time_thread_running");
    if (me) {
      thread_running = me->getTH1F();
      assert(me->getTH1F()->GetXaxis()->GetXmin() == min);
      assert(me->getTH1F()->GetXaxis()->GetXmax() == max);
      thread_running->Reset();
    } else {
      thread_running =
          booker.book1D("module_time_thread_running", "module thread running timing", bins, min, max)->getTH1F();
      thread_running->SetYTitle("running processing (thread) time [ms]");
      for (uint32_t i = 1; i <= bins; ++i) {
        const char* module = counter->GetXaxis()->GetBinLabel(i);
        thread_running->GetXaxis()->SetBinLabel(i, module);
      }
    }

    me = getter.get(subsubdir + "/module_efficiency");
    if (me) {
      efficiency = me->getTH1F();
      assert(me->getTH1F()->GetXaxis()->GetXmin() == min);
      assert(me->getTH1F()->GetXaxis()->GetXmax() == max);
      efficiency->Reset();
    } else {
      efficiency = booker.book1D("module_efficiency", "module efficiency", bins, min, max)->getTH1F();
      efficiency->SetYTitle("filter efficiency");
      efficiency->SetMaximum(1.05);
      for (uint32_t i = 1; i <= bins; ++i) {
        const char* module = counter->GetXaxis()->GetBinLabel(i);
        efficiency->GetXaxis()->SetBinLabel(i, module);
      }
    }

    for (uint32_t i = 1; i <= bins; ++i) {
      double n = counter->GetBinContent(i);
      double p = counter->GetBinContent(i + 1);
      if (n)
        efficiency->SetBinContent(i, p / n);

      // real timing
      double t = real_total->GetBinContent(i);
      real_average->SetBinContent(i, t / events);
      if (n)
        real_running->SetBinContent(i, t / n);

      // thread timing
      t = thread_total->GetBinContent(i);
      thread_average->SetBinContent(i, t / events);
      if (n)
        thread_running->SetBinContent(i, t / n);
    }

    // vs lumi
    if (doPlotsVsOnlineLumi_)
      fillPlotsVsLumi(booker, getter, subsubdir, "vs_lumi", onlineLumiMEPSet_);
    if (doPlotsVsPixelLumi_)
      fillPlotsVsLumi(booker, getter, subsubdir, "vs_pixelLumi", pixelLumiMEPSet_);
    if (doPlotsVsPU_)
      fillPlotsVsLumi(booker, getter, subsubdir, "vs_pileup", puMEPSet_);
  }
}

#include "FWCore/MessageLogger/interface/MessageLogger.h"
void FastTimerServiceClient::fillPlotsVsLumi(DQMStore::IBooker& booker,
                                             DQMStore::IGetter& getter,
                                             std::string const& current_path,
                                             std::string const& suffix,
                                             MEPSet const& pset) {
  std::vector<std::string> menames;

  static const boost::regex byls(".*byls");
  // get all MEs in the current_path
  getter.setCurrentFolder(current_path);
  std::vector<std::string> allmenames = getter.getMEs();
  for (auto const& m : allmenames) {
    // get only MEs vs LS
    if (boost::regex_match(m, byls))
      menames.push_back(m);
  }
  // if no MEs available, return
  if (menames.empty())
    return;

  // get info for getting the lumi VS LS histogram
  std::string folder = pset.folder;
  std::string name = pset.name;
  int nbins = pset.nbins;
  double xmin = pset.xmin;
  double xmax = pset.xmax;

  // get lumi/PU VS LS ME
  getter.setCurrentFolder(folder);
  MonitorElement* lumiVsLS = getter.get(folder + "/" + name);
  // if no ME available, return
  if (!lumiVsLS) {
    edm::LogWarning("FastTimerServiceClient") << "no " << name << " ME is available in " << folder << std::endl;
    return;
  }

  // get range and binning for new MEs x-axis
  size_t size = lumiVsLS->getTProfile()->GetXaxis()->GetNbins();
  std::string xtitle = lumiVsLS->getTProfile()->GetYaxis()->GetTitle();

  std::vector<double> lumi;
  std::vector<int> LS;
  for (size_t ibin = 1; ibin <= size; ++ibin) {
    // avoid to store points w/ no info
    if (lumiVsLS->getTProfile()->GetBinContent(ibin) == 0.)
      continue;

    lumi.push_back(lumiVsLS->getTProfile()->GetBinContent(ibin));
    LS.push_back(lumiVsLS->getTProfile()->GetXaxis()->GetBinCenter(ibin));
  }

  booker.setCurrentFolder(current_path);
  getter.setCurrentFolder(current_path);
  for (auto const& m : menames) {
    std::string label = m;
    label.erase(label.find("_byls"));

    MonitorElement* me = getter.get(current_path + "/" + m);
    float ymin = 0.;
    float ymax = std::numeric_limits<float>::max();
    std::string ytitle = me->getTProfile()->GetYaxis()->GetTitle();

    MonitorElement* meVsLumi = getter.get(current_path + "/" + label + "_" + suffix);
    if (meVsLumi) {
      assert(meVsLumi->getTProfile()->GetXaxis()->GetXmin() == xmin);
      assert(meVsLumi->getTProfile()->GetXaxis()->GetXmax() == xmax);
      meVsLumi->Reset();  // do I have to do it ?!?!?
    } else {
      meVsLumi = booker.bookProfile(label + "_" + suffix, label + "_" + suffix, nbins, xmin, xmax, ymin, ymax);
      //    TProfile* meVsLumi_p = meVsLumi->getTProfile();
      meVsLumi->getTProfile()->GetXaxis()->SetTitle(xtitle.c_str());
      meVsLumi->getTProfile()->GetYaxis()->SetTitle(ytitle.c_str());
    }
    for (size_t ils = 0; ils < LS.size(); ++ils) {
      int ibin = me->getTProfile()->GetXaxis()->FindBin(LS[ils]);
      double y = me->getTProfile()->GetBinContent(ibin);

      meVsLumi->Fill(lumi[ils], y);
    }
  }
}

void FastTimerServiceClient::fillLumiMePSetDescription(edm::ParameterSetDescription& pset) {
  pset.add<std::string>("folder", "HLT/LumiMonitoring");
  pset.add<std::string>("name", "lumiVsLS");
  pset.add<int>("nbins", 440);
  pset.add<double>("xmin", 0.);
  pset.add<double>("xmax", 22000.);
}

void FastTimerServiceClient::fillPUMePSetDescription(edm::ParameterSetDescription& pset) {
  pset.add<std::string>("folder", "HLT/LumiMonitoring");
  pset.add<std::string>("name", "puVsLS");
  pset.add<int>("nbins", 260);
  pset.add<double>("xmin", 0.);
  pset.add<double>("xmax", 130.);
}

MEPSet FastTimerServiceClient::getHistoPSet(const edm::ParameterSet& pset) {
  return MEPSet{
      pset.getParameter<std::string>("folder"),
      pset.getParameter<std::string>("name"),
      pset.getParameter<int>("nbins"),
      pset.getParameter<double>("xmin"),
      pset.getParameter<double>("xmax"),
  };
}

void FastTimerServiceClient::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.addUntracked<std::string>("dqmPath", "HLT/TimerService");
  desc.add<bool>("doPlotsVsOnlineLumi", true);
  desc.add<bool>("doPlotsVsPixelLumi", false);
  desc.add<bool>("doPlotsVsPU", true);

  edm::ParameterSetDescription onlineLumiMEPSet;
  fillLumiMePSetDescription(onlineLumiMEPSet);
  desc.add<edm::ParameterSetDescription>("onlineLumiME", onlineLumiMEPSet);

  edm::ParameterSetDescription pixelLumiMEPSet;
  fillLumiMePSetDescription(pixelLumiMEPSet);
  desc.add<edm::ParameterSetDescription>("pixelLumiME", pixelLumiMEPSet);

  edm::ParameterSetDescription puMEPSet;
  fillPUMePSetDescription(puMEPSet);
  desc.add<edm::ParameterSetDescription>("puME", puMEPSet);
  desc.add<bool>("fillEveryLumiSection", true);
  descriptions.add("fastTimerServiceClient", desc);
}

// declare this class as a framework plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(FastTimerServiceClient);
