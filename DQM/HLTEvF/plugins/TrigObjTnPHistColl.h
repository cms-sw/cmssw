#ifndef DQM_HLTEvF_TrigObjTnPHistColl_h
#define DQM_HLTEvF_TrigObjTnPHistColl_h

//********************************************************************************
//
// Description:
//   Manages a set of histograms intended for tag and probe efficiency measurements
//   using TriggerObjects stored in TriggerEvent as the input
//   selection is limited to basic et/eta/phi cuts and trigger filter requirements
//   The idea that this can run on any of the following data formats RAW,RECO,AOD
//   or even as part of the HLT job
//
//   All histograms in a TrigObjTnPHistColl share the same tag defination and
//   currently the same probe et/eta/phi cuts. The tag trigger requirements may be
//   to pass multiple triggers anded or ored together
//
//   The TrigObjTnPHistColl then has a series of histograms which are filled for
//   probes which pass a specified filter. For each specified filter, a set of
//   2D histograms are produced, <var> vs mass where var is configuable via python
//   These histograms may have additional cuts, eg eta cuts which limit them to barrel
//   or endcap

//   This allows us to get the mass spectrum in each bin to allow signal & bkg fits
//
// Class Structure
//   TrigObjTnPHistColl : master object which manages a series of histograms which
//                        share a common tag defination. It selects tag and probe pairs
//                        and then sends selected probes to fill the relavent histograms
//
//   FilterSelector : specifies and cuts on the trigger filters an object has to pass.
//                    It allows ANDed and ORing of trigger filter requirements.
//                    It acheives this by grouping the filters in sets of filters (FilterSet)
//                    and an object either has to pass all of those filters in the sets or
//                    any of those filters in the set.
//                    An object can then be required to pass all defined FilterSets or any of them
//
//   PathSelector : checks that a given path has fired. Was originally supposed to use instead
//                  GenericTriggerEventFlag but that class was awkward to use in the
//                  concurrentME workflow so PathSelector was created to mimic the required
//                  functionality. It was a quick port of GenericTriggerEventFlag adapted to
//                  to work in our desired workflow and may be replaced/reworked in the future
//
//   TrigObjVarF : allows arbitary access to a given float variable of trigger::TriggerObject
//                 it can also return the abs value of that variable if so requested
//
//   HistFiller : stores the variable a histogram is to be filled with and any cuts the object
//                must additional pass. It then can fill/not fill a histogram using this information
//
//   HistDefs : has all the information necesary to define a histograms to be produced.
//              The Data sub struct containts the HistFiller object, the binning of the
//              histogram and name /title suffexs. There is one set of histogram definations
//              for a TrigObjTnPHistColl so each probe filter has identical binning
//              Each booked histogram contains a local copy of the approprate HistFiller
//
//   HistColl : a collection of histograms to be filled by a probe passing a particular trigger
//              and kinematic selection. Histograms may have additional further selection on the
//              probe (eg limiting to the barrel etc). Each histogram is booked using the central
//              histogram definations and contains a copy of the approprate HistFiller
//
//   ProbeData : a specific filter for a probe object to pass with a collection of histograms to
//               fill managed by HistColl. The filter is not measured by FilterSelector as it is
//               intentionally limited to only a single filter
//
//   TrigObjTnPHistColl : single tag selection and generic requirements for a probe
//      |
//      |--> collection of ProbeData : a set of histos to fill for probes passing a given filter
//              |
//              |--> ProbeData : filter to pass to fill histograms + histograms
//                    |
//                    |--> HistColl : hists for it to be filled
//                           |
//                           |--> collection of HistFillters+their hists, histfillter fills the hist
//
// Author : Sam Harper , RAL, Aug 2018
//
//***********************************************************************************

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"
#include "DQMOffline/Trigger/interface/VarRangeCutColl.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"

class TrigObjTnPHistColl {
public:
  typedef dqm::legacy::DQMStore DQMStore;
  typedef dqm::legacy::MonitorElement MonitorElement;

  class FilterSelector {
  public:
    class FilterSet {
    public:
      explicit FilterSet(const edm::ParameterSet& config);
      static edm::ParameterSetDescription makePSetDescription();
      const trigger::Keys getPassingKeys(const trigger::TriggerEvent& trigEvt) const;

    private:
      std::vector<std::string> filters_;
      bool isAND_;
    };

  public:
    explicit FilterSelector(const edm::ParameterSet& config);
    static edm::ParameterSetDescription makePSetDescription();
    const trigger::Keys getPassingKeys(const trigger::TriggerEvent& trigEvt) const;

  private:
    //helper functions
    static void mergeTrigKeys(trigger::Keys& keys, const trigger::Keys& keysToMerge, bool isAND);
    static void cleanTrigKeys(trigger::Keys& keys);

    std::vector<FilterSet> filterSets_;
    bool isAND_;
  };

  //A rather late addition to replace GenericTriggerEventFlag as it was incompatible with the
  //move to concurrentMEs as GenericTriggerEventFlag owns tokens
  //its more or less a direct port of that class, with the functions inspired by it
  //obviously much less feature rich, only handles HLT
  class PathSelector {
  public:
    PathSelector(const edm::ParameterSet& config);
    static edm::ParameterSetDescription makePSetDescription();
    void init(const HLTConfigProvider& hltConfig);
    bool operator()(const edm::TriggerResults& trigResults, const edm::TriggerNames& trigNames) const;

  private:
    static std::string expandSelectionStr(const std::string& selStr,
                                          const HLTConfigProvider& hltConfig,
                                          bool isAND,
                                          int verbose);
    static std::string expandPath(const std::string& pathPattern,
                                  const HLTConfigProvider& hltConfig,
                                  bool isAND,
                                  int verbose);

    std::string selectionStr_;    //with wildcard etc
    std::string expandedSelStr_;  //with wildcards expanded, set by init
    bool isANDForExpandedPaths_;
    int verbose_;
    bool isInited_;
  };

  class TrigObjVarF {
  public:
    explicit TrigObjVarF(std::string varName);
    float operator()(const trigger::TriggerObject& obj) const {
      return isAbs_ ? std::abs((obj.*varFunc_)()) : (obj.*varFunc_)();
    }

  private:
    float (trigger::TriggerObject::*varFunc_)() const;
    bool isAbs_;
  };

  class HistFiller {
  public:
    explicit HistFiller(const edm::ParameterSet& config);
    static edm::ParameterSetDescription makePSetDescription();
    void operator()(const trigger::TriggerObject& probe, float mass, dqm::reco::MonitorElement* hist) const;

  private:
    VarRangeCutColl<trigger::TriggerObject> localCuts_;
    TrigObjVarF var_;
  };

  //Histogram Defination, defines the histogram (name,title,bins,how its filled)
  class HistDefs {
  private:
    class Data {
    public:
      explicit Data(const edm::ParameterSet& config);
      static edm::ParameterSetDescription makePSetDescription();
      dqm::reco::MonitorElement* book(DQMStore::IBooker& iBooker,
                                      const std::string& name,
                                      const std::string& title,
                                      const std::vector<float>& massBins) const;
      const HistFiller& filler() const { return histFiller_; }

    private:
      HistFiller histFiller_;
      std::vector<float> bins_;
      std::string nameSuffex_;
      std::string titleSuffex_;
    };

  public:
    explicit HistDefs(const edm::ParameterSet& config);
    static edm::ParameterSetDescription makePSetDescription();
    std::vector<std::pair<HistFiller, dqm::reco::MonitorElement*> > bookHists(DQMStore::IBooker& iBooker,
                                                                              const std::string& name,
                                                                              const std::string& title) const;

  private:
    std::vector<Data> histData_;
    std::vector<float> massBins_;
  };

  class HistColl {
  public:
    HistColl() {}
    void bookHists(DQMStore::IBooker& iBooker,
                   const std::string& name,
                   const std::string& title,
                   const HistDefs& histDefs);
    void fill(const trigger::TriggerObject& probe, float mass) const;

  private:
    std::vector<std::pair<HistFiller, dqm::reco::MonitorElement*> > hists_;
  };

  class ProbeData {
  public:
    explicit ProbeData(std::string probeFilter) : probeFilter_(std::move(probeFilter)) {}
    void bookHists(const std::string& tagName, DQMStore::IBooker& iBooker, const HistDefs& histDefs);
    void fill(const trigger::size_type tagKey,
              const trigger::TriggerEvent& trigEvt,
              const VarRangeCutColl<trigger::TriggerObject>& probeCuts) const;

  private:
    std::string probeFilter_;
    HistColl hists_;
  };

public:
  TrigObjTnPHistColl(const edm::ParameterSet& config);
  static edm::ParameterSetDescription makePSetDescription();
  void init(const HLTConfigProvider& hltConfig) { evtTrigSel_.init(hltConfig); }
  void bookHists(DQMStore::IBooker& iBooker);
  void fill(const trigger::TriggerEvent& trigEvt,
            const edm::TriggerResults& trigResults,
            const edm::TriggerNames& trigNames) const;

private:
  //helper function, probably should go in a utilty class
  static const trigger::Keys getKeys(const trigger::TriggerEvent& trigEvt, const std::string& filterName);

  VarRangeCutColl<trigger::TriggerObject> tagCuts_, probeCuts_;
  FilterSelector tagFilters_;
  std::string collName_;
  std::string folderName_;
  HistDefs histDefs_;
  std::vector<ProbeData> probeHists_;
  PathSelector evtTrigSel_;
};

#endif
