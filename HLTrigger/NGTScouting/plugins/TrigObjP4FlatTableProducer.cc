// NanoAOD FlatTable producer for HLT trigger objects.
//
// The set of objects saved as table rows is the deduplicated union of:
//   (a) all objects passing the last save-tags filter of every path in pathNames
//   (b) all objects passing any filter listed in extraFilters
//
// Per-row columns:
//   pt, eta, phi, mass   - four-momentum
//   id                   - trigger object type id (TriggerTypeDefs.h)
//   <pathName>           - bool: object passes the last filter of that path
//
// Path name branch sanitisation: ':' and '/' are replaced with '_'.

// user includes
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

// standard includes
#include <algorithm>
#include <string>
#include <vector>
#include <unordered_set>

class TrigObjP4FlatTableProducer : public edm::global::EDProducer<edm::RunCache<HLTConfigProvider>> {
public:
  explicit TrigObjP4FlatTableProducer(const edm::ParameterSet&);
  ~TrigObjP4FlatTableProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

  // RunCache interface - one HLTConfigProvider initialised per run, thread-safely
  std::shared_ptr<HLTConfigProvider> globalBeginRun(const edm::Run&, const edm::EventSetup&) const override;
  void globalEndRun(const edm::Run&, const edm::EventSetup&) const override {}

  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  // Return the (sorted, unique) set of TriggerObjectCollection indices for
  // objects that pass filterLabel.  Returns empty if the filter is not found.
  static std::vector<trigger::size_type> indicesPassingFilter(const trigger::TriggerEvent& trigEvent,
                                                              const std::string& filterLabel,
                                                              const std::string& processName);

  // Sanitise a path/filter name into a valid ROOT branch name.
  static std::string toBranchName(std::string name);

  // ---------- config ----------
  const edm::InputTag trigEventTag_;
  const edm::EDGetTokenT<trigger::TriggerEvent> trigEventToken_;
  const std::string tableName_;
  const std::vector<std::string> pathNames_;     // HLT paths  -> last-filter seeds + bool columns
  const std::vector<std::string> extraFilters_;  // extra filters -> additional seed objects
};

// ---------------------------------------------------------------------------
// constructor
// ---------------------------------------------------------------------------
TrigObjP4FlatTableProducer::TrigObjP4FlatTableProducer(const edm::ParameterSet& ps)
    : trigEventTag_(ps.getParameter<edm::InputTag>("triggerEvent")),
      trigEventToken_(consumes<trigger::TriggerEvent>(trigEventTag_)),
      tableName_(ps.getParameter<std::string>("tableName")),
      pathNames_(ps.getParameter<std::vector<std::string>>("pathNames")),
      extraFilters_(ps.getParameter<std::vector<std::string>>("extraFilters")) {
  produces<nanoaod::FlatTable>();
}

// ---------------------------------------------------------------------------
// RunCache
// ---------------------------------------------------------------------------
std::shared_ptr<HLTConfigProvider> TrigObjP4FlatTableProducer::globalBeginRun(const edm::Run& run,
                                                                              const edm::EventSetup& setup) const {
  auto hltConfig = std::make_shared<HLTConfigProvider>();
  bool changed = false;
  if (!hltConfig->init(run, setup, trigEventTag_.process(), changed))
    edm::LogWarning("TrigObjP4FlatTableProducer")
        << "HLTConfigProvider::init() failed for process " << trigEventTag_.process();
  return hltConfig;
}

// ---------------------------------------------------------------------------
// helper – indices of objects passing a filter
// ---------------------------------------------------------------------------
std::vector<trigger::size_type> TrigObjP4FlatTableProducer::indicesPassingFilter(const trigger::TriggerEvent& trigEvent,
                                                                                 const std::string& filterLabel,
                                                                                 const std::string& processName) {
  for (size_t ifilt = 0; ifilt < trigEvent.sizeFilters(); ++ifilt) {
    std::string fullname = trigEvent.filterTag(ifilt).label();  // just label part
    //const trigger::size_type filterIdx = trigEvent.filterIndex(edm::InputTag(filterLabel, "", processName).encode());
    if (fullname == filterLabel) {
      if (ifilt >= trigEvent.sizeFilters())
        return {};
      const trigger::Keys& keys = trigEvent.filterKeys(ifilt);
      return {keys.begin(), keys.end()};  // already sorted by TriggerEvent
    }
  }
  return {};
}

// ---------------------------------------------------------------------------
// helper - sanitise name -> ROOT branch name
// ---------------------------------------------------------------------------
std::string TrigObjP4FlatTableProducer::toBranchName(std::string name) {
  for (char& c : name)
    if (c == ':' || c == '/' || c == '-' || c == '.')
      c = '_';
  return name;
}

// ---------------------------------------------------------------------------
// produce
// ---------------------------------------------------------------------------
void TrigObjP4FlatTableProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup&) const {
  const auto& trigEvent = iEvent.get(trigEventToken_);
  const auto& hltConfig = *runCache(iEvent.getRun().index());
  const std::string& proc = trigEventTag_.process();
  const trigger::TriggerObjectCollection& allObjs = trigEvent.getObjects();

  // -----------------------------------------------------------------------
  // Step 1 - resolve the last save-tags filter for every requested path.
  //          We store it so we can reuse it for the boolean flag columns.
  // -----------------------------------------------------------------------
  struct PathInfo {
    std::string name;        // original path name
    std::string lastFilter;  // last save-tags filter (empty if path unknown)
  };
  std::vector<PathInfo> pathInfos;
  pathInfos.reserve(pathNames_.size());

  for (const std::string& path : pathNames_) {
    PathInfo pi;
    pi.name = path;
    const unsigned int pathIdx = hltConfig.triggerIndex(path);
    if (pathIdx < hltConfig.size()) {
      const auto& saveTags = hltConfig.saveTagsModules(pathIdx);
      if (!saveTags.empty()) {
        pi.lastFilter = saveTags.back();
      } else {
        edm::LogInfo("TrigObjP4FlatTableProducer") << "Path " << path << " has no save-tags modules - skipped as seed";
      }
    } else {
      edm::LogInfo("TrigObjP4FlatTableProducer") << "Path " << path << " not found in HLT menu - skipped as seed";
    }
    pathInfos.push_back(std::move(pi));
  }

  // -----------------------------------------------------------------------
  // Step 2 - build the union of seeding object indices:
  //          - last filter of every known path
  //          - every filter in extraFilters_
  // Use an ordered set so the final table is deterministically ordered by
  // object index (same ordering as TriggerObjectCollection).
  // -----------------------------------------------------------------------
  std::unordered_set<trigger::size_type> seedSet;

  for (const PathInfo& pi : pathInfos) {
    LogTrace("TrigObjP4FlatTableProducer") << "path: " << pi.name << " filter: " << pi.lastFilter << std::endl;
    if (pi.lastFilter.empty()) {
      continue;
    }
    for (auto idx : indicesPassingFilter(trigEvent, pi.lastFilter, proc)) {
      seedSet.insert(idx);
    }
  }

  for (const std::string& filter : extraFilters_) {
    for (auto idx : indicesPassingFilter(trigEvent, filter, proc))
      seedSet.insert(idx);
  }

  // Sort to give stable row ordering
  std::vector<trigger::size_type> rowIndices(seedSet.begin(), seedSet.end());
  std::sort(rowIndices.begin(), rowIndices.end());
  const size_t nObj = rowIndices.size();

  LogTrace("TrigObjP4FlatTableProducer") << "nObj=" << nObj << " allObjs.size()=" << allObjs.size() << std::endl;

  // -----------------------------------------------------------------------
  // Step 3 - fill kinematic columns
  // -----------------------------------------------------------------------
  std::vector<float> pt(nObj), eta(nObj), phi(nObj), mass(nObj);
  std::vector<int> id(nObj);

  for (size_t i = 0; i < nObj; ++i) {
    LogTrace("TrigObjP4FlatTableProducer") << "Processing object " << i << " with rowIndices[" << i << "]" << std::endl;

    const trigger::TriggerObject& obj = allObjs[rowIndices[i]];
    pt[i] = obj.pt();
    eta[i] = obj.eta();
    phi[i] = obj.phi();
    mass[i] = obj.mass();
    id[i] = obj.id();

    LogTrace("TrigObjP4FlatTableProducer") << "  -> pt=" << pt[i] << ", eta=" << eta[i] << ", phi=" << phi[i]
                                           << ", mass=" << mass[i] << ", id=" << id[i] << std::endl;
  }

  // -----------------------------------------------------------------------
  // Step 4 - per-path boolean flags:
  //          for each row, did this object pass the last filter of the path?
  // -----------------------------------------------------------------------
  // Pre-index the row positions for O(1) lookup when filling flags
  std::unordered_map<trigger::size_type, size_t> rowPosition;
  rowPosition.reserve(nObj);
  for (size_t i = 0; i < nObj; ++i)
    rowPosition[rowIndices[i]] = i;

  struct PathFlag {
    std::string branchName;
    std::string docString;
    std::vector<bool> fired;
  };
  std::vector<PathFlag> pathFlags;
  pathFlags.reserve(pathInfos.size());

  for (const PathInfo& pi : pathInfos) {
    PathFlag pf;
    pf.branchName = toBranchName(pi.name);
    pf.docString = "object passes last filter of path " + pi.name;
    pf.fired.assign(nObj, false);

    if (!pi.lastFilter.empty()) {
      for (auto objIdx : indicesPassingFilter(trigEvent, pi.lastFilter, proc)) {
        auto it = rowPosition.find(objIdx);
        if (it != rowPosition.end())
          pf.fired[it->second] = true;
      }
    }
    pathFlags.push_back(std::move(pf));
  }

  // -----------------------------------------------------------------------
  // Step 5 - assemble FlatTable
  // -----------------------------------------------------------------------
  auto table = std::make_unique<nanoaod::FlatTable>(nObj, tableName_, false, false);

  table->addColumn<float>("pt", pt, "trigger object pT [GeV]");
  table->addColumn<float>("eta", eta, "trigger object eta");
  table->addColumn<float>("phi", phi, "trigger object phi [rad]");
  table->addColumn<float>("mass", mass, "trigger object mass [GeV]");
  table->addColumn<int>("id",
                        id,
                        "trigger object type id – see DataFormats/HLTReco/interface/TriggerTypeDefs.h "
                        "(e.g. 81=photon, 82=electron, 83=muon, 84=tau, 85=jet, 86=bjet, 87=MET, 0=other)");

  for (const PathFlag& pf : pathFlags)
    table->addColumn<bool>(pf.branchName, pf.fired, pf.docString);

  iEvent.put(std::move(table));
}

// ---------------------------------------------------------------------------
// fillDescriptions
// ---------------------------------------------------------------------------
void TrigObjP4FlatTableProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment(
      "Produces a nanoAOD FlatTable of HLT trigger objects. "
      "The row set is the union of objects passing the last save-tags filter "
      "of every path in pathNames and every filter in extraFilters.");

  desc.add<edm::InputTag>("triggerEvent", edm::InputTag("hltTriggerSummaryAOD", "", "HLT"))
      ->setComment("TriggerEvent summary product");
  desc.add<std::string>("tableName", "TrigObj")->setComment("nanoAOD table name");
  desc.add<std::vector<std::string>>("pathNames", {})
      ->setComment(
          "HLT paths: their last save-tags filter seeds the object set "
          "and each path gets a per-object bool column");
  desc.add<std::vector<std::string>>("extraFilters", {})
      ->setComment(
          "Additional HLT filter labels whose passing objects are included "
          "in the table (no extra bool column is added for these)");

  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(TrigObjP4FlatTableProducer);
