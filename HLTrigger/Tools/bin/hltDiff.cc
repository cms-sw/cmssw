/*      hltDiff: compare TriggerResults event by event
 *
 *      Compare two TriggerResults collections event by event.
 *      These can come from two collections in the same file(s), or from two different sets of files.
 */

#include <vector>
#include <set>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cstdio>
#include <iomanip>
#include <memory>
#include <algorithm>

#include <cstring>
#include <unistd.h>
#include <getopt.h>
#include <cstdio>
#include <cmath>

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>

#include <TFile.h>
#include <TCanvas.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TGraphAsymmErrors.h>

#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/FWLite/interface/Handle.h"
#include "DataFormats/FWLite/interface/Event.h"
#include "DataFormats/FWLite/interface/ChainEvent.h"
#include "HLTrigger/HLTcore/interface/HLTConfigData.h"

void error(std::ostream& out) { out << "Try 'hltDiff --help' for more information" << std::endl; }

void error(std::ostream& out, const char* message) {
  out << message << std::endl;
  error(out);
}

void error(std::ostream& out, const std::string& message) {
  out << message << std::endl;
  error(out);
}

class HLTConfigInterface {
public:
  virtual ~HLTConfigInterface() = default;
  virtual std::string const& processName() const = 0;
  virtual unsigned int size() const = 0;
  virtual unsigned int size(unsigned int trigger) const = 0;
  virtual std::string const& triggerName(unsigned int trigger) const = 0;
  virtual unsigned int triggerIndex(unsigned int trigger) const = 0;
  virtual std::string const& moduleLabel(unsigned int trigger, unsigned int module) const = 0;
  virtual std::string const& moduleType(unsigned int trigger, unsigned int module) const = 0;
  virtual bool prescaler(unsigned int trigger, unsigned int module) const = 0;
};

class HLTConfigDataEx : public HLTConfigInterface {
public:
  explicit HLTConfigDataEx(HLTConfigData data) : data_(std::move(data)), moduleTypes_(size()), prescalers_(size()) {
    for (unsigned int t = 0; t < data_.size(); ++t) {
      prescalers_[t].resize(size(t));
      moduleTypes_[t].resize(size(t));
      for (unsigned int m = 0; m < data_.size(t); ++m) {
        std::string type = data_.moduleType(moduleLabel(t, m));
        prescalers_[t][m] = (type == "HLTPrescaler");
        moduleTypes_[t][m] = &*moduleTypeSet_.insert(std::move(type)).first;
      }
    }
  }
  ~HLTConfigDataEx() override = default;
  std::string const& processName() const override { return data_.processName(); }

  unsigned int size() const override { return data_.size(); }

  unsigned int size(unsigned int trigger) const override { return data_.size(trigger); }

  virtual std::vector<std::string> const& triggerNames() const { return data_.triggerNames(); }

  std::string const& triggerName(unsigned int trigger) const override { return data_.triggerName(trigger); }

  unsigned int triggerIndex(unsigned int trigger) const override { return trigger; }

  std::string const& moduleLabel(unsigned int trigger, unsigned int module) const override {
    return data_.moduleLabel(trigger, module);
  }

  std::string const& moduleType(unsigned int trigger, unsigned int module) const override {
    return *moduleTypes_.at(trigger).at(module);
  }

  bool prescaler(unsigned int trigger, unsigned int module) const override {
    return prescalers_.at(trigger).at(module);
  }

private:
  HLTConfigData data_;
  std::set<std::string> moduleTypeSet_;
  std::vector<std::vector<std::string const*>> moduleTypes_;
  std::vector<std::vector<bool>> prescalers_;
};

const char* event_state(bool state) { return state ? "accepted" : "rejected"; }

class HLTCommonConfig {
public:
  enum class Index { First = 0, Second = 1 };

  class View : public HLTConfigInterface {
  public:
    View(HLTCommonConfig const& config, HLTCommonConfig::Index index) : config_(config), index_(index) {}
    ~View() override = default;
    std::string const& processName() const override;
    unsigned int size() const override;
    unsigned int size(unsigned int trigger) const override;
    std::string const& triggerName(unsigned int trigger) const override;
    unsigned int triggerIndex(unsigned int trigger) const override;
    std::string const& moduleLabel(unsigned int trigger, unsigned int module) const override;
    std::string const& moduleType(unsigned int trigger, unsigned int module) const override;
    bool prescaler(unsigned int trigger, unsigned int module) const override;

  private:
    HLTCommonConfig const& config_;
    Index index_;
  };

  HLTCommonConfig(HLTConfigDataEx const& first, HLTConfigDataEx const& second)
      : first_(first), second_(second), firstView_(*this, Index::First), secondView_(*this, Index::Second) {
    for (unsigned int f = 0; f < first.size(); ++f)
      for (unsigned int s = 0; s < second.size(); ++s)
        if (first.triggerName(f) == second.triggerName(s)) {
          triggerIndices_.push_back(std::make_pair(f, s));
          break;
        }
  }

  View const& getView(Index index) const {
    if (index == Index::First)
      return firstView_;
    else
      return secondView_;
  }

  std::string const& processName(Index index) const {
    if (index == Index::First)
      return first_.processName();
    else
      return second_.processName();
  }

  unsigned int size(Index index) const { return triggerIndices_.size(); }

  unsigned int size(Index index, unsigned int trigger) const {
    if (index == Index::First)
      return first_.size(trigger);
    else
      return second_.size(trigger);
  }

  std::string const& triggerName(Index index, unsigned int trigger) const {
    if (index == Index::First)
      return first_.triggerName(triggerIndices_.at(trigger).first);
    else
      return second_.triggerName(triggerIndices_.at(trigger).second);
  }

  unsigned int triggerIndex(Index index, unsigned int trigger) const {
    if (index == Index::First)
      return triggerIndices_.at(trigger).first;
    else
      return triggerIndices_.at(trigger).second;
  }

  std::string const& moduleLabel(Index index, unsigned int trigger, unsigned int module) const {
    if (index == Index::First)
      return first_.moduleLabel(triggerIndices_.at(trigger).first, module);
    else
      return second_.moduleLabel(triggerIndices_.at(trigger).second, module);
  }

  std::string const& moduleType(Index index, unsigned int trigger, unsigned int module) const {
    if (index == Index::First)
      return first_.moduleType(triggerIndices_.at(trigger).first, module);
    else
      return second_.moduleType(triggerIndices_.at(trigger).second, module);
  }

  bool prescaler(Index index, unsigned int trigger, unsigned int module) const {
    if (index == Index::First)
      return first_.prescaler(triggerIndices_.at(trigger).first, module);
    else
      return second_.prescaler(triggerIndices_.at(trigger).second, module);
  }

private:
  HLTConfigDataEx const& first_;
  HLTConfigDataEx const& second_;

  View firstView_;
  View secondView_;

  std::vector<std::pair<unsigned int, unsigned int>> triggerIndices_;
};

std::string const& HLTCommonConfig::View::processName() const { return config_.processName(index_); }

unsigned int HLTCommonConfig::View::size() const { return config_.size(index_); }

unsigned int HLTCommonConfig::View::size(unsigned int trigger) const { return config_.size(index_, trigger); }

std::string const& HLTCommonConfig::View::triggerName(unsigned int trigger) const {
  return config_.triggerName(index_, trigger);
}

unsigned int HLTCommonConfig::View::triggerIndex(unsigned int trigger) const {
  return config_.triggerIndex(index_, trigger);
}

std::string const& HLTCommonConfig::View::moduleLabel(unsigned int trigger, unsigned int module) const {
  return config_.moduleLabel(index_, trigger, module);
}

std::string const& HLTCommonConfig::View::moduleType(unsigned int trigger, unsigned int module) const {
  return config_.moduleType(index_, trigger, module);
}

bool HLTCommonConfig::View::prescaler(unsigned int trigger, unsigned int module) const {
  return config_.prescaler(index_, trigger, module);
}

enum State {
  Ready = edm::hlt::Ready,
  Pass = edm::hlt::Pass,
  Fail = edm::hlt::Fail,
  Exception = edm::hlt::Exception,
  Prescaled,
  Invalid
};

const char* path_state(State state) {
  static const char* message[] = {"not run", "accepted", "rejected", "exception", "prescaled", "invalid"};

  if (state > 0 and state < Invalid)
    return message[state];
  else
    return message[Invalid];
}

inline State prescaled_state(int state, int path, int module, HLTConfigInterface const& config) {
  if (state == Fail and config.prescaler(path, module))
    return Prescaled;
  return (State)state;
}

// return a copy of a string denoting an InputTag without the process name
// i.e.
//    "module"                  --> "module"
//    "module:instance"         --> "module:instance"
//    "module::process"         --> "module"
//    "module:instance:process" --> "module:instance"
//
std::string strip_process_name(std::string const& s) {
  if (std::count(s.begin(), s.end(), ':') == 2) {
    // remove the process name and the second ':' separator
    size_t end = s.find_last_of(':');
    if (end > 0 and s.at(end - 1) == ':')
      // no instance name, remove also the first ':' separator
      --end;
    return s.substr(0, end);
  } else {
    // no process name, return the string unchanged
    return s;
  }
}

void print_detailed_path_state(std::ostream& out, State state, int path, int module, HLTConfigInterface const& config) {
  auto const& label = config.moduleLabel(path, module);
  auto const& type = config.moduleType(path, module);

  out << "'" << path_state(state) << "'";
  if (state == Fail)
    out << " by module " << module << " '" << label << "' [" << type << "]";
  else if (state == Exception)
    out << " at module " << module << " '" << label << "' [" << type << "]";
}

void print_trigger_candidates(std::ostream& out, trigger::TriggerEvent const& summary, edm::InputTag const& filter) {
  // find the index of the collection of trigger candidates corresponding to the filter
  unsigned int index = summary.filterIndex(filter);

  if (index >= summary.sizeFilters()) {
    // the collection of trigger candidates corresponding to the filter could not be found
    out << "            not found\n";
    return;
  }

  if (summary.filterKeys(index).empty()) {
    // the collection of trigger candidates corresponding to the filter is empty
    out << "            none\n";
    return;
  }

  for (unsigned int i = 0; i < summary.filterKeys(index).size(); ++i) {
    auto key = summary.filterKeys(index)[i];
    auto id = summary.filterIds(index)[i];
    trigger::TriggerObject const& candidate = summary.getObjects().at(key);
    out << "            "
        << "filter id: " << id << ", "
        << "object id: " << candidate.id() << ", "
        << "pT: " << candidate.pt() << ", "
        << "eta: " << candidate.eta() << ", "
        << "phi: " << candidate.phi() << ", "
        << "mass: " << candidate.mass() << "\n";
  }
}

void print_trigger_collection(std::ostream& out, trigger::TriggerEvent const& summary, std::string const& tag) {
  auto iterator = std::find(summary.collectionTags().begin(), summary.collectionTags().end(), tag);
  if (iterator == summary.collectionTags().end()) {
    // the collection of trigger candidates could not be found
    out << "            not found\n";
    return;
  }

  unsigned int index = iterator - summary.collectionTags().begin();
  unsigned int begin = (index == 0) ? 0 : summary.collectionKey(index - 1);
  unsigned int end = summary.collectionKey(index);

  if (end == begin) {
    // the collection of trigger candidates is empty
    out << "            none\n";
    return;
  }

  for (unsigned int key = begin; key < end; ++key) {
    trigger::TriggerObject const& candidate = summary.getObjects().at(key);
    out << "            "
        << "object id: " << candidate.id() << ", "
        << "pT: " << candidate.pt() << ", "
        << "eta: " << candidate.eta() << ", "
        << "phi: " << candidate.phi() << ", "
        << "mass: " << candidate.mass() << "\n";
  }
}

std::string getProcessNameFromBranch(std::string const& branch) {
  std::vector<boost::iterator_range<std::string::const_iterator>> tokens;
  boost::split(tokens, branch, boost::is_any_of("_."), boost::token_compress_off);
  return boost::copy_range<std::string>(tokens[3]);
}

std::unique_ptr<HLTConfigDataEx> getHLTConfigData(fwlite::EventBase const& event, std::string process) {
  auto const& history = event.processHistory();
  if (process.empty()) {
    // determine the process name from the most recent "TriggerResults" object
    auto const& branch =
        event.getBranchNameFor(edm::Wrapper<edm::TriggerResults>::typeInfo(), "TriggerResults", "", process.c_str());
    process = getProcessNameFromBranch(branch);
  }

  edm::ProcessConfiguration config;
  if (not history.getConfigurationForProcess(process, config)) {
    std::cerr << "error: the process " << process << " is not in the Process History" << std::endl;
    exit(1);
  }
  const edm::ParameterSet* pset = edm::pset::Registry::instance()->getMapped(config.parameterSetID());
  if (pset == nullptr) {
    std::cerr << "error: the configuration for the process " << process << " is not available in the Provenance"
              << std::endl;
    exit(1);
  }
  return std::make_unique<HLTConfigDataEx>(HLTConfigData(pset));
}

struct TriggerDiff {
  TriggerDiff() : count(0), gained(0), lost(0), internal(0) {}

  unsigned int count;
  unsigned int gained;
  unsigned int lost;
  unsigned int internal;

  static std::string format(unsigned int value, char sign = '+') {
    if (value == 0)
      return std::string("-");

    char buffer[12];  // sign, 10 digits, null
    memset(buffer, 0, 12);

    unsigned int digit = 10;
    while (value > 0) {
      buffer[digit] = value % 10 + 48;
      value /= 10;
      --digit;
    }
    buffer[digit] = sign;

    return std::string(buffer + digit);
  }

  unsigned int total() const { return this->gained + this->lost + this->internal; }
};

std::ostream& operator<<(std::ostream& out, TriggerDiff diff) {
  out << std::setw(12) << diff.count << std::setw(12) << TriggerDiff::format(diff.gained, '+') << std::setw(12)
      << TriggerDiff::format(diff.lost, '-') << std::setw(12) << TriggerDiff::format(diff.internal, '~');
  return out;
}

class JsonOutputProducer {
private:
  static size_t tab_spaces;

  // static variables and methods for printing specific JSON elements
  static std::string indent(size_t _nTabs) {
    std::string str = "\n";
    while (_nTabs) {
      int nSpaces = tab_spaces;
      while (nSpaces) {
        str.push_back(' ');
        nSpaces--;
      }
      _nTabs--;
    }

    return str;
  }

  static std::string key(const std::string& _key, const std::string& _delim = "") {
    std::string str = "\"\":";
    str.insert(1, _key);
    str.append(_delim);

    return str;
  }

  static std::string key_string(const std::string& _key, const std::string& _string, const std::string& _delim = "") {
    std::string str = key(_key, _delim);
    str.push_back('"');
    str.append(_string);
    str.push_back('"');
    return str;
  }

  static std::string key_int(const std::string& _key, int _int, const std::string& _delim = "") {
    std::string str = key(_key, _delim);
    str.append(std::to_string(_int));

    return str;
  }

  static std::string string(const std::string& _string, const std::string& _delim = "") {
    std::string str = "\"\"";
    str.insert(1, _string);
    str.append(_delim);

    return str;
  }

  static std::string list_string(const std::vector<std::string>& _values, const std::string& _delim = "") {
    std::string str = "[";
    for (auto it = _values.begin(); it != _values.end(); ++it) {
      str.append(_delim);
      str.push_back('"');
      str.append(*it);
      str.push_back('"');
      if (it != --_values.end())
        str.push_back(',');
    }
    str.append(_delim);
    str.push_back(']');

    return str;
  }

public:
  bool writeJson;
  std::string out_filename_base;
  bool useSingleOutFile;
  // structs holding particular JSON objects
  struct JsonConfigurationBlock {
    std::string file_base;  // common part at the beginning of all files
    std::vector<std::string> files;
    std::string process;
    std::vector<std::string> skipped_triggers;

    std::string serialise(size_t _indent = 0) const {
      std::ostringstream json;
      json << indent(_indent);  // line
      json << key_string("file_base", file_base) << ',';
      json << indent(_indent);  // line
      json << key("files") << list_string(files) << ',';
      json << indent(_indent);  // line
      json << key_string("process", process) << ',';
      json << indent(_indent);  // line
      json << key("skipped_triggers") << list_string(skipped_triggers);

      return json.str();
    }

    void extractFileBase() {
      std::string file0 = files.at(0);
      // determining the last position at which all filenames have the same character
      for (size_t i = 0; i < file0.length(); ++i) {
        bool identicalInAll = true;
        char character = file0.at(i);
        for (std::string file : files) {
          if (file.at(i) == character)
            continue;
          identicalInAll = false;
          break;
        }
        if (!identicalInAll)
          break;
        file_base.push_back(character);
      }
      const unsigned int file_base_len = file_base.length();
      if (file_base_len < 1)
        return;
      // removing the file_base from each filename
      for (std::string& file : files) {
        file.erase(0, file_base_len);
      }
    }

    JsonConfigurationBlock() : file_base(""), files(0), process(""), skipped_triggers(0) {}
  };

  struct JsonConfiguration {
    JsonConfigurationBlock o;  // old
    JsonConfigurationBlock n;  // new
    bool prescales;
    int events;

    std::string serialise(size_t _indent = 0) const {
      std::ostringstream json;
      json << indent(_indent) << key("configuration") << '{';  // line open
      json << indent(_indent + 1) << key("o") << '{';          // line open
      json << o.serialise(_indent + 2);                        // block
      json << indent(_indent + 1) << "},";                     // line close
      json << indent(_indent + 1) << key("n") << '{';          // line open
      json << n.serialise(_indent + 2);                        // line block
      json << indent(_indent + 1) << "},";                     // line close
      std::string prescales_str = prescales ? "true" : "false";
      json << indent(_indent + 1) << key("prescales") << prescales_str << ',';  // line
      json << indent(_indent + 1) << key("events") << events;                   // line
      json << indent(_indent) << "}";                                           // line close

      return json.str();
    }

    JsonConfiguration() : o(), n() {}
  };

  struct JsonVars {
    std::vector<std::string> state;
    std::vector<std::string> trigger;
    std::vector<std::pair<int, int>> trigger_passed_count;  // <old, new>
    std::vector<std::string> label;
    std::vector<std::string> type;

    std::string serialise(size_t _indent = 0) const {
      std::ostringstream json;
      json << indent(_indent) << key("vars") << '{';                                 // line open
      json << indent(_indent + 1) << key("state") << list_string(state) << ',';      // line
      json << indent(_indent + 1) << key("trigger") << list_string(trigger) << ',';  // line
      json << indent(_indent + 1) << key("trigger_passed_count") << '[';             // line
      for (auto it = trigger_passed_count.begin(); it != trigger_passed_count.end(); ++it) {
        json << '{' << key("o") << (*it).first << ',' << key("n") << (*it).second << '}';
        if (it != trigger_passed_count.end() - 1)
          json << ',';
      }
      json << "],";
      json << indent(_indent + 1) << key("label") << list_string(label) << ',';  // line
      json << indent(_indent + 1) << key("type") << list_string(type);           // line
      json << indent(_indent) << '}';                                            // line close

      return json.str();
    }

    JsonVars() : state(0), trigger(0), trigger_passed_count(0), label(0), type(0) {}
  };

  // class members
  JsonConfiguration configuration;
  JsonVars vars;

private:
  unsigned int labelId(std::string labelName) {
    unsigned int id = std::find(vars.label.begin(), vars.label.end(), labelName) - vars.label.begin();
    if (id < vars.label.size())
      return id;
    vars.label.push_back(labelName);
    return vars.label.size() - 1;
  }

  unsigned int typeId(std::string typeName) {
    unsigned int id = std::find(vars.type.begin(), vars.type.end(), typeName) - vars.type.begin();
    if (id < vars.type.size())
      return id;
    vars.type.push_back(typeName);
    return vars.type.size() - 1;
  }

public:
  struct JsonEventState {
    State s;  // state
    int m;    // module id
    int l;    // label id
    int t;    // type id

    std::string serialise(size_t _indent = 0) const {
      std::ostringstream json;
      json << key_int("s", int(s));  // line
      // No more information needed if the state is 'accepted'
      if (s == State::Pass)
        return json.str();
      json << ',';
      json << key_int("m", m) << ',';
      json << key_int("l", l) << ',';
      json << key_int("t", t);

      return json.str();
    }

    JsonEventState() : s(State::Ready), m(-1), l(-1), t(-1) {}
    JsonEventState(State _s, int _m, int _l, int _t) : s(_s), m(_m), l(_l), t(_t) {}
  };

  struct JsonTriggerEventState {
    int tr;            // trigger id
    JsonEventState o;  // old
    JsonEventState n;  // new

    std::string serialise(size_t _indent = 0) const {
      std::ostringstream json;
      json << indent(_indent) << key_int("t", tr) << ',';                   // line
      json << indent(_indent) << key("o") << '{' << o.serialise() << "},";  // line
      json << indent(_indent) << key("n") << '{' << n.serialise() << "}";   // line

      return json.str();
    }

    JsonTriggerEventState() : tr(-1), o(), n() {}
    JsonTriggerEventState(int _tr) : tr(_tr), o(), n() {}
  };

  struct JsonEvent {
    int run;
    int lumi;
    int event;
    std::vector<JsonTriggerEventState> triggerStates;

    std::string serialise(size_t _indent = 0) const {
      std::ostringstream json;
      json << indent(_indent) << '{' << "\"r\"" << ':' << run << ",\"l\":" << lumi << ",\"e\":" << event
           << ",\"t\":[";  // line open
      for (auto it = triggerStates.begin(); it != triggerStates.end(); ++it) {
        json << '{';                           // line open
        json << (*it).serialise(_indent + 2);  // block
        json << indent(_indent + 1) << '}';    // line close
        if (it != --triggerStates.end())
          json << ',';
      }
      json << indent(_indent) << ']' << '}';  // line close

      return json.str();
    }

    JsonEvent(int _run, int _lumi, int _event) : run(_run), lumi(_lumi), event(_event), triggerStates(0) {}

    JsonTriggerEventState& pushTrigger(int _tr) {
      // check whether the last trigger is the one
      if (!triggerStates.empty()) {
        JsonTriggerEventState& lastTrigger = triggerStates.back();
        if (lastTrigger.tr == _tr)
          return lastTrigger;
      }
      triggerStates.push_back(JsonTriggerEventState(_tr));
      return triggerStates.back();
    }
  };

  // class members
  std::map<int, std::vector<JsonEvent>> m_run_events;

  // methods
  JsonOutputProducer(bool _writeJson, std::string _file_name)
      : writeJson(_writeJson), out_filename_base(std::move(_file_name)) {
    useSingleOutFile = out_filename_base.length() > 0;
  }

  JsonEvent& pushEvent(int _run, int _lumi, int _event) {
    // ensuring that this RUN is present in the producer
    if ((m_run_events.count(_run) == 0 && !useSingleOutFile) || m_run_events.empty())
      m_run_events.emplace(_run, std::vector<JsonEvent>());
    std::vector<JsonEvent>& v_events = useSingleOutFile ? m_run_events.begin()->second : m_run_events.at(_run);
    // check whether the last  event is the one
    if (!v_events.empty()) {
      JsonEvent& lastEvent = v_events.back();
      if (lastEvent.run == _run && lastEvent.lumi == _lumi && lastEvent.event == _event)
        return lastEvent;
    }
    v_events.push_back(JsonEvent(_run, _lumi, _event));
    return v_events.back();
  }

  JsonEventState eventState(State _s, int _m, const std::string& _l, const std::string& _t) {
    return JsonEventState(_s, _m, this->labelId(_l), this->typeId(_t));
  }

  std::string output_filename_base(int _run) const {
    if (useSingleOutFile)
      return out_filename_base;

    char name[1000];
    sprintf(name,
            "DQM_V0001_R%.9d__OLD_%s__NEW_%s_DQM",
            _run,
            configuration.o.process.c_str(),
            configuration.n.process.c_str());

    return std::string(name);
  }

  void write() {
    if (!writeJson)
      return;
    std::set<std::string> filesCreated;
    std::ofstream out_file;
    if (!m_run_events.empty()) {
      // Creating a separate file for each run
      for (const auto& runEvents : m_run_events) {
        const int run = runEvents.first;
        const std::vector<JsonEvent>& v_events = runEvents.second;
        // Writing the output to a JSON file
        std::string output_name = output_filename_base(run) += ".json";
        out_file.open(output_name, std::ofstream::out);
        out_file << '{';  // line open
        out_file << configuration.serialise(1) << ',';
        out_file << vars.serialise(1) << ',';
        // writing block for each event
        out_file << indent(1) << key("events") << '[';  // line open
        for (auto it = v_events.begin(); it != v_events.end(); ++it) {
          out_file << (*it).serialise(2);
          if (it != --v_events.end())
            out_file << ',';
        }
        out_file << indent(1) << ']';  // line close
        out_file << indent(0) << "}";  // line close
        out_file.close();
        // Adding file name to the list of created files
        filesCreated.insert(output_name);
      }
    } else {
      // Creating a single file containing with only configuration part
      std::string output_name = output_filename_base(0) += ".json";
      out_file.open(output_name, std::ofstream::out);
      out_file << '{';  // line open
      out_file << configuration.serialise(1) << ',';
      out_file << vars.serialise(1) << ',';
      // writing block for each event
      out_file << indent(1) << key("events") << '[';  // line open
      // for (std::vector<JsonEvent>::const_iterator it = v_events.begin(); it != v_events.end(); ++it) {
      //   out_file << (*it).serialise(2);
      //   if (it != --v_events.end()) out_file << ',';
      // }
      out_file << indent(1) << ']';  // line close
      out_file << indent(0) << "}";  // line close
      out_file.close();
      // Adding file name to the list of created files
      filesCreated.insert(output_name);
    }

    if (!filesCreated.empty()) {
      std::cout << "Created the following JSON files:" << std::endl;
      for (const std::string& filename : filesCreated)
        std::cout << " " << filename << std::endl;
    }
  }
};
size_t JsonOutputProducer::tab_spaces = 0;

class SummaryOutputProducer {
private:
  const JsonOutputProducer& json;
  int run;

  struct Pair {
    double v;
    double e;

    Pair(double _v, double _e) : v(_v), e(_e){};
    Pair(int _v, int _e) : v(_v), e(_e){};
  };

  struct Event {
    int run;
    int lumi;
    int event;

    Event(int _run, int _lumi, int _event) : run(_run), lumi(_lumi), event(_event){};
    bool operator<(const Event& b) const { return std::tie(run, lumi, event) < std::tie(b.run, b.lumi, b.event); }
  };

  struct GenericSummary {
    const JsonOutputProducer& json;
    int id;
    std::string name;
    std::set<Event> v_gained;
    std::set<Event> v_lost;
    std::set<Event> v_changed;

    GenericSummary(int _id, const JsonOutputProducer& _json, const std::vector<std::string>& _names)
        : json(_json), id(_id) {
      name = _names.at(id);
    }

    int addEntry(const JsonOutputProducer::JsonEvent& _event, const int _triggerIndex) {
      const JsonOutputProducer::JsonTriggerEventState& state = _event.triggerStates.at(_triggerIndex);
      const Event event = Event(_event.run, _event.lumi, _event.event);
      int moduleId = state.o.l;
      if (state.o.s == State::Pass && state.n.s == State::Fail) {
        moduleId = state.n.l;
        v_lost.insert(event);
      } else if (state.o.s == State::Fail && state.n.s == State::Pass) {
        v_gained.insert(event);
      } else if (state.o.s == State::Fail && state.n.s == State::Fail) {
        v_changed.insert(event);
      }

      return moduleId;
    }

    Pair gained() const { return Pair(double(v_gained.size()), sqrt(double(v_gained.size()))); }

    Pair lost() const { return Pair(double(v_lost.size()), sqrt(double(v_lost.size()))); }

    Pair changed() const { return Pair(double(v_changed.size()), sqrt(double(v_changed.size()))); }

    bool keepForC() const { return !v_changed.empty(); }

    bool keepForGL() const { return !v_gained.empty() || !v_lost.empty(); }
  };

  struct TriggerSummary : GenericSummary {
    int accepted_o;
    int accepted_n;
    std::map<int, GenericSummary> m_modules;

    TriggerSummary(int _id, const JsonOutputProducer& _json)
        : GenericSummary(_id, _json, _json.vars.trigger),
          accepted_o(_json.vars.trigger_passed_count.at(id).first),
          accepted_n(_json.vars.trigger_passed_count.at(id).second) {}

    void addEntry(const JsonOutputProducer::JsonEvent& _event,
                  const int _triggerIndex,
                  const std::vector<std::string>& _moduleNames) {
      int moduleLabelId = GenericSummary::addEntry(_event, _triggerIndex);
      // Updating number of events affected by the particular module
      if (m_modules.count(moduleLabelId) == 0)
        m_modules.emplace(moduleLabelId, GenericSummary(moduleLabelId, json, _moduleNames));
      m_modules.at(moduleLabelId).addEntry(_event, _triggerIndex);
    }

    Pair gained(int type = 0) const {
      Pair gained(GenericSummary::gained());
      if (type == 0)
        return gained;  // Absolute number of affected events
      double all(accepted_n);
      Pair fraction = Pair(gained.v / (all + 1e-10), sqrt(all) / (all + 1e-10));
      if (type == 1)
        return fraction;  // Relative number of affected events with respect to all accepted
      if (type == 2)
        return Pair(std::max(0.0, fraction.v - fraction.e), 0.0);  // Smallest value given the uncertainty
      return Pair(fraction.v / (fraction.e + 1e-10), 0.0);         // Significance of the effect as N std. deviations
    }

    Pair lost(int type = 0) const {
      Pair lost(GenericSummary::lost());
      if (type == 0)
        return lost;
      double all(accepted_o);
      Pair fraction = Pair(lost.v / (all + 1e-10), sqrt(all) / (all + 1e-10));
      if (type == 1)
        return fraction;
      if (type == 2)
        return Pair(std::max(0.0, fraction.v - fraction.e), 0.0);  // Smallest value given the uncertainty
      return Pair(fraction.v / (fraction.e + 1e-10), 0.0);
    }

    Pair changed(int type = 0) const {
      Pair changed(GenericSummary::changed());
      if (type == 0)
        return changed;
      double all(json.configuration.events - accepted_o);
      Pair fraction = Pair(changed.v / (all + 1e-10), sqrt(all) / (all + 1e-10));
      if (type == 1)
        return fraction;
      if (type == 2)
        return Pair(std::max(0.0, fraction.v - fraction.e), 0.0);  // Smallest value given the uncertainty
      return Pair(fraction.v / (fraction.e + 1e-10), 0.0);
    }
  };

private:
  std::map<int, TriggerSummary> m_triggerSummary;
  std::map<int, GenericSummary> m_moduleSummary;

  void prepareSummaries(const int _run, const std::vector<JsonOutputProducer::JsonEvent>& _events) {
    this->run = _run;
    // Initialising the summary objects for trigger/module
    m_triggerSummary.clear();
    m_moduleSummary.clear();
    const size_t nTriggers(json.vars.trigger.size());
    const size_t nModules(json.vars.label.size());
    for (size_t i = 0; i < nTriggers; ++i)
      m_triggerSummary.emplace(i, TriggerSummary(i, json));
    for (size_t i = 0; i < nModules; ++i)
      m_moduleSummary.emplace(i, GenericSummary(i, json, json.vars.label));

    // Add each affected trigger in each event to the trigger/module summary objects
    for (const JsonOutputProducer::JsonEvent& event : _events) {
      for (size_t iTrigger = 0; iTrigger < event.triggerStates.size(); ++iTrigger) {
        const JsonOutputProducer::JsonTriggerEventState& state = event.triggerStates.at(iTrigger);
        m_triggerSummary.at(state.tr).addEntry(event, iTrigger, json.vars.label);
        const int moduleId = state.o.s == State::Fail ? state.o.l : state.n.l;
        m_moduleSummary.at(moduleId).addEntry(event, iTrigger);
      }
    }
  }

  std::string writeHistograms() const {
    std::map<std::string, TH1*> m_histo;
    // Counting the numbers of bins for different types of histograms
    // *_c - changed; *_gl - gained or lost
    int nTriggers(0), nTriggers_c(0), nTriggers_gl(0), nModules_c(0), nModules_gl(0);

    for (const auto& idSummary : m_triggerSummary) {
      if (idSummary.second.accepted_o > 0)
        ++nTriggers;
      if (idSummary.second.keepForGL())
        ++nTriggers_gl;
      if (idSummary.second.keepForC())
        ++nTriggers_c;
    }
    for (const auto& idSummary : m_moduleSummary) {
      if (idSummary.second.keepForGL())
        ++nModules_gl;
      if (idSummary.second.keepForC())
        ++nModules_c;
    }
    // Manually increasing N bins to have histograms with meaningful axis ranges
    nTriggers = std::max(1, nTriggers);
    nTriggers_gl = std::max(1, nTriggers_gl);
    nTriggers_c = std::max(1, nTriggers_c);
    nModules_c = std::max(1, nModules_c);
    nModules_gl = std::max(1, nModules_gl);

    // Initialising overview histograms
    std::string name = "trigger_accepted";
    m_histo.emplace(name, new TH1F(name.c_str(), ";;Events accepted^{OLD}", nTriggers, 0, nTriggers));
    name = "trigger_gained";
    m_histo.emplace(name, new TH1F(name.c_str(), ";;Events gained", nTriggers_gl, 0, nTriggers_gl));
    name = "trigger_lost";
    m_histo.emplace(name, new TH1F(name.c_str(), ";;Events lost", nTriggers_gl, 0, nTriggers_gl));
    name = "trigger_changed";
    m_histo.emplace(name, new TH1F(name.c_str(), ";;Events changed", nTriggers_c, 0, nTriggers_c));
    name = "trigger_gained_frac";
    m_histo.emplace(name, new TH1F(name.c_str(), ";;#frac{gained}{accepted}", nTriggers_gl, 0, nTriggers_gl));
    name = "trigger_lost_frac";
    m_histo.emplace(name, new TH1F(name.c_str(), ";;#frac{lost}{accepted}", nTriggers_gl, 0, nTriggers_gl));
    name = "trigger_changed_frac";
    m_histo.emplace(name, new TH1F(name.c_str(), ";;#frac{changed}{all - accepted}", nTriggers_c, 0, nTriggers_c));
    name = "module_changed";
    m_histo.emplace(name, new TH1F(name.c_str(), ";;Events changed", nModules_c, 0, nModules_c));
    name = "module_gained";
    m_histo.emplace(name, new TH1F(name.c_str(), ";;Events gained", nModules_gl, 0, nModules_gl));
    name = "module_lost";
    m_histo.emplace(name, new TH1F(name.c_str(), ";;Events lost", nModules_gl, 0, nModules_gl));

    // Filling the per-trigger bins in the summary histograms
    size_t bin(0), bin_c(0), bin_gl(0);
    for (const auto& idSummary : m_triggerSummary) {
      const TriggerSummary& summary = idSummary.second;
      if (summary.accepted_o > 0) {
        ++bin;
        // Setting bin contents
        m_histo.at("trigger_accepted")->SetBinContent(bin, summary.accepted_o);
        // Setting bin labels
        m_histo.at("trigger_accepted")->GetXaxis()->SetBinLabel(bin, summary.name.c_str());
      }
      if (summary.keepForGL()) {
        ++bin_gl;
        // Setting bin contents
        m_histo.at("trigger_gained")->SetBinContent(bin_gl, summary.gained().v);
        m_histo.at("trigger_lost")->SetBinContent(bin_gl, -summary.lost().v);
        m_histo.at("trigger_gained_frac")->SetBinContent(bin_gl, summary.gained(1).v);
        m_histo.at("trigger_lost_frac")->SetBinContent(bin_gl, -summary.lost(1).v);
        // Setting bin errors
        m_histo.at("trigger_gained_frac")->SetBinError(bin_gl, summary.gained(1).e);
        m_histo.at("trigger_lost_frac")->SetBinError(bin_gl, -summary.lost(1).e);
        // Setting bin labels
        m_histo.at("trigger_gained")->GetXaxis()->SetBinLabel(bin_gl, summary.name.c_str());
        m_histo.at("trigger_lost")->GetXaxis()->SetBinLabel(bin_gl, summary.name.c_str());
        m_histo.at("trigger_gained_frac")->GetXaxis()->SetBinLabel(bin_gl, summary.name.c_str());
        m_histo.at("trigger_lost_frac")->GetXaxis()->SetBinLabel(bin_gl, summary.name.c_str());
      }
      if (summary.keepForC()) {
        ++bin_c;
        // Setting bin contents
        m_histo.at("trigger_changed")->SetBinContent(bin_c, summary.changed().v);
        m_histo.at("trigger_changed_frac")->SetBinContent(bin_c, summary.changed(1).v);
        // Setting bin errors
        m_histo.at("trigger_changed_frac")->SetBinError(bin_c, summary.changed(1).e);
        // Setting bin labels
        m_histo.at("trigger_changed")->GetXaxis()->SetBinLabel(bin_c, summary.name.c_str());
        m_histo.at("trigger_changed_frac")->GetXaxis()->SetBinLabel(bin_c, summary.name.c_str());
      }
    }

    // Filling the per-module bins in the summary histograms
    bin = 0;
    bin_c = 0;
    bin_gl = 0;
    for (const auto& idSummary : m_moduleSummary) {
      ++bin;
      const GenericSummary& summary = idSummary.second;
      if (summary.keepForGL()) {
        ++bin_gl;
        // Setting bin contents
        m_histo.at("module_gained")->SetBinContent(bin_gl, summary.gained().v);
        m_histo.at("module_lost")->SetBinContent(bin_gl, -summary.lost().v);
        // Setting bin labels
        m_histo.at("module_gained")->GetXaxis()->SetBinLabel(bin_gl, summary.name.c_str());
        m_histo.at("module_lost")->GetXaxis()->SetBinLabel(bin_gl, summary.name.c_str());
      }
      if (summary.keepForC()) {
        ++bin_c;
        // Setting bin contents
        m_histo.at("module_changed")->SetBinContent(bin_c, summary.changed().v);
        // Setting bin labels
        m_histo.at("module_changed")->GetXaxis()->SetBinLabel(bin_c, summary.name.c_str());
      }
    }

    // Styling the histograms
    for (const auto& nameHisto : m_histo) {
      const std::string name = nameHisto.first;
      TH1* histo = nameHisto.second;
      if (name.find("gained") != std::string::npos || name.find("changed") != std::string::npos) {
        if (name.find("frac") != std::string::npos)
          histo->GetYaxis()->SetRangeUser(0.0, 1.0);
      }
      if (name.find("lost") != std::string::npos) {
        if (name.find("frac") != std::string::npos)
          histo->GetYaxis()->SetRangeUser(-1.0, 0.0);
      }
    }

    // Storing histograms to a ROOT file
    std::string file_name = json.output_filename_base(this->run) += ".root";
    auto out_file = new TFile(file_name.c_str(), "RECREATE");
    // Storing the histograms in a proper folder according to the DQM convention
    char savePath[1000];
    sprintf(savePath, "DQMData/Run %d/HLT/Run summary/EventByEvent/", this->run);
    out_file->mkdir(savePath);
    gDirectory->cd(savePath);
    gDirectory->Write();
    for (const auto& nameHisto : m_histo)
      nameHisto.second->Write(nameHisto.first.c_str());
    out_file->Close();

    return file_name;
  }

  std::string writeCSV_trigger() const {
    std::string file_name = json.output_filename_base(this->run) += "_trigger.csv";
    FILE* out_file = fopen((file_name).c_str(), "w");

    fprintf(out_file,
            "Total,Accepted OLD,Accepted NEW,Gained,Lost,|G|/A_N + "
            "|L|/AO,sigma(AN)+sigma(AO),Changed,C/(T-AO),sigma(T-AO),trigger\n");
    for (const auto& idSummary : m_triggerSummary) {
      const SummaryOutputProducer::TriggerSummary& S = idSummary.second;
      fprintf(out_file,
              "%d,%d,%d,%+.f,%+.f,%.2f%%,%.2f%%,~%.f,~%.2f%%,%.2f%%,%s\n",
              this->json.configuration.events,
              S.accepted_o,
              S.accepted_n,
              S.gained().v,
              -1.0 * S.lost().v,
              (S.gained(1).v + S.lost(1).v) * 100.0,
              (S.gained(1).e + S.lost(1).e) * 100.0,
              S.changed().v,
              S.changed(1).v * 100.0,
              S.changed(1).e * 100.0,
              S.name.c_str());
    }

    fclose(out_file);

    return file_name;
  }

  std::string writeCSV_module() const {
    std::string file_name = json.output_filename_base(this->run) += "_module.csv";
    FILE* out_file = fopen((file_name).c_str(), "w");

    fprintf(out_file, "Total,Gained,Lost,Changed,module\n");
    for (const auto& idSummary : m_moduleSummary) {
      const SummaryOutputProducer::GenericSummary& S = idSummary.second;
      fprintf(out_file,
              "%d,+%.f,-%.f,~%.f,%s\n",
              this->json.configuration.events,
              S.gained().v,
              S.lost().v,
              S.changed().v,
              S.name.c_str());
    }

    fclose(out_file);

    return file_name;
  }

public:
  bool storeROOT;
  bool storeCSV;

  SummaryOutputProducer(const JsonOutputProducer& _json, bool _storeROOT, bool _storeCSV)
      : json(_json), run(0), storeROOT(_storeROOT), storeCSV(_storeCSV) {}

  void write() {
    std::vector<std::string> filesCreated;
    // Processing every run from the JSON producer
    if (!json.m_run_events.empty()) {
      for (const auto& runEvents : json.m_run_events) {
        prepareSummaries(runEvents.first, runEvents.second);
        if (storeROOT) {
          filesCreated.push_back(writeHistograms());
        }
        if (storeCSV) {
          filesCreated.push_back(writeCSV_trigger());
          filesCreated.push_back(writeCSV_module());
        }
      }
    } else {
      if (storeROOT) {
        filesCreated.push_back(writeHistograms());
      }
      if (storeCSV) {
        filesCreated.push_back(writeCSV_trigger());
        filesCreated.push_back(writeCSV_module());
      }
    }

    if (!filesCreated.empty()) {
      std::cout << "Created the following summary files:" << std::endl;
      for (const std::string& filename : filesCreated)
        std::cout << " " << filename << std::endl;
    }
  }
};

bool check_file(std::string const& file) {
  std::unique_ptr<TFile> f(TFile::Open(file.c_str()));
  return (f and not f->IsZombie());
}

bool check_files(std::vector<std::string> const& files) {
  bool flag = true;
  for (auto const& file : files)
    if (not check_file(file)) {
      flag = false;
      std::cerr << "hltDiff: error: file " << file << " does not exist, or is not a regular file." << std::endl;
    }
  return flag;
}

class HltDiff {
public:
  std::vector<std::string> old_files;
  std::string old_process;
  std::vector<std::string> new_files;
  std::string new_process;
  unsigned int max_events;
  bool ignore_prescales;
  bool csv_out;
  bool json_out;
  bool root_out;
  std::string output_file;
  bool file_check;
  bool debug;
  bool quiet;
  unsigned int verbose;

  HltDiff()
      : old_files(0),
        old_process(""),
        new_files(0),
        new_process(""),
        max_events(1e9),
        ignore_prescales(true),
        csv_out(false),
        json_out(false),
        root_out(false),
        output_file(""),
        file_check(false),
        debug(false),
        quiet(false),
        verbose(0) {}

  void compare() const {
    std::shared_ptr<fwlite::ChainEvent> old_events;
    std::shared_ptr<fwlite::ChainEvent> new_events;

    if (not file_check or check_files(old_files))
      old_events = std::make_shared<fwlite::ChainEvent>(old_files);
    else
      return;

    if (new_files.size() == 1 and new_files[0] == "-")
      new_events = old_events;
    else if (not file_check or check_files(new_files))
      new_events = std::make_shared<fwlite::ChainEvent>(new_files);
    else
      return;

    // creating the structure holding data for JSON and ROOT output
    JsonOutputProducer json(json_out, output_file);

    json.configuration.prescales = ignore_prescales;
    // setting the old configuration
    json.configuration.o.process = old_process;
    json.configuration.o.files = old_files;
    json.configuration.o.extractFileBase();
    // setting the new configuration
    json.configuration.n.process = new_process;
    json.configuration.n.files = new_files;
    json.configuration.n.extractFileBase();

    // initialising configurations to be compared
    std::unique_ptr<HLTConfigDataEx> old_config_data;
    std::unique_ptr<HLTConfigDataEx> new_config_data;
    std::unique_ptr<HLTCommonConfig> common_config;
    HLTConfigInterface const* old_config = nullptr;
    HLTConfigInterface const* new_config = nullptr;

    unsigned int counter = 0;
    unsigned int skipped = 0;
    unsigned int affected = 0;
    bool new_run = true;
    std::vector<TriggerDiff> differences;

    // loop over the reference events
    const unsigned int nEvents = std::min((int)old_events->size(), (int)max_events);
    const unsigned int counter_denominator = std::max(1, int(nEvents / 10));
    for (old_events->toBegin(); not old_events->atEnd(); ++(*old_events)) {
      // printing progress on every 10%
      if (counter % (counter_denominator) == 0) {
        std::cout << "Processed events: " << counter << " out of " << nEvents << " ("
                  << 10 * counter / (counter_denominator) << "%)" << std::endl;
      }

      // seek the same event in the "new" files
      edm::EventID const& id = old_events->id();
      if (new_events != old_events and not new_events->to(id)) {
        if (debug)
          std::cerr << "run " << id.run() << ", lumi " << id.luminosityBlock() << ", event " << id.event()
                    << ": not found in the 'new' files, skipping." << std::endl;
        ++skipped;
        continue;
      }

      // read the TriggerResults and TriggerEvent
      fwlite::Handle<edm::TriggerResults> old_results_h;
      edm::TriggerResults const* old_results = nullptr;
      old_results_h.getByLabel<fwlite::Event>(*old_events->event(), "TriggerResults", "", old_process.c_str());
      if (old_results_h.isValid())
        old_results = old_results_h.product();
      else {
        if (debug)
          std::cerr << "run " << id.run() << ", lumi " << id.luminosityBlock() << ", event " << id.event()
                    << ": 'old' TriggerResults not found, skipping." << std::endl;
        continue;
      }

      fwlite::Handle<trigger::TriggerEvent> old_summary_h;
      trigger::TriggerEvent const* old_summary = nullptr;
      old_summary_h.getByLabel<fwlite::Event>(*old_events->event(), "hltTriggerSummaryAOD", "", old_process.c_str());
      if (old_summary_h.isValid())
        old_summary = old_summary_h.product();

      fwlite::Handle<edm::TriggerResults> new_results_h;
      edm::TriggerResults const* new_results = nullptr;
      new_results_h.getByLabel<fwlite::Event>(*new_events->event(), "TriggerResults", "", new_process.c_str());
      if (new_results_h.isValid())
        new_results = new_results_h.product();
      else {
        if (debug)
          std::cerr << "run " << id.run() << ", lumi " << id.luminosityBlock() << ", event " << id.event()
                    << ": 'new' TriggerResults not found, skipping." << std::endl;
        continue;
      }

      fwlite::Handle<trigger::TriggerEvent> new_summary_h;
      trigger::TriggerEvent const* new_summary = nullptr;
      new_summary_h.getByLabel<fwlite::Event>(*new_events->event(), "hltTriggerSummaryAOD", "", new_process.c_str());
      if (new_summary_h.isValid())
        new_summary = new_summary_h.product();

      // initialise the trigger configuration
      if (new_run) {
        new_run = false;
        old_events->fillParameterSetRegistry();
        new_events->fillParameterSetRegistry();

        old_config_data = getHLTConfigData(*old_events->event(), old_process);
        new_config_data = getHLTConfigData(*new_events->event(), new_process);
        if (new_config_data->triggerNames() == old_config_data->triggerNames()) {
          old_config = old_config_data.get();
          new_config = new_config_data.get();
        } else {
          common_config = std::make_unique<HLTCommonConfig>(*old_config_data, *new_config_data);
          old_config = &common_config->getView(HLTCommonConfig::Index::First);
          new_config = &common_config->getView(HLTCommonConfig::Index::Second);
          std::cout << "Warning: old and new TriggerResults come from different HLT menus. Only the common "
                    << old_config->size() << " triggers are compared.\n"
                    << std::endl;
        }

        differences.clear();
        differences.resize(old_config->size());

        // adding the list of selected triggers to JSON output
        std::vector<std::string> states_str;
        for (int i = State::Ready; i != State::Invalid; i++)
          states_str.push_back(std::string(path_state(static_cast<State>(i))));
        json.vars.state = states_str;
        for (size_t triggerId = 0; triggerId < old_config->size(); ++triggerId) {
          json.vars.trigger.push_back(old_config->triggerName(triggerId));
          json.vars.trigger_passed_count.push_back(std::pair<int, int>(0, 0));
        }
        // getting names of triggers existing only in the old configuration
        for (auto const& it : old_config_data->triggerNames()) {
          if (std::find(json.vars.trigger.begin(), json.vars.trigger.end(), it) != json.vars.trigger.end())
            continue;
          json.configuration.o.skipped_triggers.push_back(it);
        }
        // getting names of triggers existing only in the new configuration
        for (auto const& it : new_config_data->triggerNames()) {
          if (std::find(json.vars.trigger.begin(), json.vars.trigger.end(), it) != json.vars.trigger.end())
            continue;
          json.configuration.n.skipped_triggers.push_back(it);
        }
      }

      // compare the TriggerResults
      bool needs_header = true;
      bool event_affected = false;
      for (unsigned int p = 0; p < old_config->size(); ++p) {
        // FIXME explicitly converting the indices is a hack, it should be properly encapsulated instead
        unsigned int old_index = old_config->triggerIndex(p);
        unsigned int new_index = new_config->triggerIndex(p);
        State old_state = prescaled_state(old_results->state(old_index), p, old_results->index(old_index), *old_config);
        State new_state = prescaled_state(new_results->state(new_index), p, new_results->index(new_index), *new_config);

        if (old_state == Pass) {
          ++differences.at(p).count;
        }
        if (old_state == Pass)
          ++json.vars.trigger_passed_count.at(p).first;
        if (new_state == Pass)
          ++json.vars.trigger_passed_count.at(p).second;

        bool trigger_affected = false;
        if (not ignore_prescales or (old_state != Prescaled and new_state != Prescaled)) {
          if (old_state == Pass and new_state != Pass) {
            ++differences.at(p).lost;
            trigger_affected = true;
          } else if (old_state != Pass and new_state == Pass) {
            ++differences.at(p).gained;
            trigger_affected = true;
          } else if (old_results->index(old_index) != new_results->index(new_index)) {
            ++differences.at(p).internal;
            trigger_affected = true;
          }
        }

        if (not trigger_affected)
          continue;

        event_affected = true;
        const unsigned int old_moduleIndex = old_results->index(old_index);
        const unsigned int new_moduleIndex = new_results->index(new_index);
        // storing the event to JSON, without any trigger results for the moment
        JsonOutputProducer::JsonEvent& event = json.pushEvent(id.run(), id.luminosityBlock(), id.event());
        JsonOutputProducer::JsonTriggerEventState& state = event.pushTrigger(p);
        state.o = json.eventState(old_state,
                                  old_moduleIndex,
                                  old_config->moduleLabel(p, old_moduleIndex),
                                  old_config->moduleType(p, old_moduleIndex));
        state.n = json.eventState(new_state,
                                  new_moduleIndex,
                                  new_config->moduleLabel(p, new_moduleIndex),
                                  new_config->moduleType(p, new_moduleIndex));

        if (verbose > 0) {
          if (needs_header) {
            needs_header = false;
            std::cout << "run " << id.run() << ", lumi " << id.luminosityBlock() << ", event " << id.event() << ": "
                      << "old result is '" << event_state(old_results->accept()) << "', "
                      << "new result is '" << event_state(new_results->accept()) << "'" << std::endl;
          }
          // print the Trigger path and filter responsible for the discrepancy
          std::cout << "    Path " << old_config->triggerName(p) << ":\n"
                    << "        old state is ";
          print_detailed_path_state(std::cout, old_state, p, old_moduleIndex, *old_config);
          std::cout << ",\n"
                    << "        new state is ";
          print_detailed_path_state(std::cout, new_state, p, new_moduleIndex, *new_config);
          std::cout << std::endl;
        }
        if (verbose > 1 and old_summary and new_summary) {
          // print TriggerObjects for the filter responsible for the discrepancy
          unsigned int module = std::min(old_moduleIndex, new_moduleIndex);
          std::cout << "    Filter " << old_config->moduleLabel(p, module) << ":\n";
          std::cout << "        old trigger candidates:\n";
          print_trigger_candidates(std::cout,
                                   *old_summary,
                                   edm::InputTag(old_config->moduleLabel(p, module), "", old_config->processName()));
          std::cout << "        new trigger candidates:\n";
          print_trigger_candidates(std::cout,
                                   *new_summary,
                                   edm::InputTag(new_config->moduleLabel(p, module), "", new_config->processName()));
        }
        if (verbose > 0)
          std::cout << std::endl;
      }
      if (event_affected)
        ++affected;

      // compare the TriggerEvent
      if (event_affected and verbose > 2 and old_summary and new_summary) {
        std::map<std::string, std::pair<std::string, std::string>> collections;
        for (auto const& old_collection : old_summary->collectionTags())
          collections[strip_process_name(old_collection)].first = old_collection;
        for (auto const& new_collection : new_summary->collectionTags())
          collections[strip_process_name(new_collection)].second = new_collection;

        for (auto const& collection : collections) {
          std::cout << "    Collection " << collection.first << ":\n";
          std::cout << "        old trigger candidates:\n";
          print_trigger_collection(std::cout, *old_summary, collection.second.first);
          std::cout << "        new trigger candidates:\n";
          print_trigger_collection(std::cout, *new_summary, collection.second.second);
          std::cout << std::endl;
        }
      }

      ++counter;
      if (nEvents and counter >= nEvents)
        break;
    }

    json.configuration.events = counter;

    if (not counter) {
      std::cout << "There are no common events between the old and new files";
      if (skipped)
        std::cout << ", " << skipped << " events were skipped";
      std::cout << "." << std::endl;
    } else {
      std::cout << "Found " << counter << " matching events, out of which " << affected
                << " have different HLT results";
      if (skipped)
        std::cout << ", " << skipped << " events were skipped";
      std::cout << "\n" << std::endl;
    }
    // Printing the summary of affected triggers with affected-event counts
    if (!quiet) {
      bool summaryHeaderPrinted = false;
      for (size_t p = 0; p < old_config->size(); ++p) {
        if (differences.at(p).total() < 1)
          continue;
        if (!summaryHeaderPrinted)
          std::cout << std::setw(12) << "Events" << std::setw(12) << "Accepted" << std::setw(12) << "Gained"
                    << std::setw(12) << "Lost" << std::setw(12) << "Other"
                    << "  "
                    << "Trigger" << std::endl;
        std::cout << std::setw(12) << counter << differences.at(p) << "  " << old_config->triggerName(p) << std::endl;
        summaryHeaderPrinted = true;
      }
    }

    // writing all the required output
    json.write();  // to JSON file for interactive visualisation
    SummaryOutputProducer summary(json, this->root_out, this->csv_out);
    summary.write();  // to ROOT file for fast validation with static plots
  }

  void usage(std::ostream& out) const {
    out << "\
usage: hltDiff -o|--old-files FILE1.ROOT [FILE2.ROOT ...] [-O|--old-process LABEL[:INSTANCE[:PROCESS]]]\n\
               -n|--new-files FILE1.ROOT [FILE2.ROOT ...] [-N|--new-process LABEL[:INSTANCE[:PROCESS]]]\n\
               [-m|--max-events MAXEVENTS] [-p|--prescales] [-c|--csv-output] [-j|--json-output]\n\
               [-r|--root-output] [-f|--file-check] [-d|--debug] [-q|--quiet] [-v|--verbose]\n\
               [-h|--help] [-F|--output-file] FILE_NAME\n\
\n\
  -o|--old-files FILE1.ROOT [FILE2.ROOT ...]\n\
      input file(s) with the old (reference) trigger results\n\
\n\
  -O|--old-process PROCESS\n\
      process name of the collection with the old (reference) trigger results\n\
      default: take the 'TriggerResults' from the last process\n\
\n\
  -n|--new-files FILE1.ROOT [FILE2.ROOT ...]\n\
      input file(s) with the new trigger results to be compared with the reference\n\
      to read these from a different collection in the same files as\n\
      the reference, use '-n -' and specify the collection with -N (see below)\n\
\n\
  -N|--new-process PROCESS\n\
      process name of the collection with the new (reference) trigger results\n\
      default: take the 'TriggerResults' from the last process\n\
\n\
  -m|--max-events MAXEVENTS\n\
      compare only the first MAXEVENTS events\n\
      default: compare all the events in the original (reference) files\n\
\n\
  -p|--prescales\n\
      do not ignore differences caused by HLTPrescaler modules\n\
\n\
  -c|--csv-output\n\
      produce comparison results in a CSV format\n\
\n\
  -j|--json-output\n\
      produce comparison results in a JSON format\n\
\n\
  -r|--root-output\n\
      produce comparison results as histograms in a ROOT file\n\
\n\
  -F|--output-file FILE_NAME\n\
      combine all RUNs to files with the specified custom name: FILE_NAME.json, FILE_NAME.root\n\
      default: a separate output file will be produced for each RUN with names suitable for the DQM GUI\n\
\n\
  -f|--file-check\n\
      check existence of every old and new file before running the comparison\n\
      safer if files are run for the first time, but can cause a substantial delay\n\
\n\
  -d|--debug\n\
      display messages about missing events and collectiions\n\
\n\
  -q|--quiet\n\
      don't display summary printout with the list of affected trigger paths\n\
\n\
  -v|--verbose LEVEL\n\
      set verbosity level:\n\
      1: event-by-event comparison results\n\
      2: + print the trigger candidates of the affected filters\n\
      3: + print all the trigger candidates for the affected events\n\
      default: 1\n\
\n\
  -h|--help\n\
      print this help message, and exit"
        << std::endl;
  }
};

int main(int argc, char** argv) {
  // options
  const char optstring[] = "dfo:O:n:N:m:pcjrF:v::hq";
  const option longopts[] = {
      option{"debug", no_argument, nullptr, 'd'},
      option{"file-check", no_argument, nullptr, 'f'},
      option{"old-files", required_argument, nullptr, 'o'},
      option{"old-process", required_argument, nullptr, 'O'},
      option{"new-files", required_argument, nullptr, 'n'},
      option{"new-process", required_argument, nullptr, 'N'},
      option{"max-events", required_argument, nullptr, 'm'},
      option{"prescales", no_argument, nullptr, 'p'},
      option{"csv-output", optional_argument, nullptr, 'c'},
      option{"json-output", optional_argument, nullptr, 'j'},
      option{"root-output", optional_argument, nullptr, 'r'},
      option{"output-file", optional_argument, nullptr, 'F'},
      option{"verbose", optional_argument, nullptr, 'v'},
      option{"help", no_argument, nullptr, 'h'},
      option{"quiet", no_argument, nullptr, 'q'},
  };

  // Creating an HltDiff object with the default configuration
  auto hlt = new HltDiff();

  // parse the command line options
  int c = -1;
  while ((c = getopt_long(argc, argv, optstring, longopts, nullptr)) != -1) {
    switch (c) {
      case 'd':
        hlt->debug = true;
        break;

      case 'f':
        hlt->file_check = true;
        break;

      case 'o':
        hlt->old_files.emplace_back(optarg);
        while (optind < argc) {
          if (argv[optind][0] == '-')
            break;
          hlt->old_files.emplace_back(argv[optind]);
          ++optind;
        }
        break;

      case 'O':
        hlt->old_process = optarg;
        break;

      case 'n':
        hlt->new_files.emplace_back(optarg);
        while (optind < argc) {
          if (argv[optind][0] == '-')
            break;
          hlt->new_files.emplace_back(argv[optind]);
          ++optind;
        }
        break;

      case 'N':
        hlt->new_process = optarg;
        break;

      case 'm':
        hlt->max_events = atoi(optarg);
        break;

      case 'p':
        hlt->ignore_prescales = false;
        break;

      case 'c':
        hlt->csv_out = true;
        break;

      case 'j':
        hlt->json_out = true;
        break;

      case 'r':
        hlt->root_out = true;
        break;

      case 'F':
        hlt->output_file = optarg;
        break;

      case 'v':
        hlt->verbose = 1;
        if (optarg) {
          hlt->verbose = std::max(1, atoi(optarg));
        } else if (!optarg && nullptr != argv[optind] && '-' != argv[optind][0]) {
          // workaround for a bug in getopt which doesn't allow space before optional arguments
          const char* tmp_optarg = argv[optind++];
          hlt->verbose = std::max(1, atoi(tmp_optarg));
        }
        break;

      case 'h':
        hlt->usage(std::cerr);
        exit(0);
        break;

      case 'q':
        hlt->quiet = true;
        break;

      default:
        error(std::cerr);
        exit(1);
        break;
    }
  }

  if (hlt->old_files.empty()) {
    error(std::cerr, "hltDiff: please specify the 'old' file(s)");
    exit(1);
  }
  if (hlt->new_files.empty()) {
    error(std::cerr, "hltDiff: please specify the 'new' file(s)");
    exit(1);
  }

  hlt->compare();

  return 0;
}
