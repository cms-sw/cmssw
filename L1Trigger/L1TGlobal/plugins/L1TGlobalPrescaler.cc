#include <vector>
#include <array>
#include <memory>
#include <cassert>
#include <string>
#include <stdexcept>
#include <cstring>

#include <boost/format.hpp>

template <class T, std::size_t N>
std::array<T, N> make_array(std::vector<T> const & values) {
  assert(N == values.size());
  std::array<T, N> ret;
  std::copy(values.begin(), values.end(), ret.begin());
  return ret;
}

template <class T>
bool empty(T const& container) {
  return container.empty();
}

bool empty(const char * container) {
  return container != nullptr;
}


using namespace std::literals;

namespace {

  template <class T>
  struct Entry {
    T            value;
    const char * tag;
    const char * description;
  };

  template <typename S, typename T, unsigned int N>
  std::string build_comment_from_entries(S pre, const Entry<T> (&entries)[N]) {
    std::string comment{ pre };
    size_t length = 0;
    for (auto entry: entries)
      if (entry.tag)
        length = std::max(std::strlen(entry.tag), length);
    for (auto entry: entries)
      if (entry.tag) {
        comment.reserve(comment.size() + length + std::strlen(entry.description) + 8);
        comment += "\n  \"";
        comment += entry.tag;
        comment += "\": ";
        for (unsigned int i = 0; i < length-std::strlen(entry.tag); ++i) comment += ' ';
        comment += entry.description;
      }
    return comment;
  }

  template <typename S1, typename S2, typename T, unsigned int N>
  std::string build_comment_from_entries(S1 pre, const Entry<T> (&entries)[N], S2 post) {
    std::string comment = build_comment_from_entries(pre, entries);
    comment += '\n';
    comment += post;
    return comment;
  }


  template <class T>
#if __cplusplus > 201400
  // extended constexpr support in C++14 and later
  constexpr
#endif
  T get_enum_value(Entry<T> const * entries, const char * tag) {
      for (; entries->tag; ++entries)
        if (std::strcmp(entries->tag, tag) == 0)
          return entries->value;
      throw std::logic_error("invalid tag "s + tag);
  }

  template <class T>
#if __cplusplus > 201400
  // extended constexpr support in C++14 and later
  constexpr
#endif
  T get_enum_value(Entry<T> const * entries, const char * tag, T default_value) {
      for (; entries->tag; ++entries)
        if (std::strcmp(entries->tag, tag) == 0)
          return entries->value;
      return default_value;
  }


}

// ############################################################################


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/one/EDFilter.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "CondFormats/DataRecord/interface/L1TGlobalPrescalesVetosRcd.h"
#include "CondFormats/L1TObjects/interface/L1TGlobalPrescalesVetos.h"
#include "DataFormats/L1TGlobal/interface/GlobalAlgBlk.h"

class L1TGlobalPrescaler : public edm::one::EDFilter<> {
public:
  L1TGlobalPrescaler(edm::ParameterSet const& config);

  virtual bool filter(edm::Event& event, edm::EventSetup const& setup) override;

  static  void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

private:
  enum class Mode {
    ApplyPrescaleValues,    // apply the given prescale values
    ApplyPrescaleRatios,    // apply prescales equal to ratio between the given values and the ones read from the EventSetup
    ApplyColumnValues,      // apply the prescale values from the EventSetup corresponding to the given column index
    ApplyColumnRatios,      // apply prescales equal to ratio between the values corresponsing to the given column index, and the ones read from the EventSetup
    Invalid = -1
  };

  static const
  constexpr Entry<Mode> s_modes[] {
    { Mode::ApplyPrescaleValues, "applyPrescaleValues", "apply the given prescale values" },
    { Mode::ApplyPrescaleRatios, "applyPrescaleRatios", "apply prescales equal to ratio between the given values and the ones read from the EventSetup" },
    { Mode::ApplyColumnValues,   "applyColumnValues",   "apply the prescale values from the EventSetup corresponding to the given column index" },
    { Mode::ApplyColumnRatios,   "applyColumnRatios",   "apply prescales equal to ratio between the values corresponsing to the given column index, and the ones read from the EventSetup" },
    { Mode::Invalid,             nullptr,               nullptr }
  };

  const Mode m_mode;
  const edm::EDGetTokenT<GlobalAlgBlkBxCollection>           m_l1tResultsToken;
  const std::array<double, GlobalAlgBlk::maxPhysicsTriggers> m_l1tPrescales;
  std::array<double, GlobalAlgBlk::maxPhysicsTriggers>       m_prescales;
  std::array<unsigned int, GlobalAlgBlk::maxPhysicsTriggers> m_counters;
  const int m_l1tPrescaleColumn;
  int m_oldIndex;

};

const
constexpr Entry<L1TGlobalPrescaler::Mode> L1TGlobalPrescaler::s_modes[];


L1TGlobalPrescaler::L1TGlobalPrescaler(edm::ParameterSet const& config) :
  m_mode( get_enum_value(s_modes, config.getParameter<std::string>("mode").c_str(), Mode::Invalid) ),
  m_l1tResultsToken( consumes<GlobalAlgBlkBxCollection>(config.getParameter<edm::InputTag>("l1tResults")) ),
  m_l1tPrescales( m_mode == Mode::ApplyPrescaleValues or m_mode == Mode::ApplyPrescaleRatios ?
      make_array<double, GlobalAlgBlk::maxPhysicsTriggers>(config.getParameter<std::vector<double>>("l1tPrescales")) :
      std::array<double, GlobalAlgBlk::maxPhysicsTriggers>() ),
  m_l1tPrescaleColumn( m_mode == Mode::ApplyColumnValues or m_mode == Mode::ApplyColumnRatios ?
      config.getParameter<uint32_t>("l1tPrescaleColumn") : 0 ),
  m_oldIndex(-1)
{
  switch (m_mode) {
    // if the mode is "applyPrescaleValues", use the given values
    case Mode::ApplyPrescaleValues:
      m_prescales = m_l1tPrescales;
      break;

    // otherwise we need to read the prescale values from the EventSetup
    case Mode::ApplyPrescaleRatios:
    case Mode::ApplyColumnValues:
    case Mode::ApplyColumnRatios:
      break;

    // this should never happen
    case Mode::Invalid:
      throw edm::Exception(edm::errors::Configuration) << "invalid mode \"" << config.getParameter<std::string>("mode") << "\"";
  }

  m_counters.fill(0);
  produces<GlobalAlgBlkBxCollection>();
}

bool L1TGlobalPrescaler::filter(edm::Event& event, edm::EventSetup const& setup) {
  edm::Handle<GlobalAlgBlkBxCollection> handle;
  event.getByToken(m_l1tResultsToken, handle);

  // if the input collection does not have any information for bx 0,
  // produce an empty collection, and fail
  if (handle->isEmpty(0)) {
    std::unique_ptr<GlobalAlgBlkBxCollection> result(new GlobalAlgBlkBxCollection());
    event.put(std::move(result));
    return false;
  }

  // read the prescale index
  int index = handle->at(0,0).getPreScColumn();
  assert(index >= 0);

  // Mode::ApplyPrescaleRatios
  // apply prescales equal to ratio between the given values and the ones read from the EventSetup
  if (m_mode == Mode::ApplyPrescaleRatios and m_oldIndex != index) {
    edm::ESHandle<L1TGlobalPrescalesVetos> h;
    setup.get<L1TGlobalPrescalesVetosRcd>().get(h);
    auto const & prescaleTable = h->prescale_table_;
    if (index >= (int) prescaleTable.size())
      throw edm::Exception(edm::errors::LogicError) << boost::format("The prescale index %d is invalid, it should be smaller than the prescale table size %d.") % index % prescaleTable.size();
    auto const & prescales = prescaleTable[index];
    unsigned long i = 0;
    for (; i < std::min(prescales.size(), (unsigned long) GlobalAlgBlk::maxPhysicsTriggers); ++i)
      if (prescales[i] == 0)
        // if the trigger was disabled, keep it disabled
        m_prescales[i] = 0.;
      else
        // if the target prescale is lower than the original prescale, keep the trigger unprescaled
        m_prescales[i] = m_l1tPrescales[i] < prescales[i] ? 1. : (double) m_l1tPrescales[i] / prescales[i];
    for (; i < (unsigned long) GlobalAlgBlk::maxPhysicsTriggers; ++i)
      // disable the triggers not included in the prescale table
      m_prescales[i] = 0.;
    // reset the prescales
    m_counters.fill(0);
    m_oldIndex = index;
  }

  // Mode::ApplyColumnValues
  // apply the prescale values from the EventSetup corresponding to the given column index
  if (m_mode == Mode::ApplyColumnValues and m_oldIndex != m_l1tPrescaleColumn) {
    edm::ESHandle<L1TGlobalPrescalesVetos> h;
    setup.get<L1TGlobalPrescalesVetosRcd>().get(h);
    auto const & prescaleTable = h->prescale_table_;
    if (m_l1tPrescaleColumn >= (int) prescaleTable.size())
      throw edm::Exception(edm::errors::Configuration) << boost::format("The prescale index %d is invalid, it should be smaller than the prescale table size %d.") % m_l1tPrescaleColumn % prescaleTable.size();
    auto const & targets = prescaleTable[m_l1tPrescaleColumn];
    unsigned long i = 0;
    for (; i < std::min(targets.size(), (unsigned long) GlobalAlgBlk::maxPhysicsTriggers); ++i)
      // read the prescales from the EventSetup
      m_prescales[i] = targets[i];
    for (; i < (unsigned long) GlobalAlgBlk::maxPhysicsTriggers; ++i)
      // disable the triggers not included in the prescale table
      m_prescales[i] = 0.;
    // reset the prescales
    m_counters.fill(0);
    m_oldIndex = m_l1tPrescaleColumn;
  }

  // Mode::ApplyColumnRatios
  // apply prescales equal to ratio between the values corresponsing to the given column index, and the ones read from the EventSetup
  if (m_mode == Mode::ApplyColumnRatios and m_oldIndex != index) {
    edm::ESHandle<L1TGlobalPrescalesVetos> h;
    setup.get<L1TGlobalPrescalesVetosRcd>().get(h);
    auto const & prescaleTable = h->prescale_table_;
    if (index >= (int) prescaleTable.size())
      throw edm::Exception(edm::errors::LogicError) << boost::format("The prescale index %d is invalid, it should be smaller than the prescale table size %d.") % index % prescaleTable.size();
    if (m_l1tPrescaleColumn >= (int) prescaleTable.size())
      throw edm::Exception(edm::errors::Configuration) << boost::format("The prescale index %d is invalid, it should be smaller than the prescale table size %d.") % m_l1tPrescaleColumn % prescaleTable.size();
    auto const & prescales = prescaleTable[index];
    auto const & targets = prescaleTable[m_l1tPrescaleColumn];
    unsigned long i = 0;
    for (; i < std::min({prescales.size(), targets.size(), (unsigned long) GlobalAlgBlk::maxPhysicsTriggers}); ++i)
      if (prescales[i] == 0)
        // if the trigger was disabled, keep it disabled
        m_prescales[i] = 0.;
      else
        // if the target prescale is lower than the original prescale, keep the trigger unprescaled
        m_prescales[i] = targets[i] < prescales[i] ? 1. : (double) targets[i] / prescales[i];
    for (; i < (unsigned long) GlobalAlgBlk::maxPhysicsTriggers; ++i)
      // disable the triggers not included in the prescale table
      m_prescales[i] = 0.;
    // reset the prescales
    m_counters.fill(0);
    m_oldIndex = index;
  }

  // make a copy of the GlobalAlgBlk for bx 0
  GlobalAlgBlk algoBlock = handle->at(0,0);

  bool finalOr = false;

  for (unsigned int i = 0; i < GlobalAlgBlk::maxPhysicsTriggers; ++i) {
    if (m_prescales[i] == 0) {
      // mask this trigger: reset the bit
      algoBlock.setAlgoDecisionFinal(i, false);
    } else if (algoBlock.getAlgoDecisionFinal(i)) {
      // prescale this trigger
      ++m_counters[i];
      if (std::fmod(m_counters[i], m_prescales[i]) < 1)
        // the prescale is successful, keep the bit set
        finalOr = true;
      else
        // the prescale failed, reset the bit
        algoBlock.setAlgoDecisionFinal(i, false);
    }
  }

  // set the final OR
  algoBlock.setFinalORPreVeto(finalOr);
  if (algoBlock.getFinalORVeto())
    finalOr = false;
  algoBlock.setFinalOR(finalOr);

  // set the new prescale column
  if (m_mode == Mode::ApplyColumnValues or m_mode == Mode::ApplyColumnRatios)
    algoBlock.setPreScColumn(m_l1tPrescaleColumn);

  // create a new GlobalAlgBlkBxCollection, and set the new prescaled decisions for bx 0
  std::unique_ptr<GlobalAlgBlkBxCollection> result(new GlobalAlgBlkBxCollection());
  result->push_back(0, algoBlock);
  event.put(std::move(result));

  return finalOr;
}


void L1TGlobalPrescaler::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
  // collection with the original uGT results
  edm::ParameterDescription<edm::InputTag> l1tResults("l1tResults", edm::InputTag("gtStage2Digis"), true);
  l1tResults.setComment("Collection with the original uGT results");

  // define how to apply the prescale values
  edm::ParameterDescription<std::string> mode("mode", "applyPrescaleValues", true);
  mode.setComment(build_comment_from_entries("Define how to apply the prescale values:", s_modes));

  // target prescale values (for modes "applyPrescaleValues" or "applyPrescaleRatios")
  edm::ParameterDescription<std::vector<double>> l1tPrescales("l1tPrescales", std::vector<double>(GlobalAlgBlk::maxPhysicsTriggers, 1.), true);
  l1tPrescales.setComment("Target prescale values (for modes \"applyPrescaleValues\" or \"applyPrescaleRatios\")");

  // target prescale column (for modes "applyColumnValues" or "applyColumnRatios")
  edm::ParameterDescription<uint32_t> l1tPrescaleColumn("l1tPrescaleColumn", 0, true);
  l1tPrescaleColumn.setComment("Target prescale column (for modes \"applyColumnValues\" or \"applyColumnRatios\")");

  // validaton of all possible configurations and applyPrescaleValues example
  {
    edm::ParameterSetDescription desc;
    desc.addNode(l1tResults);
    desc.ifValue(mode,
      // if mode is "applyPrescaleValues" or "applyPrescaleRatios", read the target prescales
      "applyPrescaleValues" >> l1tPrescales or
      "applyPrescaleRatios" >> l1tPrescales or
      // if mode is "applyColumnValues" or "applyColumnRatios", read the target column
      "applyColumnValues"   >> l1tPrescaleColumn or
      "applyColumnRatios"   >> l1tPrescaleColumn );
    descriptions.add("l1tGlobalPrescaler", desc);
  }

  // applyColumnRatios example
  {
    edm::ParameterSetDescription desc;
    desc.addNode(l1tResults);
    desc.add<std::string>("mode", "applyColumnRatios")->setComment("apply prescales equal to ratio between the values corresponsing to the given column index, and the ones read from the EventSetup");
    desc.addNode(l1tPrescaleColumn);
    descriptions.add("l1tGlobalPrescalerTargetColumn", desc);
  }
}


// register as a framework plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(L1TGlobalPrescaler);
