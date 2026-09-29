#ifndef DQMServices_Components_interface_GenericObjectDQMSource_h
#define DQMServices_Components_interface_GenericObjectDQMSource_h

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "DQMServices/Components/interface/DQMVariable.h"
#include "DQMServices/Components/interface/DQMVectorVariable.h"

#include <cstddef>
#include <string>
#include <type_traits>
#include <vector>

// Trait template supplying the list of plottable variables for object type T.
// Must be specialized per type before GenericObjectDQMSource<T> is instantiated,
// e.g. see TrackDQMVariables.h for the reco::Track specialization.
//
// A specialization only needs `variables()` (scalar members). If T also has
// vector-valued members (e.g. Run3ScoutingElectron::trkpt()), it may
// additionally define `vectorVariables()` returning
// std::vector<DQMVectorVariable<T>> -- this is entirely optional and
// detected automatically, so existing scalar-only specializations are
// unaffected.
template <typename T>
struct DQMVariableTraits;

namespace dqmgeneric_detail {
  template <typename T, typename = void>
  struct has_vector_variables : std::false_type {};

  template <typename T>
  struct has_vector_variables<T, std::void_t<decltype(DQMVariableTraits<T>::vectorVariables())>> : std::true_type {};
}  // namespace dqmgeneric_detail

// Generic DQM source: books and fills one 1D histogram per DQMVariable<T>
// (as supplied by DQMVariableTraits<T>::variables()) for every object in an
// input collection of type Collection (default std::vector<T>, matching the
// usual reco::*Collection typedefs). If DQMVariableTraits<T> also defines
// vectorVariables(), one additional histogram per DQMVectorVariable<T> is
// booked and filled once per element on every event.
//
// Adding a plot for a new member of T means adding one line to the trait
// specialization's variable list -- this class itself never changes.
template <typename T, typename Collection = std::vector<T>>
class GenericObjectDQMSource : public DQMEDAnalyzer {
public:
  explicit GenericObjectDQMSource(edm::ParameterSet const& iConfig)
      : token_(consumes<Collection>(iConfig.getParameter<edm::InputTag>("src"))),
        folder_(iConfig.getParameter<std::string>("folder")),
        variables_(DQMVariableTraits<T>::variables()) {
    if constexpr (dqmgeneric_detail::has_vector_variables<T>::value) {
      vectorVariables_ = DQMVariableTraits<T>::vectorVariables();
    }
  }

  ~GenericObjectDQMSource() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("src");
    desc.add<std::string>("folder");
    descriptions.addDefault(desc);
  }

  void bookHistograms(DQMStore::IBooker& ibooker, edm::Run const&, edm::EventSetup const&) override {
    ibooker.setCurrentFolder(folder_);

    histos_.clear();
    histos_.reserve(variables_.size());
    for (auto const& var : variables_) {
      histos_.push_back(
          ibooker.book1D(var.name, var.title + ";" + var.title + ";Entries", var.nbins, var.xmin, var.xmax));
    }

    vectorHistos_.clear();
    vectorHistos_.reserve(vectorVariables_.size());
    for (auto const& var : vectorVariables_) {
      vectorHistos_.push_back(
          ibooker.book1D(var.name, var.title + ";" + var.title + ";Entries", var.nbins, var.xmin, var.xmax));
    }
  }

  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override {
    edm::Handle<Collection> handle;
    iEvent.getByToken(token_, handle);
    if (!handle.isValid())
      return;

    for (auto const& obj : *handle) {
      for (std::size_t i = 0; i < variables_.size(); ++i) {
        histos_[i]->Fill(variables_[i].accessor(obj));
      }
      for (std::size_t i = 0; i < vectorVariables_.size(); ++i) {
        for (double value : vectorVariables_[i].accessor(obj)) {
          vectorHistos_[i]->Fill(value);
        }
      }
    }
  }

private:
  edm::EDGetTokenT<Collection> token_;
  std::string folder_;
  std::vector<DQMVariable<T>> variables_;
  std::vector<MonitorElement*> histos_;
  std::vector<DQMVectorVariable<T>> vectorVariables_;
  std::vector<MonitorElement*> vectorHistos_;
};

#endif
