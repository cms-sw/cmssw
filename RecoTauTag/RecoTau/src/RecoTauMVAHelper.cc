#include "RecoTauTag/RecoTau/interface/RecoTauMVAHelper.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <boost/foreach.hpp>
#include <boost/bind.hpp>
#include <sstream>

#include "CondFormats/DataRecord/interface/TauTagMVAComputerRcd.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "RecoTauTag/RecoTau/interface/RecoTauDiscriminantPlugins.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/isFinite.h"

namespace reco { namespace tau {

RecoTauMVAHelper::RecoTauMVAHelper(const std::string& name,
                                   const std::string& eslabel,
                                   const edm::ParameterSet& pluginOptions):
    name_(name), eslabel_(eslabel), pluginOptions_(pluginOptions) {}

void RecoTauMVAHelper::setEvent(const edm::Event& evt,
                                const edm::EventSetup &es) {
  // Update our MVA from the DB
  using PhysicsTools::Calibration::MVAComputerContainer;
  edm::ESHandle<MVAComputerContainer> handle;
  if (eslabel_.size()) {
    es.get<TauTagMVAComputerRcd>().get(eslabel_, handle);
  } else {
    es.get<TauTagMVAComputerRcd>().get(handle);
  }
  const MVAComputerContainer *container = handle.product();
  // Load our MVA
  bool reload = computer_.update(container, name_.c_str());
  // If the MVA changed, update our list of discriminant plugins
  if (reload && computer_.get())
    loadDiscriminantPlugins(container->find(name_));
  // Update the event info for all of our discriminators
  BOOST_FOREACH(PluginMap::value_type plugin, plugins_) {
    plugin.second->setup(*const_cast<edm::Event*>(&evt), es);
  }
}

void RecoTauMVAHelper::loadDiscriminantPlugins(
    const PhysicsTools::Calibration::MVAComputer &comp) {
  typedef std::vector<PhysicsTools::Calibration::Variable> VarList;
  // List of input variables for this MVA.
  const VarList &vars = comp.inputSet;
  // Load the plugin for each of the Var if needed
  BOOST_FOREACH(const VarList::value_type& var, vars) {
    // Check to make sure it isn't a magic variable
    if (std::strncmp(var.name.c_str(), "__", 2) != 0) {
      // If we haven't added yet, build it.
      PhysicsTools::AtomicId varId(var.name);
      if (!plugins_.count(varId)) {
        edm::ParameterSet options;
        if (pluginOptions_.exists(var.name)) {
          options = pluginOptions_.getParameter<edm::ParameterSet>(var.name);
        };
        // Make sure it has a name (required by base class)
        if (!options.exists("name"))
          options.addParameter("name", "MVA_" + var.name);
        // Check if we want to specify the plugin name manually.  This is
        // required for things like the discriminant from discriminators, which
        // take an InputTag.  If we want to have more than one, we have to be
        // able take the MVA name (like FlightPathSig) and map it to
        // RecoTauDiscriminantFromDiscriminator[disc input tag = flight path sig]
        // Otherwise we just keep our regular plugin mapping.
        std::string pluginName = reco::tau::discPluginName(var.name);
        if (options.exists("plugin")) {
          pluginName = options.getParameter<std::string>("plugin");
        }
        plugins_.insert(
            varId, RecoTauDiscriminantPluginFactory::get()->create(
                pluginName, options));
      }
    }
  }
}

void RecoTauMVAHelper::fillValues(const reco::PFTauRef& tau) const {
  // Loop over the relevant discriminators and the output
  for (PluginMap::const_iterator plugin = plugins_.begin();
       plugin != plugins_.end(); ++plugin) {
    PhysicsTools::AtomicId id = plugin->first;
    std::vector<double> pluginOutput = (plugin->second)->operator()(tau);
    // Check for nans
    for(size_t instance = 0; instance < pluginOutput.size(); ++instance) {
      if (edm::isNotFinite(pluginOutput[instance])) {
        std::ostringstream error;
        error << "A nan was detected in"
            << " the tau MVA variable " << id << " returning zero instead!"
            << " The PFTau: " << *tau << std::endl;
        tau->dump(error);
        edm::LogError("CorruptedMVAInput") << error.str();
        pluginOutput[instance] = 0.0;
      }
    }
    //std::cout << "id: " << id << " first: " << pluginOutput[0] << std::endl;
    // Build values and copy into values vector
    std::for_each(pluginOutput.begin(), pluginOutput.end(),
                  boost::bind(&PhysicsTools::Variable::ValueList::add,
                              boost::ref(values_), id, _1));
  }
}

// Get values
const PhysicsTools::Variable::ValueList&
RecoTauMVAHelper::discriminants(const PFTauRef& tau) const {
  values_.clear();
  fillValues(tau);
  return values_;
}

// Apply the MVA to a given tau
double RecoTauMVAHelper::operator()(const reco::PFTauRef &tau) const {
  // Clear output
  values_.clear();
  // Build the values
  fillValues(tau);
  // Call the MVA
  return computer_->eval(values_);
}

void RecoTauMVAHelper::train(const reco::PFTauRef &tau, bool target,
                             double weight) const {
  static const PhysicsTools::AtomicId kTargetId("__TARGET__");
  static const PhysicsTools::AtomicId kWeightId("__WEIGHT__");
  if (!computer_)
    return;
  values_.clear();
  values_.add(kTargetId, target);
  values_.add(kWeightId, weight);
  // Build the discriminant values
  fillValues(tau);
  computer_->eval(values_);
}

}}  // end namespace reco::tau
