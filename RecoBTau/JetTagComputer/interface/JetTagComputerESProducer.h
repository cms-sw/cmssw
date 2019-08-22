#ifndef RecoBTau_JetTagComputerESProducer_h
#define RecoBTau_JetTagComputerESProducer_h

#include <string>
#include <memory>
#include <type_traits>

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputerRecord.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputer.h"

/**
 * The idea here is to provide two implementations for
 * JetTagConmputerESProducer: one for those ConcreteJetTagComputers
 * that consume ES products and thus need the ESGetTokens, and one for
 * those that do not. All ConcreteJetTagComputers are required to have
 * a nested type called 'Tokens'.
 *
 * Those that need the ESGetTokens should define the nested type as a
 * struct/class containing the ESGetTokens and a constructor taking
 * edm::ParameterSet and edm::ESConsumesCollector as arguments. In
 * this case the constructor of ConcreteJetTagComputer takes this
 * 'Tokens' object as an additional argument.
 *
 * Those that do not need ESGetTokens should define the nested type as
 * void, and in this case no further modifications are needed.
 */
namespace jet_tag_computer_esproducer_impl {
  template <typename ConcreteJetTagComputer, bool>
  class JetTagComputerESProducer : public edm::ESProducer {
  private:
    // check that the template parameter inherits from JetTagComputer
    static_assert(std::is_convertible_v<ConcreteJetTagComputer*, JetTagComputer*>);

  public:
    using Tokens = typename ConcreteJetTagComputer::Tokens;

    JetTagComputerESProducer(const edm::ParameterSet& pset)
        : m_tokens(pset, setWhatProduced(this, pset.getParameter<std::string>("@module_label"))), m_pset(pset) {}

    std::unique_ptr<JetTagComputer> produce(const JetTagComputerRecord& record) {
      std::unique_ptr<JetTagComputer> jetTagComputer = std::make_unique<ConcreteJetTagComputer>(m_pset, m_tokens);
      jetTagComputer->initialize(record);
      jetTagComputer->setupDone();
      return jetTagComputer;
    }

  private:
    const Tokens m_tokens;
    const edm::ParameterSet m_pset;
  };

  template <typename ConcreteJetTagComputer>
  class JetTagComputerESProducer<ConcreteJetTagComputer, true> : public edm::ESProducer {
  private:
    // check that the template parameter inherits from JetTagComputer
    static_assert(std::is_convertible_v<ConcreteJetTagComputer*, JetTagComputer*>);

  public:
    JetTagComputerESProducer(const edm::ParameterSet& pset) : m_pset(pset) {
      setWhatProduced(this, pset.getParameter<std::string>("@module_label"));
    }

    std::unique_ptr<JetTagComputer> produce(const JetTagComputerRecord& record) {
      std::unique_ptr<JetTagComputer> jetTagComputer = std::make_unique<ConcreteJetTagComputer>(m_pset);
      jetTagComputer->initialize(record);
      jetTagComputer->setupDone();
      return jetTagComputer;
    }

  private:
    const edm::ParameterSet m_pset;
  };
}  // namespace jet_tag_computer_esproducer_impl

template <typename T>
using JetTagComputerESProducer =
    jet_tag_computer_esproducer_impl::JetTagComputerESProducer<T, std::is_same_v<typename T::Tokens, void>>;

#endif  // RecoBTau_JetTagComputerESProducer_h
