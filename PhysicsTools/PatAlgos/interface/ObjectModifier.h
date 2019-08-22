#ifndef __PhysicsTools_PatAlgos_ObjectModifier_h__
#define __PhysicsTools_PatAlgos_ObjectModifier_h__

#include "CommonTools/CandAlgos/interface/ModifyObjectValueBase.h"
#include <memory>

namespace pat {
  template <class T>
  class ObjectModifier {
  public:
    typedef std::unique_ptr<ModifyObjectValueBase> ModifierPointer;

    ObjectModifier(const edm::ParameterSet& conf, edm::ConsumesCollector&& cc);
    ~ObjectModifier() {}

    void setEvent(const edm::Event& event) {
      for (unsigned i = 0; i < modifiers_.size(); ++i)
        modifiers_[i]->setEvent(event);
    }

    void setEventContent(const edm::EventSetup& setup) {
      for (unsigned i = 0; i < modifiers_.size(); ++i)
        modifiers_[i]->setEventContent(setup);
    }

    void modify(T& obj) const {
      for (unsigned i = 0; i < modifiers_.size(); ++i)
        modifiers_[i]->modifyObject(obj);
    }

  private:
    std::vector<ModifierPointer> modifiers_;
  };

  template <class T>
  ObjectModifier<T>::ObjectModifier(const edm::ParameterSet& conf, edm::ConsumesCollector&& cc) {
    const std::vector<edm::ParameterSet>& mods = conf.getParameterSetVector("modifications");
    for (unsigned i = 0; i < mods.size(); ++i) {
      const edm::ParameterSet& iconf = mods[i];
      const std::string& mname = iconf.getParameter<std::string>("modifierName");
      modifiers_.emplace_back(ModifyObjectValueFactory::get()->create(mname, iconf, cc));
    }
  }
}  // namespace pat

#endif
