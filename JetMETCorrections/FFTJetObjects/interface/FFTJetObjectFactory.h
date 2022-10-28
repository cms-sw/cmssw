#ifndef JetMETCorrections_FFTJetObjects_FFTJetObjectFactory_h
#define JetMETCorrections_FFTJetObjects_FFTJetObjectFactory_h

//
// Simple factory for objects which can be constructed from ParameterSet
//
#include <map>
#include <string>

#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/Utilities/interface/Exception.h"

template <class Base>
struct AbsFFTJetObjectFactory {
  virtual ~AbsFFTJetObjectFactory() {}
  virtual Base* create(const edm::ParameterSet& ps) const = 0;
};

template <class Base, class Derived>
struct ConcreteFFTJetObjectFactory : public AbsFFTJetObjectFactory<Base> {
  ~ConcreteFFTJetObjectFactory() override {}
  inline Derived* create(const edm::ParameterSet& ps) const override { return new Derived(ps); }
};

template <class Base>
struct DefaultFFTJetObjectFactory : public std::map<std::string, AbsFFTJetObjectFactory<Base>*> {
  typedef Base base_type;

  inline DefaultFFTJetObjectFactory() : std::map<std::string, AbsFFTJetObjectFactory<Base>*>() {}

  virtual ~DefaultFFTJetObjectFactory() {
    for (typename std::map<std::string, AbsFFTJetObjectFactory<Base>*>::iterator it = this->begin(); it != this->end();
         ++it)
      delete it->second;
  }

  inline Base* create(const std::string& derivedType, const edm::ParameterSet& ps) const {
    typename std::map<std::string, AbsFFTJetObjectFactory<Base>*>::const_iterator it = this->find(derivedType);
    if (it == this->end())
      throw cms::Exception("KeyNotFound")
          << "Derived type \"" << derivedType << "\" is not registered with the factory\n";
    return it->second->create(ps);
  }

  DefaultFFTJetObjectFactory(const DefaultFFTJetObjectFactory&) = delete;
  DefaultFFTJetObjectFactory& operator=(const DefaultFFTJetObjectFactory&) = delete;
};

//
// Singleton for the factory
//
template <class Factory>
class StaticFFTJetObjectFactory {
public:
  typedef typename Factory::Base::base_type base_type;

  static const Factory& instance() {
    static Factory const obj;
    return obj;
  }

  template <class T>
  static void registerType(const std::string& className) {
    Factory& rd = const_cast<Factory&>(instance());
    delete rd[className];
    rd[className] = new ConcreteFFTJetObjectFactory<base_type, T>();
  }

  StaticFFTJetObjectFactory() = delete;
};

#endif  // JetMETCorrections_FFTJetObjects_FFTJetObjectFactory_h
