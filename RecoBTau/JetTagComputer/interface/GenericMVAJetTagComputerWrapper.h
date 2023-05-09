#ifndef RecoBTau_JetTagComputer_GenericMVAJetTagComputerWrapper_h
#define RecoBTau_JetTagComputer_GenericMVAJetTagComputerWrapper_h

#include <memory>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/BTauReco/interface/BaseTagInfo.h"
#include "DataFormats/BTauReco/interface/TaggingVariable.h"
#include "RecoBTau/JetTagComputer/interface/GenericMVAJetTagComputer.h"

/*
 * This header defines a bunch of templated convenience wrappers
 *
 *      GenericMVAJetTagComputerWrapper<class Provider, ...>
 *
 * whereas ... dennotes argument pairs, with the
 *   first argument being a specific TagInfo class
 *   second argument being a constant string dennoting the cfg parameter
 *
 * If only one TagInfo is passed, the cfg parameter name can be ommited
 * (will default to "tagInfo"). 
 *
 * The wrapper will derive from the template argument class Provider,
 * pass the ParameterSet of the JetTagComputer to the (optional) constructor
 * and call "TaggingVariableList operator () (...) const" for every jet to
 * tag, with ... being the already casted and type-verified specific
 * TagInfo references (number of arguments matching the template instance).
 *
 * The TaggingVariables will be fed into the MVAComputer, which will compute
 * the final discriminator from an ES calibration object define in the cfg.
 */

namespace btau_dummy {
  struct Null {};
  constexpr const char none[] = "";
}  // namespace btau_dummy

// 4 named TagInfos

template <class Provider,
          class TI1,
          const char *ti1 = btau_dummy::none,
          class TI2 = btau_dummy::Null,
          const char *ti2 = btau_dummy::none,
          class TI3 = btau_dummy::Null,
          const char *ti3 = btau_dummy::none,
          class TI4 = btau_dummy::Null,
          const char *ti4 = btau_dummy::none>
class GenericMVAJetTagComputerWrapper : public GenericMVAJetTagComputer, private Provider {
public:
  GenericMVAJetTagComputerWrapper(const edm::ParameterSet &params, Tokens tokens)
      : GenericMVAJetTagComputer(params, tokens), Provider(params) {
    uses(0, ti1);
    uses(1, ti2);
    uses(2, ti3);
    uses(3, ti4);
  }

protected:
  reco::TaggingVariableList taggingVariables(const TagInfoHelper &info) const override {
    return (static_cast<const Provider &>(*this))(
        info.get<TI1>(0), info.get<TI2>(1), info.get<TI3>(2), info.get<TI4>(3));
  }
};

// 3 named TagInfos

template <class Provider, class TI1, const char *ti1, class TI2, const char *ti2, class TI3, const char *ti3>
class GenericMVAJetTagComputerWrapper<Provider, TI1, ti1, TI2, ti2, TI3, ti3, btau_dummy::Null, btau_dummy::none>
    : public GenericMVAJetTagComputer, private Provider {
public:
  GenericMVAJetTagComputerWrapper(const edm::ParameterSet &params, Tokens tokens)
      : GenericMVAJetTagComputer(params, tokens), Provider(params) {
    uses(0, ti1);
    uses(1, ti2);
    uses(2, ti3);
  }

protected:
  reco::TaggingVariableList taggingVariables(const TagInfoHelper &info) const override {
    return (static_cast<const Provider &>(*this))(info.get<TI1>(0), info.get<TI2>(1), info.get<TI3>(2));
  }
};

// 2 named TagInfos

template <class Provider, class TI1, const char *ti1, class TI2, const char *ti2>
class GenericMVAJetTagComputerWrapper<Provider,
                                      TI1,
                                      ti1,
                                      TI2,
                                      ti2,
                                      btau_dummy::Null,
                                      btau_dummy::none,
                                      btau_dummy::Null,
                                      btau_dummy::none> : public GenericMVAJetTagComputer,
                                                          private Provider {
public:
  GenericMVAJetTagComputerWrapper(const edm::ParameterSet &params, Tokens tokens)
      : GenericMVAJetTagComputer(params, tokens), Provider(params) {
    uses(0, ti1);
    uses(1, ti2);
  }

protected:
  reco::TaggingVariableList taggingVariables(const TagInfoHelper &info) const override {
    return (static_cast<const Provider &>(*this))(info.get<TI1>(0), info.get<TI2>(1));
  }
};

// 1 named TagInfo

template <class Provider, class TI1, const char *ti1>
class GenericMVAJetTagComputerWrapper<Provider,
                                      TI1,
                                      ti1,
                                      btau_dummy::Null,
                                      btau_dummy::none,
                                      btau_dummy::Null,
                                      btau_dummy::none,
                                      btau_dummy::Null,
                                      btau_dummy::none> : public GenericMVAJetTagComputer,
                                                          private Provider {
public:
  GenericMVAJetTagComputerWrapper(const edm::ParameterSet &params, Tokens tokens)
      : GenericMVAJetTagComputer(params, tokens), Provider(params) {
    uses(0, ti1);
  }

protected:
  reco::TaggingVariableList taggingVariables(const TagInfoHelper &info) const override {
    return (static_cast<const Provider &>(*this))(info.get<TI1>(0));
  }
};

// default TagInfo

template <class Provider, class TI1>
class GenericMVAJetTagComputerWrapper<Provider,
                                      TI1,
                                      btau_dummy::none,
                                      btau_dummy::Null,
                                      btau_dummy::none,
                                      btau_dummy::Null,
                                      btau_dummy::none,
                                      btau_dummy::Null,
                                      btau_dummy::none> : public GenericMVAJetTagComputer,
                                                          private Provider {
public:
  GenericMVAJetTagComputerWrapper(const edm::ParameterSet &params, Tokens tokens)
      : GenericMVAJetTagComputer(params, tokens), Provider(params) {}

protected:
  reco::TaggingVariableList taggingVariables(const TagInfoHelper &info) const override {
    return (static_cast<const Provider &>(*this))(info.get<TI1>(0));
  }
};

#endif  // RecoBTau_JetTagComputer_GenericMVAJetTagComputerWrapper_h
