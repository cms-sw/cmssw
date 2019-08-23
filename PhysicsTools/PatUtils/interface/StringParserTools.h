#ifndef PhysicsTools_PatUtils_interface_StringParserTools_h
#define PhysicsTools_PatUtils_interface_StringParserTools_h

#include "CommonTools/Utils/interface/StringObjectFunction.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/PatCandidates/interface/Photon.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/PatCandidates/interface/GenericParticle.h"
#include "DataFormats/PatCandidates/interface/PFParticle.h"

class PATStringObjectFunction {
public:
  PATStringObjectFunction() {}
  PATStringObjectFunction(const std::string &string);

  double operator()(const reco::Candidate &c) const;

private:
  std::string expr_;
  std::shared_ptr<StringObjectFunction<reco::Candidate> > candFunc_;

  std::shared_ptr<StringObjectFunction<pat::Electron> > eleFunc_;
  std::shared_ptr<StringObjectFunction<pat::Muon> > muFunc_;
  std::shared_ptr<StringObjectFunction<pat::Tau> > tauFunc_;
  std::shared_ptr<StringObjectFunction<pat::Photon> > gamFunc_;
  std::shared_ptr<StringObjectFunction<pat::Jet> > jetFunc_;
  std::shared_ptr<StringObjectFunction<pat::MET> > metFunc_;
  std::shared_ptr<StringObjectFunction<pat::GenericParticle> > gpFunc_;
  std::shared_ptr<StringObjectFunction<pat::PFParticle> > pfFunc_;

  template <typename Obj>
  std::shared_ptr<StringObjectFunction<Obj> > tryGet(const std::string &str) {
    try {
      return std::shared_ptr<StringObjectFunction<Obj> >(new StringObjectFunction<Obj>(str));
    } catch (cms::Exception const &) {
      return std::shared_ptr<StringObjectFunction<Obj> >();
    }
  }

  template <typename Obj>
  double tryEval(const reco::Candidate &c, const std::shared_ptr<StringObjectFunction<Obj> > &func) const {
    if (func.get())
      return (*func)(static_cast<const Obj &>(c));
    else
      throwBadType(typeid(c));
    assert(false);
    return 0;
  }

  // out of line throw exception
  void throwBadType(const std::type_info &ty1) const;
};

class PATStringCutObjectSelector {
public:
  PATStringCutObjectSelector() {}
  PATStringCutObjectSelector(const std::string &string);

  bool operator()(const reco::Candidate &c) const;

private:
  std::string expr_;
  std::shared_ptr<StringCutObjectSelector<reco::Candidate> > candFunc_;

  std::shared_ptr<StringCutObjectSelector<pat::Electron> > eleFunc_;
  std::shared_ptr<StringCutObjectSelector<pat::Muon> > muFunc_;
  std::shared_ptr<StringCutObjectSelector<pat::Tau> > tauFunc_;
  std::shared_ptr<StringCutObjectSelector<pat::Photon> > gamFunc_;
  std::shared_ptr<StringCutObjectSelector<pat::Jet> > jetFunc_;
  std::shared_ptr<StringCutObjectSelector<pat::MET> > metFunc_;
  std::shared_ptr<StringCutObjectSelector<pat::GenericParticle> > gpFunc_;
  std::shared_ptr<StringCutObjectSelector<pat::PFParticle> > pfFunc_;

  template <typename Obj>
  std::shared_ptr<StringCutObjectSelector<Obj> > tryGet(const std::string &str) {
    try {
      return std::shared_ptr<StringCutObjectSelector<Obj> >(new StringCutObjectSelector<Obj>(str));
    } catch (cms::Exception const &) {
      return std::shared_ptr<StringCutObjectSelector<Obj> >();
    }
  }

  template <typename Obj>
  bool tryEval(const reco::Candidate &c, const std::shared_ptr<StringCutObjectSelector<Obj> > &func) const {
    if (func.get())
      return (*func)(static_cast<const Obj &>(c));
    else
      throwBadType(typeid(c));
    assert(false);
    return false;
  }

  // out of line throw exception
  void throwBadType(const std::type_info &ty1) const;
};

#endif
