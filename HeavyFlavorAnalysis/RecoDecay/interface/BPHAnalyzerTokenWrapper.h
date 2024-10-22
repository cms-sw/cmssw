#ifndef HeavyFlavorAnalysis_RecoDecay_BPHAnalyzerTokenWrapper_h
#define HeavyFlavorAnalysis_RecoDecay_BPHAnalyzerTokenWrapper_h
/** \classes BPHModuleWrapper, BPHTokenWrapper, BPHESTokenWrapper,
 *           BPHEventSetupWrapper and BPHAnalyzerWrapper
 *
 *  Description: 
 *    Common interfaces to define modules and get objects
 *    from "old" and "new" CMSSW version in an uniform way
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoCandidate.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EDConsumerBase.h"
#include "FWCore/Framework/interface/ESConsumesCollector.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/stream/EDAnalyzer.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

//---------------
// C++ Headers --
//---------------
#include <string>
#include <map>
#include <memory>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class BPHModuleWrapper {
public:
  typedef edm::one::EDAnalyzer<> one_analyzer;
  typedef edm::one::EDProducer<> one_producer;
  typedef edm::stream::EDAnalyzer<> stream_analyzer;
  typedef edm::stream::EDProducer<> stream_producer;
};

template <class Obj>
class BPHTokenWrapper {
public:
  typedef edm::EDGetTokenT<Obj> type;
  bool get(const edm::Event& ev, edm::Handle<Obj>& obj) { return ev.getByToken(token, obj); }
  type token;
};

template <class Obj, class Rec>
class BPHESTokenWrapper {
public:
  typedef edm::ESGetToken<Obj, Rec> type;
  bool get(const edm::EventSetup& es, edm::ESHandle<Obj>& obj) {
    obj = es.get<Rec>().getHandle(token);
    return obj.isValid();
  }
  type token;
};

template <class T>
class BPHAnalyzerWrapper : public T {
protected:
  template <class Obj>
  void consume(BPHTokenWrapper<Obj>& tw, const std::string& label) {
    edm::InputTag tag(label);
    tw.token = this->template consumes<Obj>(tag);
    return;
  }
  template <class Obj>
  void consume(BPHTokenWrapper<Obj>& tw, const edm::InputTag& tag) {
    tw.token = this->template consumes<Obj>(tag);
    return;
  }
  template <class Obj, class Rec>
  void esConsume(BPHESTokenWrapper<Obj, Rec>& tw) {
    tw.token = this->template esConsumes<Obj, Rec>();
    return;
  }
  template <class Obj, class Rec>
  void esConsume(BPHESTokenWrapper<Obj, Rec>& tw, const std::string& label) {
    tw.token = this->template esConsumes<Obj, Rec>(edm::ESInputTag("", label));
    return;
  }
  template <class Obj, class Rec>
  void esConsume(BPHESTokenWrapper<Obj, Rec>& tw, const edm::ESInputTag& tag) {
    tw.token = this->template esConsumes<Obj>(tag);
    return;
  }
};

class BPHEventSetupWrapper {
public:
  explicit BPHEventSetupWrapper(const edm::EventSetup& es)
      : ep(&es), twMap(new std::map<BPHRecoCandidate::esType, void*>) {}
  BPHEventSetupWrapper(const edm::EventSetup& es, BPHRecoCandidate::esType type, void* token)
      : BPHEventSetupWrapper(es) {
    (*twMap)[type] = token;
  }
  BPHEventSetupWrapper(const edm::EventSetup& es, std::map<BPHRecoCandidate::esType, void*> tokenMap)
      : BPHEventSetupWrapper(es) {
    twMap->insert(tokenMap.begin(), tokenMap.end());
  }
  BPHEventSetupWrapper(const BPHEventSetupWrapper& es) = default;
  BPHEventSetupWrapper(const BPHEventSetupWrapper* es) : BPHEventSetupWrapper(*es) {}
  BPHEventSetupWrapper(const BPHEventSetupWrapper& es, BPHRecoCandidate::esType type, void* token)
      : BPHEventSetupWrapper(es) {
    (*twMap)[type] = token;
  }
  BPHEventSetupWrapper(BPHEventSetupWrapper& es, std::map<BPHRecoCandidate::esType, void*> tokenMap)
      : BPHEventSetupWrapper(es) {
    twMap->insert(tokenMap.begin(), tokenMap.end());
  }
  const edm::EventSetup* get() const { return ep; }
  operator const edm::EventSetup&() const { return *ep; }
  template <class Obj, class Rec>
  BPHESTokenWrapper<Obj, Rec>* get(BPHRecoCandidate::esType type) const {
    const auto& iter = twMap->find(type);
    return (iter == twMap->end() ? nullptr : static_cast<BPHESTokenWrapper<Obj, Rec>*>(iter->second));
  }

private:
  const edm::EventSetup* ep;
  std::shared_ptr<std::map<BPHRecoCandidate::esType, void*>> twMap;
};

#endif
