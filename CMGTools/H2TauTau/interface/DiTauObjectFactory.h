#ifndef DITAUOBJECTFACTORY_H_
#define DITAUOBJECTFACTORY_H_

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/PatCandidates/interface/CompositeCandidate.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/METReco/interface/MET.h"

#include <algorithm>
#include <set>

namespace cmg {

typedef pat::CompositeCandidate DiTauObject;

template<typename T, typename U>
class DiTauObjectFactory : public edm::EDProducer 
{
    public:
        DiTauObjectFactory(const edm::ParameterSet& ps) :             
            leg1Label_(ps.getParameter<edm::InputTag>("leg1Collection")),
            leg2Label_(ps.getParameter<edm::InputTag>("leg2Collection")),
            metLabel_(ps.getParameter<edm::InputTag>("metCollection"))
        {
          produces<std::vector<DiTauObject>>();
        }

        void produce(edm::Event&, const edm::EventSetup&);
        static void set(const std::pair<T, U>& pair, const reco::MET& met, cmg::DiTauObject& obj);
        static void set(const reco::MET& met, cmg::DiTauObject& obj);

    private:
        const edm::InputTag leg1Label_;
        const edm::InputTag leg2Label_;
        const edm::InputTag metLabel_;
};

///Make when the types are different
template<typename T, typename U>
cmg::DiTauObject makeDiTau(const T& l1, const U& l2){
    cmg::DiTauObject diTauObj = cmg::DiTauObject();
    diTauObj.addDaughter(l1);
    diTauObj.addDaughter(l2);
    diTauObj.setP4(l1.p4() + l2.p4());
    diTauObj.setCharge(l1.charge() + l2.charge());
    return diTauObj;
}

///Make when the types are the same - sorts the legs. It's still ensured below
///that it's only run if the tau collections are identical.
template<typename T>
cmg::DiTauObject makeDiTau(const T& l1, const T& l2){
    cmg::DiTauObject diTauObj = cmg::DiTauObject();
    if(l1.pt() >= l2.pt()){
        diTauObj.addDaughter(l1);
        diTauObj.addDaughter(l2);
    }else{
        diTauObj.addDaughter(l2);
        diTauObj.addDaughter(l1);
    }
    diTauObj.setP4(l1.p4() + l2.p4());
    diTauObj.setCharge(l1.charge() + l2.charge());
    return diTauObj;
}

template<typename T, typename U>
void cmg::DiTauObjectFactory<T, U>::set(const std::pair<T, U>& pair, const reco::MET& met, cmg::DiTauObject& obj) {

  T first = pair.first;
  U second = pair.second;

  // reset daughters
  obj.clearDaughters();

  obj.addDaughter(first);
  obj.addDaughter(second);
  
  obj.setP4(first.p4() + second.p4());
  obj.setCharge(first.charge() + second.charge());
  obj.addDaughter(met);
}

template<typename T, typename U>
void cmg::DiTauObjectFactory<T, U>::set(const reco::MET& met, cmg::DiTauObject& obj) {
  obj.addDaughter(met);
}

template<typename T, typename U>
void cmg::DiTauObjectFactory<T, U>::produce(edm::Event& iEvent, const edm::EventSetup&){
  
  typedef edm::View<T> collection1;
  typedef edm::View<U> collection2;
  typedef edm::View<reco::MET> met_collection;
  
  edm::Handle<collection1> leg1Cands;
  iEvent.getByLabel(this->leg1Label_, leg1Cands);
  
  edm::Handle<collection2> leg2Cands;
  iEvent.getByLabel(this->leg2Label_, leg2Cands);
  
  edm::Handle<met_collection> metCands;
  bool metAvailable = false;
  if (!(metLabel_ == edm::InputTag())) {
    metAvailable = true; 
    iEvent.getByLabel(this->metLabel_, metCands);
  }

  std::auto_ptr<std::vector<DiTauObject>> result(new std::vector<DiTauObject>);

  bool patMet = false;

  const bool sameCollection = (leg1Cands.id () == leg2Cands.id());
  for (auto& metCand : *metCands) {
    const pat::MET* patMET = dynamic_cast<const pat::MET*>(&metCand);
    if (patMET) {
      patMet = true;
      if (! patMET->hasUserCand("lepton1") || ! patMET->hasUserCand("lepton2"))
        edm::LogWarning("produce") << "Cannot access MET user candidates" << std::endl;
      const T* first = dynamic_cast<const T*>(patMET->userCand("lepton1").get());
      const U* second = dynamic_cast<const U*>(patMET->userCand("lepton2").get());
      if (!first || !second)
        edm::LogWarning("produce") << "MET user candidates have incompatible type" << std::endl;
      cmg::DiTauObject cmgTmp = sameCollection ? cmg::makeDiTau<T>(*first, *second) : cmg::makeDiTau<T, U>(*first, *second); 
      cmg::DiTauObjectFactory<T, U>::set(*patMET, cmgTmp);
      result->push_back(cmgTmp);
    }
  }


  if (!patMet) {
    for (size_t i1 = 0; i1 < leg1Cands->size(); ++i1) {
      for (size_t i2 = 0; i2 < leg2Cands->size(); ++i2) {

        // if the same collection, only produce each possible pair once
        if (sameCollection && (i1 >= i2)) 
          continue;
        
        //enable sorting only if we are using the same collection - see Savannah #20217
        cmg::DiTauObject cmgTmp = sameCollection ? cmg::makeDiTau<T>((*leg1Cands)[i1], (*leg2Cands)[i2]) : cmg::makeDiTau<T, U>((*leg1Cands)[i1], (*leg2Cands)[i2]); 
        
        if (metAvailable && ! metCands->empty()) {
            if (metCands->size() < result->size()+1)
              edm::LogWarning("produce") << "Fewer MET candidates than leg1/leg2 combinations; are the inputs to the MET producer and the di-tau object producer the same?" << std::endl;
            cmg::DiTauObjectFactory<T, U>::set(metCands->at(result->size()), cmgTmp);
            result->push_back(cmgTmp);
        }
      }
    }
  }

  iEvent.put(result); 
}

} // namespace cmg


#endif /*DITAUOBJECTFACTORY_H_*/
