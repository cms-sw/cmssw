#ifndef HLTrigger_HLTCountNumberOfObject_H
/**\class HLTCountNumberOfObject
 * Description:
 * templated EDFilter to count the number of object in a given collection, using View
 * \author Jean-Roch Vlimant
*/

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/MessageService/interface/MessageLogger.h"

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

template <class OColl>
class HLTCountNumberOfObject : public HLTFilter {
public:
  explicit HLTCountNumberOfObject(const edm::ParameterSet& iConfig) :
    src_(iConfig.getParameter<edm::InputTag>("src")),
    minN_(iConfig.getParameter<int>("MinN")),
    maxN_(iConfig.getParameter<int>("MaxN"))
      {
	produces<trigger::TriggerFilterObjectWithRefs>();
      };
  
  ~HLTCountNumberOfObject(){};
  
private:
  virtual bool filter(edm::Event& iEvent, const edm::EventSetup&)
  {
    // The filtered object. which is put empty.
    std::auto_ptr<trigger::TriggerFilterObjectWithRefs> filterproduct (new trigger::TriggerFilterObjectWithRefs(path(),module()));

    edm::Handle<OColl> oHandle;
    iEvent.getByLabel(src_, oHandle);
    int s=oHandle->size();
    bool answer=true;
    if (minN_!=-1) answer = answer && (s>=minN_);
    if (maxN_!=-1) answer = answer && (s<=maxN_);
    LogDebug("HLTCountNumberOfObject")<<module()<<" sees: "<<s<<" objects. Filtere answer is: "<<(answer?"true":"false");

    iEvent.put(filterproduct);
    return answer;
  }
  virtual void endJob(){};
 
  edm::InputTag src_;
  int minN_,maxN_;
};


#endif
