#ifndef HLTrigger_CountNumberOfObject_H
/**\class CountNumberOfObject
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

template <class OColl>
class CountNumberOfObject : public edm::EDFilter {
public:
  explicit CountNumberOfObject(const edm::ParameterSet& iConfig) :
    moduleName_(iConfig.getParameter<std::string>("@module_label")),
    src_(iConfig.getParameter<edm::InputTag>("src")),
    min_(iConfig.getParameter<int>("min")),
    max_(iConfig.getParameter<int>("max"))
      {};
  
  ~CountNumberOfObject(){};
  
private:
  virtual void beginJob(const edm::EventSetup&){};
  virtual bool filter(edm::Event& iEvent, const edm::EventSetup&)
  {
    edm::Handle<OColl> oHandle;
    iEvent.getByLabel(src_, oHandle);
    int s=oHandle->size();
    bool answer=(s>=min_ && s<=max_);
    LogDebug("CountNumberOfObject")<<moduleName_<<" sees: "<<s<<" objects. Filtere answer is: "<<(answer?"true":"false")<<std::endl;
    return answer;
  }
  virtual void endJob(){};
 
  std::string moduleName_;
  edm::InputTag src_;
  int min_,max_;
};


#endif
