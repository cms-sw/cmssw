/** \class HLTMhtFilter
*
*
*  \author Gheorghe Lungu
*
*/

#include "HLTrigger/JetMET/interface/HLTMhtFilter.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/METReco/interface/METFwd.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/METCollection.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include <vector>


//
// constructors and destructor
//
HLTMhtFilter::HLTMhtFilter(const edm::ParameterSet& iConfig)
{
  inputMhtTag_ = iConfig.getParameter< edm::InputTag > ("inputMhtTag");
  saveTags_     = iConfig.getParameter<bool>("saveTags");
  minMht_= iConfig.getParameter<double> ("minMht");
  
  //register your products
  produces<trigger::TriggerFilterObjectWithRefs>();
}

HLTMhtFilter::~HLTMhtFilter(){}

void HLTMhtFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("inputMhtTag",edm::InputTag("hltMht30"));
  desc.add<bool>("saveTags",false);
  desc.add<double>("minMht",0.0);
  descriptions.add("hltMhtFilter",desc);
}

// ------------ method called to produce the data  ------------
bool
  HLTMhtFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace std;
  using namespace edm;
  using namespace reco;
  using namespace trigger;
  // The filter object
  auto_ptr<trigger::TriggerFilterObjectWithRefs> filterobject (new trigger::TriggerFilterObjectWithRefs(path(),module()));
  if (saveTags_) filterobject->addCollectionTag(inputMhtTag_);

  METRef ref;
  
  Handle<METCollection> recomhts;
  iEvent.getByLabel(inputMhtTag_,recomhts);

  // look at all candidates,  check cuts and add to filter object
  int n(0), flag(0);
  double mht(0);
    
  
  for (METCollection::const_iterator recomht = recomhts->begin(); recomht != recomhts->end(); recomht++) {
    mht = recomht->pt();
  }
  
  if( mht > minMht_) flag=1;
  
  if (flag==1) {
    for (METCollection::const_iterator recomht = recomhts->begin(); recomht != recomhts->end(); recomht++) {
      ref = METRef(recomhts,distance(recomhts->begin(),recomht));
      filterobject->addObject(TriggerMET,ref);
      n++;
    } 
  }
  
  // filter decision
  bool accept(n>0);
  
  // put filter object into the Event
  iEvent.put(filterobject);
  
  return accept;
}
