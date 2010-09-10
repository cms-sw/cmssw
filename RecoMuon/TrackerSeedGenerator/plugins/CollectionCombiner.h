#ifndef CollectionCombiner_H
#define CollectionCombiner_H


/** \class CollectionCombiner 
 * Description: this templated EDProducer can merge (no duplicate removal) any number of collection of the same type.
 * the usage is to declare a concrete combiner in SealModule:
 * typedef CollectionCombiner<std::vector< Trajectory> > TrajectoryCombiner;
 * DEFINE_FWK_MODULE(TrajectoryCombiner);
 * edm::View cannot be used, because the template argument is used for the input and the output type.
 *
 * \author Jean-Roch Vlimant
 */


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

template <typename Collection>
class CollectionCombiner : public edm::EDProducer{
public:
  explicit CollectionCombiner(const edm::ParameterSet&);
  ~CollectionCombiner();
  
private:
  virtual void produce(edm::Event&, const edm::EventSetup&);
  
  // ----------member data ---------------------------
  std::vector<edm::InputTag> labels;
};

template <typename Collection>
CollectionCombiner<Collection>::CollectionCombiner(const edm::ParameterSet& iConfig){
  labels = iConfig.getParameter<std::vector<edm::InputTag> >("labels");
  produces<Collection>();
}
template <typename Collection>
CollectionCombiner<Collection>::~CollectionCombiner(){}

template <typename Collection>
void CollectionCombiner<Collection>::produce(edm::Event& iEvent, const edm::EventSetup& es)
{
  unsigned int i=0,i_max=labels.size();
  edm::Handle<Collection> handle;
  std::auto_ptr<Collection> merged(new Collection());
  for (;i!=i_max;++i){
    iEvent.getByLabel(labels[i], handle);
    merged->insert(merged->end(), handle->begin(), handle->end());
  }
  iEvent.put(merged);
}


#endif
