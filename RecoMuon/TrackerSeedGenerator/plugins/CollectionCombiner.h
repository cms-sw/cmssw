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
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

template <typename Collection>
class CollectionCombiner : public edm::global::EDProducer<>{
public:
  explicit CollectionCombiner(const edm::ParameterSet&);
  ~CollectionCombiner();
  
private:
  virtual void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  
  // ----------member data ---------------------------
  std::vector<edm::InputTag> labels;
  std::vector<edm::EDGetTokenT<Collection> > collectionTokens; 

};

template <typename Collection>
CollectionCombiner<Collection>::CollectionCombiner(const edm::ParameterSet& iConfig){
  labels = iConfig.getParameter<std::vector<edm::InputTag> >("labels");
  produces<Collection>();
  for (unsigned int i=0;i<labels.size();++i)
    collectionTokens.push_back(consumes<Collection>(labels.at(i)));
}
template <typename Collection>
CollectionCombiner<Collection>::~CollectionCombiner(){}

template <typename Collection>
void CollectionCombiner<Collection>::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& es) const
{
  unsigned int i=0,i_max=labels.size();
  edm::Handle<Collection> handle;
  std::auto_ptr<Collection> merged(new Collection());
  for (;i!=i_max;++i){
    iEvent.getByToken(collectionTokens[i], handle);
    merged->insert(merged->end(), handle->begin(), handle->end());
  }
  iEvent.put(merged);
}


#endif
