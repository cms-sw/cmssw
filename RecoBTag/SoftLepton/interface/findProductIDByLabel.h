#ifndef edm_findProductIDByLabel_h
#define edm_findProductIDByLabel_h

// Free function to test for the presence of a Handtle<T> in the Event. 
// If it is found, its ProductID is returned. Otherwise, an invalid ProductID is returned, and no exception is thrown.

// This should be dropped in favour of Event::findProductIDByLabel<T>() as it becomes available -
// see https://hypernews.cern.ch/HyperNews/CMS/get/edmFramework/849/1/1/1.html and the associated thread.

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Selector.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Common/interface/Handle.h"

#include "../interface/InputTagSelector.h"

namespace edm {

template <typename T>
edm::ProductID findProductIDByLabel(const edm::Event  & event,
                                    const std::string & label)
{
  std::vector<Handle<T> > results;
  event.getMany<T>( edm::ModuleLabelSelector(label), results );

  if (results.empty())
    return ProductID();
  else
    return results.front().id();
}

template <typename T>
edm::ProductID findProductIDByLabel(const edm::Event  & event,
                                    const std::string & label,
                                    const std::string & productInstanceName)
{
  std::vector<Handle<T> > results;
  event.getMany<T>( edm::Selector(edm::ModuleLabelSelector(label) && edm::ProductInstanceNameSelector(productInstanceName)), results );

  if (results.empty())
    return ProductID();
  else
    return results.front().id();
}

template <typename T>
edm::ProductID findProductIDByLabel(const edm::Event    & event,
                                    const edm::InputTag & tag)
{
  std::vector<Handle<T> > results;
  event.getMany<T>( edm::InputTagSelector(tag), results );

  if (results.empty())
    return ProductID();
  else
    return results.front().id();
}

}

#endif // edm_findProductIDByLabel_h
