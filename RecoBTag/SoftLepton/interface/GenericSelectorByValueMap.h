#ifndef GenericSelectorByValueMap_h
#define GenericSelectorByValueMap_h

/** \class GenericSelectorByValueMap
 *
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

template <typename T>
class GenericSelectorByValueMap : public edm::EDProducer {
public:

  explicit GenericSelectorByValueMap(edm::ParameterSet const & config);
  ~GenericSelectorByValueMap();

private:

  void produce(edm::Event & event, edm::EventSetup const & setup);

  edm::InputTag m_electrons;
  edm::InputTag m_selection;

  double m_cut;
};


//------------------------------------------------------------------------------

#include <vector>
#include <memory>

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/PtrVector.h"

//------------------------------------------------------------------------------

template <typename T>
GenericSelectorByValueMap<T>::GenericSelectorByValueMap<T>(edm::ParameterSet const & config) :
  m_electrons(config.getParameter<edm::InputTag>("input")),
  m_selection(config.getParameter<edm::InputTag>("selection")),
  m_cut(config.getParameter<double>("cut"))
{
  // register the product
  produces<edm::PtrVector<T> >();
}

//------------------------------------------------------------------------------

template <typename T>
GenericSelectorByValueMap<T>::~GenericSelectorByValueMap<T>() {
}

//------------------------------------------------------------------------------

template <typename T>
void GenericSelectorByValueMap<T>::produce(edm::Event & event, const edm::EventSetup & setup)
{
  std::auto_ptr<edm::PtrVector<T> > candidates(new edm::PtrVector<T>());

  // read the collection of GsfElectrons from the Event
  edm::Handle<edm::View<T> > h_electrons;
  event.getByLabel(m_electrons, h_electrons);
  edm::View<T> const & electrons = * h_electrons;

  // read the selection map from the Event
  edm::Handle<edm::ValueMap<float> > h_selection;
  event.getByLabel(m_selection, h_selection);
  edm::ValueMap<float> const & selectionMap = * h_selection;
 
  for (unsigned int i = 0; i < electrons.size(); ++i) {
    edm::Ptr<T> ptr = electrons.ptrAt(i);
    if (selectionMap[ptr] > m_cut)
     candidates->push_back(ptr); 
  }

  // put the product in the event
  event.put(candidates);
}

#endif // GenericSelectorByValueMap_h
