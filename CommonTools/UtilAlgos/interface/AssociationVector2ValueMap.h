#ifndef CommonTools_UtilAlgos_AssociationVector2ValueMap_h
#define CommonTools_UtilAlgos_AssociationVector2ValueMap_h
/* \class AssociationVector2ValueMap
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: AssociationVector2ValueMap.h,v 1.2 2010/02/20 20:55:15 wmtan Exp $
 */

#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"

template<typename KeyRefProd, typename CVal>
class AssociationVector2ValueMap : public edm::EDProducer {
 public:
  AssociationVector2ValueMap(const edm::ParameterSet&);
 private:
  typedef edm::AssociationVector<KeyRefProd, CVal> av_t;
  typedef typename CVal::value_type value_t;
  typedef edm::ValueMap<value_t> vm_t;
  typedef typename av_t::CKey collection_t;
  void produce(edm::Event&, const edm::EventSetup&) override;
  edm::EDGetTokenT<av_t> av_;
};

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"
#include "DataFormats/Common/interface/CloneTrait.h"

template<typename KeyRefProd, typename CVal>
AssociationVector2ValueMap<KeyRefProd, CVal>::AssociationVector2ValueMap(const edm::ParameterSet& cfg) :
  av_(consumes<av_t>(cfg.template getParameter<edm::InputTag>("src"))) {
  produces<vm_t>();
}

template<typename KeyRefProd, typename CVal>
void AssociationVector2ValueMap<KeyRefProd, CVal>::produce(edm::Event& evt, const edm::EventSetup&) {
  using namespace edm;
  using namespace std;
  Handle<av_t> av;
  evt.getByToken(av_, av);

  auto_ptr<vm_t> vm(new vm_t);
  typename vm_t::Filler filler(*vm);
  filler.fill();
  size_t size = av->size();
  vector<value_t> values;
  values.reserve(size);
  for(typename av_t::const_iterator i = av->begin(); i != av->end(); ++i) {
    values.push_back(i->second);
  }
  filler.insert(av->keyProduct(), values.begin(), values.end());
  evt.put(vm);
}

#endif
