#ifndef CommonTools_UtilAlgos_AssociationMapOneToOne2Association_h
#define CommonTools_UtilAlgos_AssociationMapOneToOne2Association_h
/* \class AssociationMapOneToOne2Association
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: AssociationMapOneToOne2Association.h,v 1.2 2010/02/20 20:55:14 wmtan Exp $
 */

#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/OneToOne.h"
#include "DataFormats/Common/interface/Association.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"

template<typename CKey, typename CVal>
class AssociationMapOneToOne2Association : public edm::EDProducer {
 public:
  AssociationMapOneToOne2Association(const edm::ParameterSet&);
 private:
  typedef edm::AssociationMap<edm::OneToOne<CKey, CVal> > am_t;
  typedef edm::Association<CVal> as_t;
  void produce(edm::Event&, const edm::EventSetup&) override;
  edm::EDGetTokenT<am_t> am_;
};

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"
#include "DataFormats/Common/interface/CloneTrait.h"

template<typename CKey, typename CVal>
AssociationMapOneToOne2Association<CKey, CVal>::AssociationMapOneToOne2Association(const edm::ParameterSet& cfg) :
  am_(consumes<am_t>(cfg.template getParameter<edm::InputTag>("src"))) {
  produces<as_t>();
}

template<typename CKey, typename CVal>
void AssociationMapOneToOne2Association<CKey, CVal>::produce(edm::Event& evt, const edm::EventSetup&) {
  using namespace edm;
  using namespace std;
  Handle<am_t> am;
  evt.getByToken(am_, am);

  auto_ptr<as_t> as(new as_t);
  typename as_t::Filler filler(*as);
  filler.fill();
  size_t size = am->size();
  vector<int> indices;
  indices.reserve(size);
  for(typename am_t::const_iterator i = am->begin(); i != am->end(); ++i) {
    indices.push_back(i->val.key());
  }
  filler.insert(am->refProd().key, indices.begin(), indices.end());
  evt.put(as);
}

#endif
