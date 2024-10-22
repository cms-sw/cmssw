#ifndef CommonTools_UtilAlgos_AssociationVectorSelector_h
#define CommonTools_UtilAlgos_AssociationVectorSelector_h
/* \class AssociationVectorSelector
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: AssociationVectorSelector.h,v 1.2 2010/02/20 20:55:16 wmtan Exp $
 */

#include "DataFormats/Common/interface/AssociationVector.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "CommonTools/UtilAlgos/interface/AnySelector.h"

template <typename KeyRefProd, typename CVal, typename KeySelector = AnySelector, typename ValSelector = AnySelector>
class AssociationVectorSelector : public edm::stream::EDProducer<> {
public:
  AssociationVectorSelector(const edm::ParameterSet&);

private:
  typedef edm::AssociationVector<KeyRefProd, CVal> association_t;
  typedef typename association_t::CKey collection_t;
  void produce(edm::Event&, const edm::EventSetup&) override;
  edm::EDGetTokenT<association_t> associationToken_;
  KeySelector selectKey_;
  ValSelector selectVal_;
};

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"
#include "DataFormats/Common/interface/CloneTrait.h"

template <typename KeyRefProd, typename CVal, typename KeySelector, typename ValSelector>
AssociationVectorSelector<KeyRefProd, CVal, KeySelector, ValSelector>::AssociationVectorSelector(
    const edm::ParameterSet& cfg)
    : associationToken_(consumes<association_t>(cfg.template getParameter<edm::InputTag>("association"))),
      selectKey_(reco::modules::make<KeySelector>(cfg, consumesCollector())),
      selectVal_(reco::modules::make<ValSelector>(cfg, consumesCollector())) {
  std::string alias(cfg.template getParameter<std::string>("@module_label"));
  produces<collection_t>().setBranchAlias(alias);
  produces<association_t>().setBranchAlias(alias + "Association");
}

template <typename KeyRefProd, typename CVal, typename KeySelector, typename ValSelector>
void AssociationVectorSelector<KeyRefProd, CVal, KeySelector, ValSelector>::produce(edm::Event& evt,
                                                                                    const edm::EventSetup&) {
  using namespace edm;
  using namespace std;
  Handle<association_t> association;
  evt.getByToken(associationToken_, association);
  unique_ptr<collection_t> selected(new collection_t);
  vector<typename CVal::value_type> selectedValues;
  size_t size = association->size();
  selected->reserve(size);
  selectedValues.reserve(size);
  for (typename association_t::const_iterator i = association->begin(); i != association->end(); ++i) {
    const typename association_t::key_type& obj = *i->first;
    if (selectKey_(obj) && selectVal_(i->second)) {
      typedef typename edm::clonehelper::CloneTrait<collection_t>::type clone_t;
      selected->push_back(clone_t::clone(obj));
      selectedValues.push_back(i->second);
    }
  }
  // new association must be created after the
  // selected collection is full because it uses the size
  KeyRefProd ref = evt.getRefBeforePut<collection_t>();
  unique_ptr<association_t> selectedAssociation(new association_t(ref, selected.get()));
  size = selected->size();
  OrphanHandle<collection_t> oh = evt.put(std::move(selected));
  for (size_t i = 0; i != size; ++i)
    selectedAssociation->setValue(i, selectedValues[i]);
  evt.put(std::move(selectedAssociation));
}

#endif
