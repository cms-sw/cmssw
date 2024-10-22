#ifndef UtilAlgos_UniqueMerger_h
#define UtilAlgos_UniqueMerger_h
/** \class UniqueMerger
 *
 * Merges an arbitrary number of collections
 * into a single collection, without duplicates. Based on logic from Merger.h.
 * This class template differs from Merger.h in that it uses a set instead of std::vector.
 * This requires the OutputCollection type to be sortable, 
 * which allows us to search for elements downstream efficiently.
 *
 * Template parameters:
 * - C : collection type
 * - P : policy class that specifies how objects
 *       in the collection are are cloned
 *
 * \author Lauren Hay
 *
 * \version $Revision: 1.2 $
 *
 * 
 */

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/transform.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/CloneTrait.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <vector>

template <typename InputCollection,
          typename OutputCollection = InputCollection,
          typename P = typename edm::clonehelper::CloneTrait<InputCollection>::type>
class UniqueMerger : public edm::global::EDProducer<> {
public:
  typedef std::set<typename OutputCollection::value_type> set_type;
  /// constructor from parameter set
  explicit UniqueMerger(const edm::ParameterSet&);
  /// destructor
  ~UniqueMerger() override;

private:
  /// process an event
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  /// vector of strings
  typedef std::vector<edm::EDGetTokenT<InputCollection> > vtoken;
  /// labels of the collections to be merged
  vtoken srcToken_;
  /// choose whether to skip null/invalid pointers
  bool skipNulls_;
  /// choose whether to warn when skipping pointers
  bool warnOnSkip_;
};

template <typename InputCollection, typename OutputCollection, typename P>
UniqueMerger<InputCollection, OutputCollection, P>::UniqueMerger(const edm::ParameterSet& par)
    : srcToken_(edm::vector_transform(par.template getParameter<std::vector<edm::InputTag> >("src"),
                                      [this](edm::InputTag const& tag) { return consumes<InputCollection>(tag); })),
      skipNulls_(par.getParameter<bool>("skipNulls")),
      warnOnSkip_(par.getParameter<bool>("warnOnSkip")) {
  produces<OutputCollection>();
}

template <typename InputCollection, typename OutputCollection, typename P>
UniqueMerger<InputCollection, OutputCollection, P>::~UniqueMerger() {}

template <typename InputCollection, typename OutputCollection, typename P>
void UniqueMerger<InputCollection, OutputCollection, P>::produce(edm::StreamID,
                                                                 edm::Event& evt,
                                                                 const edm::EventSetup&) const {
  set_type coll_set;
  for (auto const& s : srcToken_) {
    for (auto const& c : evt.get(s)) {
      if (!skipNulls_ || (c.isNonnull() && c.isAvailable())) {
        coll_set.emplace(P::clone(c));
      } else if (warnOnSkip_) {
        edm::LogWarning("InvalidPointer") << "Found an invalid pointer. Will not merge to collection.";
      }
    }
  }
  std::unique_ptr<OutputCollection> coll(new OutputCollection(coll_set.size()));
  std::copy(coll_set.begin(), coll_set.end(), coll->begin());
  evt.put(std::move(coll));
}

#endif
