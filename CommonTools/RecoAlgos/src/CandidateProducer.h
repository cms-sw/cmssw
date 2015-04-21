#ifndef RecoAlgos_CandidateProducer_h
#define RecoAlgos_CandidateProducer_h
/** \class CandidateProducer
 *
 * Framework module that produces a collection
 * of candidates from generic compoment
 *
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.4 $
 *
 * $Id: CandidateProducer.h,v 1.4 2010/02/11 00:10:53 wmtan Exp $
 *
 */
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "CommonTools/UtilAlgos/interface/MasterCollectionHelper.h"
#include "CommonTools/UtilAlgos/interface/AnySelector.h"
#include "CommonTools/UtilAlgos/interface/EventSetupInitTrait.h"

namespace converter {
  namespace helper {
    template<typename T>
    struct CandConverter { };

    struct ConcreteCreator {
      template<typename CColl, typename Comp, typename Conv>
      static void create(size_t idx, CColl & cands, const Comp & components, Conv & converter) {
	typename Conv::Candidate c;
	typedef edm::Ref<std::vector<typename Conv::value_type> > ref_type;
	ref_type ref = components.template getConcreteRef<ref_type>(idx);
	converter.convert(ref, c);
	cands.push_back(c);
      }
    };

    struct PolymorphicCreator {
      template<typename CColl, typename Comp, typename Conv>
      static void create(size_t idx, CColl & cands, const Comp & components, Conv & converter) {
	typename Conv::Candidate * c = new typename Conv::Candidate;
	typedef edm::Ref<std::vector<typename Conv::value_type> > ref_type;
	ref_type ref = components.template getConcreteRef<ref_type>(idx);
	converter.convert(ref, * c);
	cands.push_back(c);
      }
    };

    template<typename CColl>
    struct CandCreator {
      typedef ConcreteCreator type;
    };

    template<>
    struct CandCreator<reco::CandidateCollection> {
      typedef PolymorphicCreator type;
    };
  }
}

template<typename TColl, typename CColl, typename Selector = AnySelector,
	 typename Conv = typename converter::helper::CandConverter<typename TColl::value_type>::type,
	 typename Creator = typename converter::helper::CandCreator<CColl>::type,
	 typename Init = typename ::reco::modules::EventSetupInit<Selector>::type>
class CandidateProducer : public edm::stream::EDProducer<> {
public:
  /// constructor from parameter set
  CandidateProducer(const edm::ParameterSet & cfg) :
    srcToken_(consumes<TColl>(cfg.template getParameter<edm::InputTag>("src"))),
    converter_(cfg),
    selector_(reco::modules::make<Selector>(cfg, consumesCollector())),
    initialized_(false) {
    produces<CColl>();
  }
  /// destructor
  ~CandidateProducer() { }

private:
  /// begin job (first run)
  void beginRun(const edm::Run&, const edm::EventSetup& es) override {
    if (!initialized_) {
      converter_.beginFirstRun(es);
      initialized_ = true;
    }
  }
  /// process one event
  void produce(edm::Event& evt, const edm::EventSetup& es) override {
    edm::Handle<TColl> src;
    evt.getByToken(srcToken_, src);
    Init::init(selector_, evt, es);
    ::helper::MasterCollection<TColl> master(src, evt);
    std::auto_ptr<CColl> cands(new CColl);
    if(src->size()!= 0) {
      size_t size = src->size();
      cands->reserve(size);
      for(size_t idx = 0; idx != size; ++ idx) {
	if(selector_((*src)[idx]))
	  Creator::create(master.index(idx), *cands, master, converter_);
      }
    }
    evt.put(cands);
  }
  /// label of source collection and tag
  edm::EDGetTokenT<TColl> srcToken_;
  /// converter helper
  Conv converter_;
  /// selector
  Selector selector_;
  /// particles initialized?
  bool initialized_;
};

#endif
