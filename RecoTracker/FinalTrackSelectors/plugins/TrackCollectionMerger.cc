#include "RecoTracker/FinalTrackSelectors/interface/TrackCollectionCloner.h"


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

#include "RecoTracker/FinalTrackSelectors/interface/TrackAlgoPriorityOrder.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"

#include<vector>
#include<memory>
#include<cassert>
namespace {
  class TrackCollectionMerger final : public edm::global::EDProducer<> {
   public:
    explicit TrackCollectionMerger(const edm::ParameterSet& conf) :
      collectionCloner(*this, conf, true),
      priorityName_(conf.getParameter<std::string>("trackAlgoPriorityOrder")),
      m_foundHitBonus(conf.getParameter<double>("foundHitBonus")),
      m_lostHitPenalty(conf.getParameter<double>("lostHitPenalty")),
      m_shareFrac(conf.getParameter<double>("shareFrac")),
      m_minShareHits(conf.getParameter<unsigned int>("minShareHits")),
      m_minQuality(reco::TrackBase::qualityByName(conf.getParameter<std::string>("minQuality"))),
      m_allowFirstHitShare(conf.getParameter<bool>("allowFirstHitShare"))
{
      for (auto const & it : conf.getParameter<std::vector<edm::InputTag> >("trackProducers") )
	srcColls.emplace_back(it,consumesCollector());
      for (auto const & it : conf.getParameter<std::vector<std::string> >("inputClassifiers")) {
	srcMVAs.push_back(consumes<MVACollection>(edm::InputTag(it,"MVAValues")));
	srcQuals.push_back(consumes<QualityMaskCollection>(edm::InputTag(it,"QualityMasks")));
      }

      assert(srcColls.size()==srcQuals.size());
      

      produces<MVACollection>("MVAValues");
      produces<QualityMaskCollection>("QualityMasks");

    }
      
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<std::vector<edm::InputTag> >("trackProducers",std::vector<edm::InputTag>());
      desc.add<std::vector<std::string> >("inputClassifiers",std::vector<std::string>());
      desc.add<std::string>("trackAlgoPriorityOrder", "trackAlgoPriorityOrder");
      desc.add<double>("shareFrac",.19);
      desc.add<double>("foundHitBonus",10.);
      desc.add<double>("lostHitPenalty",5.);
      desc.add<unsigned int>("minShareHits",2);
      desc.add<bool>("allowFirstHitShare",true);
      desc.add<std::string>("minQuality","loose");
      TrackCollectionCloner::fill(desc);
      descriptions.add("TrackCollectionMerger", desc);
    }

   private:


    TrackCollectionCloner collectionCloner;
    std::vector<TrackCollectionCloner::Tokens> srcColls;

    using MVACollection = std::vector<float>;
    using QualityMaskCollection = std::vector<unsigned char>;
    
    using IHit = std::pair<unsigned int, TrackingRecHit const *>;
    using IHitV = std::vector<IHit>;
    
    std::vector<edm::EDGetTokenT<MVACollection>> srcMVAs;
    std::vector<edm::EDGetTokenT<QualityMaskCollection>> srcQuals;
	
    std::string priorityName_;

    float m_foundHitBonus;
    float m_lostHitPenalty;
    float m_shareFrac;
    unsigned int m_minShareHits;
    reco::TrackBase::TrackQuality m_minQuality;
    bool  m_allowFirstHitShare;
    
    virtual void produce(edm::StreamID, edm::Event& evt, const edm::EventSetup&) const override;
    

    bool areDuplicate(IHitV const& rh1, IHitV const& rh2) const;
      

  };

}
  
#include "CommonTools/Utils/interface/DynArray.h"


namespace {
  
  void  TrackCollectionMerger::produce(edm::StreamID, edm::Event& evt, const edm::EventSetup& es) const {

    TrackCollectionCloner::Producer producer(evt, collectionCloner);

    edm::ESHandle<TrackAlgoPriorityOrder> priorityH;
    es.get<CkfComponentsRecord>().get(priorityName_, priorityH);
    auto const & trackAlgoPriorityOrder = *priorityH;

    // load collections
    auto collsSize = srcColls.size();
    auto rSize=0U;
    declareDynArray(reco::TrackCollection const *, collsSize, trackColls);
    for (auto i=0U; i< collsSize; ++i) {
      trackColls[i] = &srcColls[i].tracks(evt);
      rSize += (*trackColls[i]).size();
    }


    unsigned char qualMask = ~0;
    if (m_minQuality!=reco::TrackBase::undefQuality) qualMask = 1<<m_minQuality; 
    
    
    // load tracks
    initDynArray(unsigned int,collsSize,nGoods,0);
    declareDynArray(float,rSize, mvas);
    declareDynArray(unsigned char,rSize, quals);
    declareDynArray(unsigned int,rSize, tkInds);
    auto k=0U;
    for (auto i=0U; i< collsSize; ++i) {
      auto const & tkColl = *trackColls[i];
      auto size = tkColl.size();
      edm::Handle<MVACollection> hmva;
      evt.getByToken(srcMVAs[i], hmva);
      assert((*hmva).size()==size);
      edm::Handle<QualityMaskCollection> hqual;
      evt.getByToken(srcQuals[i], hqual);
      for (auto j=0U; j<size; ++j) {
	if (! (qualMask&(* hqual)[j]) ) continue;
	mvas[k]=(*hmva)[j];
	quals[k] = (*hqual)[j];
	tkInds[k]=j;
	++k;
	++nGoods[i];
       }
    }

    auto ntotTk=k;
    std::vector<bool> selected(ntotTk,true);

    declareDynArray(reco::TrackBase::TrackAlgorithm, ntotTk, algo);
    declareDynArray(reco::TrackBase::TrackAlgorithm, ntotTk, oriAlgo);
    declareDynArray(reco::TrackBase::AlgoMask, ntotTk,  algoMask);
    

    // merge (if more than one collection...)

    auto merger = [&]()->void {
    
      // load hits and score
      declareDynArray(float,ntotTk,score);
      declareDynArray(IHitV, ntotTk, rh1);
      
      k=0U;
      for (auto i=0U; i< collsSize; ++i) {
	auto const & tkColl = *trackColls[i];
	for (auto j=0U; j< nGoods[i]; ++j) {
	  auto const & track = tkColl[tkInds[k]];
	  algo[k] = track.algo();
	  oriAlgo[k] = track.originalAlgo();
	  algoMask[k] = track.algoMask();

	  auto validHits=track.numberOfValidHits();
	  auto validPixelHits=track.hitPattern().numberOfValidPixelHits();
	  auto lostHits=track.numberOfLostHits();
	  score[k] = m_foundHitBonus*validPixelHits+m_foundHitBonus*validHits - m_lostHitPenalty*lostHits - track.chi2();
	  
	  auto & rhv =  rh1[k];
	  rhv.reserve(validHits) ;
	  auto compById = [](IHit const &  h1, IHit const & h2) {return h1.first < h2.first;};
	  for (auto it = track.recHitsBegin();  it != track.recHitsEnd(); ++it) {
	    auto const & hit = *(*it);
	    auto id = hit.rawId() ;
	    if(hit.geographicalId().subdetId()>2)  id &= (~3); // mask mono/stereo in strips...
	    if likely(hit.isValid()) { rhv.emplace_back(id,&hit); std::push_heap(rhv.begin(),rhv.end(),compById); }
	  }
	  std::sort_heap(rhv.begin(),rhv.end(),compById);
	  
	  
	  ++k;
	}
      }
      assert(ntotTk==k);
      
      auto seti = [&](unsigned int ii, unsigned int jj) {
	selected[jj]=false;
	selected[ii]=true;
	if (trackAlgoPriorityOrder.priority(oriAlgo[jj]) < trackAlgoPriorityOrder.priority(oriAlgo[ii])) oriAlgo[ii] = oriAlgo[jj];
	algoMask[ii] |= algoMask[jj];
	quals[ii] |= (1<<reco::TrackBase::confirmed);
	algoMask[jj] = algoMask[ii];  // in case we keep discarded
	quals[jj] |= (1<<reco::TrackBase::discarded);
      };



      auto iStart2=0U;
      for (auto i=0U; i<collsSize-1; ++i) {
	auto iStart1=iStart2;
	iStart2=iStart1+nGoods[i];
	for (auto t1=iStart1; t1<iStart2; ++t1) {
	  if (!selected[t1]) continue;
	  auto score1 = score[t1];
	  for (auto t2=iStart2; t2 <ntotTk; ++t2) {
	    if (!selected[t1]) break;
	    if (!selected[t2]) continue;
	    if (!areDuplicate(rh1[t1],rh1[t2])) continue;
	    auto score2 = score[t2];
	    constexpr float almostSame = 0.01f; // difference rather than ratio due to possible negative values for score

	    if ( score1 - score2 > almostSame ) {
	      seti(t1,t2);
	    } else if ( score2 - score1 > almostSame ) {
	      seti(t2,t1);
	    } else {  // take best
	      constexpr unsigned int qmask =  (1<<reco::TrackBase::loose|1<<reco::TrackBase::tight|1<<reco::TrackBase::highPurity);
	      if ( (quals[t1]&qmask) == (quals[t2]&qmask) ) {
		// take first
		if (trackAlgoPriorityOrder.priority(algo[t1]) <= trackAlgoPriorityOrder.priority(algo[t2])) {
                  seti(t1,t2);    
		} else {
                  seti(t2,t1);    
		}
	      } else if ( (quals[t1]&qmask) > (quals[t2]&qmask) ) 
		seti(t1,t2);    
	      else 
		seti(t2,t1);    
	    }  // end ifs...
	      
	  } // end t2
	} // end t1
      }  // end colls
    }; // end merger;


    if (collsSize>1) merger();
    
    // products
    auto pmvas = std::make_unique<MVACollection>();
    auto pquals = std::make_unique<QualityMaskCollection>();
				    
    // clone selected tracks...
    auto nsel=0U;
    auto iStart2=0U;
    auto isel=0U;
    for (auto i=0U; i<collsSize; ++i) {
      std::vector<unsigned int> selId;
      std::vector<unsigned int>  tid;
      auto iStart1=iStart2;
      iStart2=iStart1+nGoods[i];
      assert(producer.selTracks_->size()==isel);
      for (auto t1=iStart1; t1<iStart2; ++t1) {
	if (!selected[t1]) continue;
	++nsel;
	tid.push_back(t1);
	selId.push_back(tkInds[t1]);
	pmvas->push_back(mvas[t1]);
	pquals->push_back(quals[t1]);
      }
      producer(srcColls[i],selId);
      assert(producer.selTracks_->size()==nsel);
      assert(tid.size()==nsel-isel);
      auto k=0U;
      for (;isel<nsel;++isel) {
	auto & otk = (*producer.selTracks_)[isel];
	otk.setQualityMask((*pquals)[isel]);
	otk.setOriginalAlgorithm(oriAlgo[tid[k]]);
	otk.setAlgoMask(algoMask[tid[k++]]);
      }
      assert(tid.size()==k);
    }

    assert(producer.selTracks_->size()==pmvas->size());

    evt.put(std::move(pmvas),"MVAValues");
    evt.put(std::move(pquals),"QualityMasks");

    
  }



  bool TrackCollectionMerger::areDuplicate(IHitV const& rh1, IHitV const& rh2) const {
    auto nh1=rh1.size();
    auto nh2=rh2.size();


    auto share = // use_sharesInput_ ?
      [](const TrackingRecHit*  it,const TrackingRecHit*  jt)->bool { return it->sharesInput(jt,TrackingRecHit::some); };
    //:
    // [](const TrackingRecHit*  it,const TrackingRecHit*  jt)->bool {
    //   float delta = std::abs ( it->localPosition().x()-jt->localPosition().x() );
    //  return (it->geographicalId()==jt->geographicalId())&&(delta<epsilon_);
    //	  };


    //loop over rechits
    int noverlap=0;
    int firstoverlap=0;
    // check first hit  (should use REAL first hit?)
    if unlikely(m_allowFirstHitShare && rh1[0].first==rh2[0].first ) {
	if (share( rh1[0].second, rh2[0].second)) firstoverlap=1;
      }

    // exploit sorting
    unsigned int jh=0;
    unsigned int ih=0;
    while (ih!=nh1 && jh!=nh2) {
      // break if not enough to go...
      // if ( nprecut-noverlap+firstoverlap > int(nh1-ih)) break;
      // if ( nprecut-noverlap+firstoverlap > int(nh2-jh)) break;
      auto const id1 = rh1[ih].first;
      auto const id2 = rh2[jh].first;
      if (id1<id2) ++ih;
      else if (id2<id1) ++jh;
      else {
	// in case of split-hit do full conbinatorics
	auto li=ih; while( (++li)!=nh1 && id1 == rh1[li].first);
	auto lj=jh; while( (++lj)!=nh2 && id2 == rh2[lj].first);
	for (auto ii=ih; ii!=li; ++ii)
	  for (auto jj=jh; jj!=lj; ++jj) {
	    if (share( rh1[ii].second, rh2[jj].second)) noverlap++;
	  }
	jh=lj; ih=li;
      } // equal ids
      
    } //loop over ih & jh
    
    return  noverlap >= int(m_minShareHits)     &&
            (noverlap-firstoverlap) > (std::min(nh1,nh2)-firstoverlap)*m_shareFrac;

  }

  
}


#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(TrackCollectionMerger);
