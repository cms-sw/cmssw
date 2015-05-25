#include "RecoTracker/FinalTrackSelectors/interface/TrackCollectionCloner.h"


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
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


#include<vector>
#include<memory>
#include<cassert>
namespace {
  class TrackCollectionMerger final : public edm::global::EDProducer<> {
   public:
    explicit TrackCollectionMerger(const edm::ParameterSet& conf) :
      collectionCloner(*this, conf, true),
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
	srcMVAs.push_back(consumes<MVACollection>(edm::InputTag(it,"MVAVals")));
	srcQuals.push_back(consumes<QualityMaskCollection>(edm::InputTag(it,"QualityMasks")));
      }

      assert(srcColls.size()==srcQuals.size());
      

      produces<MVACollection>("MVAVals");
      produces<QualityMaskCollection>("QualityMasks");

    }
      
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<std::vector<edm::InputTag> >("trackProducers");
      desc.add<std::vector<std::string> >("inputClassifiers",std::vector<std::string>());
      desc.add<double>("ShareFrac",.19);
      desc.add<double>("foundHitBonus",10.);
      desc.add<double>("lostHitPenalty",5.);
      desc.add<unsigned int>("minShareHits",2);
      desc.add<bool>("allowFirstHitShare",true);
      desc.add<std::string>("minQuality","loose");
      descriptions.add("TrackCollectionMerger", desc);
    }

   private:


    TrackCollectionCloner collectionCloner;
    std::vector<TrackCollectionCloner::Tokens> srcColls;

        using MVACollection = std::vector<float>;
    using QualityMaskCollection = std::vector<unsigned char>;
    
 
    
    std::vector<edm::EDGetTokenT<MVACollection>> srcMVAs;
    std::vector<edm::EDGetTokenT<QualityMaskCollection>> srcQuals;

	
    float m_foundHitBonus;
    float m_lostHitPenalty;
    float m_shareFrac;
    unsigned int m_minShareHits;
    reco::TrackBase::TrackQuality m_minQuality;
    bool  m_allowFirstHitShare;
    
    virtual void produce(edm::StreamID, edm::Event& evt, const edm::EventSetup&) const override;
    
    
    

  };

}
  
#include "CommonTools/Utils/interface/DynArray.h"


namespace {
  
  void  TrackCollectionMerger::produce(edm::StreamID, edm::Event& evt, const edm::EventSetup&) const {

    TrackCollectionCloner::Producer producer(evt, collectionCloner);

    // load collections
    auto collsSize = srcColls.size();
    auto rSize=0U;
    declareDynArray(reco::TrackCollection const *, collsSize, trackColls);
    declareDynArray(edm::Handle<reco::TrackCollection>, collsSize, trackHandles);
    for (auto i=0U; i< collsSize; ++i) {
      evt.getByToken(srcColls[i].hSrcTrackToken_,trackHandles[i]);
      trackColls[i] = trackHandles[i].product();
      rSize += (*trackColls[i]).size();
    }


    unsigned char qualMask = ~0;
    if (m_minQuality!=reco::TrackBase::undefQuality) qualMask = 1<<m_minQuality; 
    
    
    // load tracks
    initDynArray(unsigned int,collsSize,nGoods,0);
    declareDynArray(float,rSize, mvas);
    declareDynArray(unsigned char,rSize, quals);
    declareDynArray(unsigned int,rSize, tkInd);
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
	tkInd[k]=j;
	++k;
	++nGoods[i];
       }
    }

    // auto ntotTk=k;
    // load hits...
    

    
    
    // products
    std::unique_ptr<MVACollection> pmvas(new MVACollection());
    std::unique_ptr<QualityMaskCollection> pquals(new QualityMaskCollection());


    evt.put(std::move(pmvas),"MVAValues");
    evt.put(std::move(pquals),"QualityMasks");

    
  }

  
}


#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(TrackCollectionMerger);
