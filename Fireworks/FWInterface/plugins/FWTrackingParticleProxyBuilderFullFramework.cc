#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWProxyBuilderBase.h"
#include "Fireworks/Core/interface/Context.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimGeneral/TrackingAnalysis/interface/SimHitTPAssociationProducer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

#include "DataFormats/DetId/interface/DetId.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Common/interface/EventBase.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "Fireworks/Core/interface/FWProxyBuilderConfiguration.h"
#include "Fireworks/Core/interface/FWParameters.h"

#include "TEveTrack.h"
#include "TEveCompound.h"
#include "TEveManager.h"
#include "TEveBrowser.h"
#include "TEveTrackPropagator.h"

class FWTrackingParticleProxyBuilderFullFramework : public FWSimpleProxyBuilderTemplate<TrackingParticle> {
public:
  FWTrackingParticleProxyBuilderFullFramework(void) : m_assocList(nullptr) {}
  ~FWTrackingParticleProxyBuilderFullFramework(void) override {}

  // virtual void setItem(const FWEventItem* iItem) override;

  REGISTER_PROXYBUILDER_METHODS();

  FWTrackingParticleProxyBuilderFullFramework(const FWTrackingParticleProxyBuilderFullFramework&) = delete;
  const FWTrackingParticleProxyBuilderFullFramework& operator=(const FWTrackingParticleProxyBuilderFullFramework&) =
      delete;

private:
  using FWSimpleProxyBuilderTemplate<TrackingParticle>::build;
  void build(const FWEventItem* iItem, TEveElementList* product, const FWViewContext*) override;

  void build(const TrackingParticle& iData,
             unsigned int iIndex,
             TEveElement& oItemHolder,
             const FWViewContext*) override;

  edm::Handle<TrackingParticleCollection> tpch;
  const SimHitTPAssociationProducer::SimHitTPAssociationList* m_assocList;
};

//______________________________________________________________________________

/*
  void FWTrackingParticleProxyBuilderFullFramework::setItem(const FWEventItem* iItem)
  {
  printf("set item\n");
  FWProxyBuilderBase::setItem(iItem);
  }
*/
//______________________________________________________________________________
void FWTrackingParticleProxyBuilderFullFramework::build(const FWEventItem* iItem,
                                                        TEveElementList* product,
                                                        const FWViewContext*) {
  // setup event handles amd call function from parent class

  const edm::Event* event = (const edm::Event*)item()->getEvent();
  if (event) {
    // get collection handle
    edm::InputTag coltag(item()->moduleLabel(), item()->productInstanceLabel(), item()->processName());
    event->getByLabel(coltag, tpch);

    // AMT todo: check if there is any other way getting the list other than this
    //           ifnot, set proces name as a configurable parameter
    edm::Handle<SimHitTPAssociationProducer::SimHitTPAssociationList> simHitsTPAssoc;
    try {
      event->getByLabel("xxx", simHitsTPAssoc);
      m_assocList = &*simHitsTPAssoc;
    } catch (const std::exception& e) {
      std::cerr << " FWTrackingParticleProxyBuilderFullFramework::setItem() Can't get hits association list "
                << e.what() << std::endl;
    }
    /*
      // debug propagator
      gEve->GetBrowser()->MapWindow();
      gEve->AddToListTree(context().getTrackPropagator(), true);
      context().getTrackPropagator()->SetRnrReferences(true);
      */
  }
  FWSimpleProxyBuilder::build(iItem, product, nullptr);
}
//______________________________________________________________________________
void FWTrackingParticleProxyBuilderFullFramework::build(const TrackingParticle& iData,
                                                        unsigned int tpIdx,
                                                        TEveElement& comp,
                                                        const FWViewContext*) {
  TEveRecTrack t;
  t.fBeta = 1.0;
  t.fP = TEveVector(iData.px(), iData.py(), iData.pz());
  t.fV = TEveVector(iData.vx(), iData.vy(), iData.vz());
  t.fSign = iData.charge();

  TEveTrack* track = new TEveTrack(&t, context().getTrackPropagator());
  if (t.fSign == 0)
    track->SetLineStyle(7);

  track->MakeTrack();
  setupAddElement(track, &comp);
  // printf("add track %d \n", tpIdx);

  if (m_assocList) {
    TEvePointSet* pointSet = new TEvePointSet;
    setupAddElement(pointSet, &comp);

    const FWGeometry* geom = item()->getGeom();
    float local[3];
    float localDir[3];
    float global[3] = {0.0, 0.0, 0.0};
    float globalDir[3] = {0.0, 0.0, 0.0};

    TrackingParticleRef tpr(tpch, tpIdx);
    std::pair<TrackingParticleRef, TrackPSimHitRef> clusterTPpairWithDummyTP(tpr, TrackPSimHitRef());
    auto range = std::equal_range(m_assocList->begin(),
                                  m_assocList->end(),
                                  clusterTPpairWithDummyTP,
                                  SimHitTPAssociationProducer::simHitTPAssociationListGreater);
    // printf("TrackingParticle[%d] P(%.1f, %.1f, %.1f) matches %d hits\n", tpIdx,iData.px(), iData.py(), iData.pz() ,(int)(range.second-range.first ));

    std::vector<const PSimHit*> phits;
    for (auto ri = range.first; ri != range.second; ++ri)
      phits.push_back(ri->second.get());

    std::sort(phits.begin(), phits.end(), [](const PSimHit* a, const PSimHit* b) { return a->tof() < b->tof(); });
    for (auto phi = phits.begin(); phi != phits.end(); ++phi) {
      const PSimHit* phit = *phi;

      local[0] = phit->localPosition().x();
      local[1] = phit->localPosition().y();
      local[2] = phit->localPosition().z();

      localDir[0] = phit->momentumAtEntry().x();
      localDir[1] = phit->momentumAtEntry().y();
      localDir[2] = phit->momentumAtEntry().z();

      geom->localToGlobal(phit->detUnitId(), local, global);
      geom->localToGlobal(phit->detUnitId(), localDir, globalDir, false);
      pointSet->SetNextPoint(global[0], global[1], global[2]);

      //printf("localP = (%f, %f, %f) globalP = (%f, %f, %f), loss = %f, tof =%f\n", localDir[0], localDir[1], localDir[2],
      //       globalDir[0], globalDir[1], globalDir[2],
      //       phit->energyLoss(), phit->tof());
      track->AddPathMark(TEvePathMark(TEvePathMark::kReference,
                                      TEveVector(global[0], global[1], global[2]),
                                      TEveVector(globalDir[0], globalDir[1], globalDir[2])));
    }
  }
}

REGISTER_FWPROXYBUILDER(FWTrackingParticleProxyBuilderFullFramework,
                        TrackingParticle,
                        "TrackingParticleWithPSimHits",
                        FWViewType::kAll3DBits | FWViewType::kAllRPZBits);
