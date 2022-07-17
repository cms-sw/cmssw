#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTClusterAssociationMap.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTStubAssociationMap.h"
#include "L1Trigger/VertexFinder/interface/AnalysisSettings.h"
#include "L1Trigger/VertexFinder/interface/InputData.h"

#include <map>

using namespace std;

namespace l1tVertexFinder {

  InputData::InputData() {}

  InputData::InputData(const edm::Event& iEvent,
                       const edm::EventSetup& iSetup,
                       const AnalysisSettings& settings,
                       const edm::EDGetTokenT<edm::HepMCProduct> hepMCToken,
                       const edm::EDGetTokenT<edm::View<reco::GenParticle>> genParticlesToken,
                       const edm::EDGetTokenT<edm::View<TrackingParticle>> tpToken,
                       const edm::EDGetTokenT<edm::ValueMap<l1tVertexFinder::TP>> tpValueMapToken,
                       const edm::EDGetTokenT<DetSetVec> stubToken,
                       edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopoToken,
                       edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> tGeomToken) {
    // Get the tracker geometry info needed to unpack the stub info.
    const TrackerTopology& tTopo = iSetup.getData(tTopoToken);
    const TrackerGeometry& tGeom = iSetup.getData(tGeomToken);

    // Get stub info, by looping over modules and then stubs inside each module.
    // Also get the association map from stubs to tracking particles.
    edm::Handle<DetSetVec> ttStubHandle;
    iEvent.getByToken(stubToken, ttStubHandle);

    std::set<DetId> lStubDetIds;
    for (DetSetVec::const_iterator p_module = ttStubHandle->begin(); p_module != ttStubHandle->end(); p_module++) {
      for (DetSet::const_iterator p_ttstub = p_module->begin(); p_ttstub != p_module->end(); p_ttstub++) {
        lStubDetIds.insert(p_ttstub->getDetId());
      }
    }

    std::map<DetId, DetId> stubGeoDetIdMap;
    for (auto gd = tGeom.dets().begin(); gd != tGeom.dets().end(); gd++) {
      DetId detid = (*gd)->geographicalId();
      if (detid.subdetId() != StripSubdetector::TOB && detid.subdetId() != StripSubdetector::TID)
        continue;  // only run on OT
      if (!tTopo.isLower(detid))
        continue;                             // loop on the stacks: choose the lower arbitrarily
      DetId stackDetid = tTopo.stack(detid);  // Stub module detid

      if (lStubDetIds.count(stackDetid) > 0) {
        assert(stubGeoDetIdMap.count(stackDetid) == 0);
        stubGeoDetIdMap[stackDetid] = detid;
      }
    }
    assert(lStubDetIds.size() == stubGeoDetIdMap.size());

    // Get TrackingParticle info
    edm::Handle<edm::View<TrackingParticle>> tpHandle;
    edm::Handle<edm::ValueMap<TP>> tpValueMapHandle;
    iEvent.getByToken(tpToken, tpHandle);
    iEvent.getByToken(tpValueMapToken, tpValueMapHandle);
    edm::ValueMap<TP> tpValueMap = *tpValueMapHandle;

    for (unsigned int i = 0; i < tpHandle->size(); i++) {
      if (tpValueMap[tpHandle->refAt(i)].use()) {
        tpPtrToRefMap_[tpHandle->ptrAt(i)] = tpHandle->refAt(i);
      }
    }

    // Find the various vertices
    genPt_ = 0.;
    genPt_PU_ = 0.;
    for (const auto& [edmPtr, edmRef] : tpPtrToRefMap_) {
      TP tp = tpValueMap[edmRef];
      if (tp.physicsCollision()) {
        genPt_ += tp->pt();
      } else {
        genPt_PU_ += tp->pt();
      }
      if (settings.debug() > 2) {
        edm::LogInfo("InputData") << "InputData::genPt in the event " << genPt_;
      }

      if (tp.useForAlgEff()) {
        vertex_.insert(tp);
      } else if (tp.useForVertexReco()) {
        bool found = false;
        for (unsigned int i = 0; i < vertices_.size(); ++i) {
          if (tp->vz() == vertices_[i].vz()) {
            vertices_[i].insert(tp);
            found = true;
            break;
          }
        }
        if (!found) {
          Vertex vertex(tp->vz());
          vertex.insert(tp);
          vertices_.push_back(vertex);
        }
      }
    }

    for (const Vertex& vertex : vertices_) {
      if (vertex.numTracks() >= settings.vx_minTracks())
        recoVertices_.push_back(vertex);
    }

    if (settings.debug() > 2)
      edm::LogInfo("InputData") << "InputData::" << vertices_.size() << " pileup vertices in the event, "
                                << recoVertices_.size() << " reconstructable";

    vertex_.computeParameters();
    if (settings.debug() > 2)
      edm::LogInfo("InputData") << "InputData::Vertex " << vertex_.z0() << " containing " << vertex_.numTracks()
                                << " total pT " << vertex_.pT();

    for (unsigned int i = 0; i < vertices_.size(); ++i) {
      vertices_[i].computeParameters();
    }

    for (unsigned int i = 0; i < recoVertices_.size(); ++i) {
      recoVertices_[i].computeParameters();
    }

    std::sort(vertices_.begin(), vertices_.end(), SortVertexByPt());
    std::sort(recoVertices_.begin(), recoVertices_.end(), SortVertexByPt());

    // Form the HepMC and GenParticle based vertices
    edm::Handle<edm::HepMCProduct> HepMCEvt;
    iEvent.getByToken(hepMCToken, HepMCEvt);

    edm::Handle<edm::View<reco::GenParticle>> GenParticleHandle;
    iEvent.getByToken(genParticlesToken, GenParticleHandle);

    if (!HepMCEvt.isValid() && !GenParticleHandle.isValid()) {
      throw cms::Exception("Neither the edm::HepMCProduct nor the generator particles are available.");
    }
    if (HepMCEvt.isValid()) {
      const HepMC::GenEvent* MCEvt = HepMCEvt->GetEvent();
      for (HepMC::GenEvent::vertex_const_iterator ivertex = MCEvt->vertices_begin(); ivertex != MCEvt->vertices_end();
           ++ivertex) {
        bool hasParentVertex = false;

        // Loop over the parents looking to see if they are coming from a production vertex
        for (HepMC::GenVertex::particle_iterator iparent = (*ivertex)->particles_begin(HepMC::parents);
             iparent != (*ivertex)->particles_end(HepMC::parents);
             ++iparent)
          if ((*iparent)->production_vertex()) {
            hasParentVertex = true;
            break;
          }

        // Reject those vertices with parent vertices
        if (hasParentVertex)
          continue;
        // Get the position of the vertex
        HepMC::FourVector pos = (*ivertex)->position();
        const double mm = 0.1;  // [mm] --> [cm]
        hepMCVertex_ = Vertex(pos.z() * mm);
        break;  // there should be one single primary vertex
      }         // end loop over gen vertices
    }
    if (GenParticleHandle.isValid()) {
      for (const auto& genpart : *GenParticleHandle) {
        if ((genpart.status() != 1) || (genpart.numberOfMothers() == 0))  // not stable or one of the incoming hadrons
          continue;
        genVertex_ = Vertex(genpart.vz());
        break;
      }
    }
    if ((hepMCVertex_.vz() == 0.0) && (genVertex_.vz() == 0.0)) {
      throw cms::Exception("Neither the HepMC vertex nor the generator particle vertex were found.");
    }

  }  // end InputData::InputData

  InputData::~InputData() {}

}  // end namespace l1tVertexFinder
