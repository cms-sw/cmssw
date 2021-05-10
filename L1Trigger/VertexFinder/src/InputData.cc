#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
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

  InputData::InputData(const edm::Event& iEvent,
                       const edm::EventSetup& iSetup,
                       const AnalysisSettings& settings,
                       const edm::EDGetTokenT<edm::HepMCProduct> hepMCToken,
                       const edm::EDGetTokenT<edm::View<reco::GenParticle>> genParticlesToken,
                       edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerGeometryToken_,
                       edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> trackerTopologyToken_,
                       const edm::EDGetTokenT<TrackingParticleCollection> tpToken,
                       const edm::EDGetTokenT<DetSetVec> stubToken,
                       const edm::EDGetTokenT<TTStubAssMap> stubTruthToken,
                       const edm::EDGetTokenT<TTClusterAssMap> clusterTruthToken) {
    vTPs_.reserve(2500);
    vAllStubs_.reserve(35000);

    // Get TrackingParticle info
    edm::Handle<TrackingParticleCollection> tpHandle;
    iEvent.getByToken(tpToken, tpHandle);

    genPt_ = 0.;
    genPt_PU_ = 0.;

    for (unsigned int i = 0; i < tpHandle->size(); i++) {
      TrackingParticlePtr tpPtr(tpHandle, i);
      // Store the TrackingParticle info, using class TP to provide easy access to the most useful info.
      TP tp(&tpHandle->at(i), settings);

      if (tp.physicsCollision()) {
        genPt_ += tp->pt();
      } else {
        genPt_PU_ += tp->pt();
      }

      // Only bother storing tp if it could be useful for tracking efficiency or fake rate measurements.
      // Also create map relating edm::Ptr<TrackingParticle> to TP.
      if (tp.use()) {
        vTPs_.push_back(tp);
        translateTP_[tpPtr] = &vTPs_.back();
      }
    }

    if (settings.debug() > 0) {
      edm::LogInfo("InputData") << "InputData::genPt in the event " << genPt_;
    }

    // Get the tracker geometry info needed to unpack the stub info.
    /*edm::ESHandle<TrackerGeometry> trackerGeometryHandle;
    iSetup.get<TrackerDigiGeometryRecord>().get(trackerGeometryHandle);
    const TrackerGeometry* trackerGeometry = trackerGeometryHandle.product();

    edm::ESHandle<TrackerTopology> trackerTopologyHandle;
    iSetup.get<TrackerTopologyRcd>().get(trackerTopologyHandle);
    const TrackerTopology* trackerTopology = trackerTopologyHandle.product();*/
    //        trackerGeometryToken_(esConsumes<TrackerGeometry, TrackerDigiGeometryRecord>(edm::ESInputTag("",""))),
    //        trackerTopologyToken_(esConsumes<TrackerTopology, TrackerTopologyRcd>(edm::ESInputTag("",""))),

    const auto& trackerGeometry_ = iSetup.getData(trackerGeometryToken_);
    const auto& trackerTopology_ = iSetup.getData(trackerTopologyToken_);
    const TrackerGeometry* trackerGeometry = &trackerGeometry_;
    const TrackerTopology* trackerTopology = &trackerTopology_;

    // Get stub info, by looping over modules and then stubs inside each module.
    // Also get the association map from stubs to tracking particles.
    edm::Handle<DetSetVec> ttStubHandle;
    edm::Handle<TTStubAssMap> mcTruthTTStubHandle;
    edm::Handle<TTClusterAssMap> mcTruthTTClusterHandle;
    iEvent.getByToken(stubToken, ttStubHandle);
    iEvent.getByToken(stubTruthToken, mcTruthTTStubHandle);
    iEvent.getByToken(clusterTruthToken, mcTruthTTClusterHandle);

    std::set<DetId> lStubDetIds;
    for (DetSetVec::const_iterator p_module = ttStubHandle->begin(); p_module != ttStubHandle->end(); p_module++) {
      for (DetSet::const_iterator p_ttstub = p_module->begin(); p_ttstub != p_module->end(); p_ttstub++) {
        lStubDetIds.insert(p_ttstub->getDetId());
      }
    }

    for (auto gd = trackerGeometry->dets().begin(); gd != trackerGeometry->dets().end(); gd++) {
      DetId detid = (*gd)->geographicalId();
      if (detid.subdetId() != StripSubdetector::TOB && detid.subdetId() != StripSubdetector::TID)
        continue;  // only run on OT
      if (!trackerTopology->isLower(detid))
        continue;                                        // loop on the stacks: choose the lower arbitrarily
      DetId stackDetid = trackerTopology->stack(detid);  // Stub module detid

      if (lStubDetIds.count(stackDetid) > 0) {
        assert(stubGeoDetIdMap_.count(stackDetid) == 0);
        stubGeoDetIdMap_[stackDetid] = detid;
      }
    }
    assert(lStubDetIds.size() == stubGeoDetIdMap_.size());

    for (DetSetVec::const_iterator p_module = ttStubHandle->begin(); p_module != ttStubHandle->end(); p_module++) {
      for (DetSet::const_iterator p_ttstub = p_module->begin(); p_ttstub != p_module->end(); p_ttstub++) {
        TTStubRef ttStubRef = edmNew::makeRefTo(ttStubHandle, p_ttstub);
        // Store the Stub info, using class Stub to provide easy access to the most useful info.
        Stub stub(ttStubRef, settings, trackerGeometry, trackerTopology);
        // Also fill truth associating stubs to tracking particles.
        //      stub.fillTruth(vTPs_, mcTruthTTStubHandle, mcTruthTTClusterHandle);
        stub.fillTruth(translateTP_, mcTruthTTStubHandle, mcTruthTTClusterHandle);
        vAllStubs_.push_back(stub);
      }
    }

    std::map<const TP*, std::vector<const Stub*>> tpStubMap;
    for (const TP& tp : vTPs_)
      tpStubMap[&tp] = std::vector<const Stub*>();
    for (const Stub& stub : vAllStubs_) {
      for (const TP* tp : stub.assocTPs()) {
        tpStubMap[tp].push_back(&stub);
      }
    }

    // Find the various vertices
    for (unsigned int j = 0; j < vTPs_.size(); j++) {
      assert(tpStubMap.count(&vTPs_.at(j)) == 1);
      vTPs_[j].setMatchingStubs(tpStubMap.find(&vTPs_.at(j))->second);
      if (vTPs_[j].useForAlgEff()) {
        vertex_.insert(vTPs_[j]);
      } else if (vTPs_[j].useForVertexReco()) {
        bool found = false;
        for (unsigned int i = 0; i < vertices_.size(); ++i) {
          if (vTPs_[j]->vz() == vertices_[i].vz()) {
            vertices_[i].insert(vTPs_[j]);
            found = true;
            break;
          }
        }
        if (!found) {
          Vertex vertex(vTPs_[j]->vz());
          vertex.insert(vTPs_[j]);
          vertices_.push_back(vertex);
        }
      }
    }

    for (const Vertex& vertex : vertices_) {
      if (vertex.numTracks() >= settings.vx_minTracks())
        recoVertices_.push_back(vertex);
    }

    if (settings.debug() > 0)
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

    std::sort(vertices_.begin(), vertices_.end(), SortVertexByZ0());
    std::sort(recoVertices_.begin(), recoVertices_.end(), SortVertexByZ0());

    // Form the HepMC and GenParticle based vertices
    edm::Handle<edm::HepMCProduct> HepMCEvt;
    iEvent.getByToken(hepMCToken, HepMCEvt);

    edm::Handle<edm::View<reco::GenParticle>> GenParticleHandle;
    iEvent.getByToken(genParticlesToken, GenParticleHandle);

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
        if ((genpart.status() != 3) || (genpart.numberOfMothers() == 0))  // not stable or one of the incoming hadrons
          continue;
        genVertex_ = Vertex(genpart.vz());
        break;
      }
    }

  }  // end InputData::InputData

}  // end namespace l1tVertexFinder
