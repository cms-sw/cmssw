#include "nn_treeProducer_raw.h"

nn_tupleProducer_raw::nn_tupleProducer_raw(const edm::ParameterSet& conf):
  magFieldToken_(esConsumes())
  ,propagatorToken_(esConsumes(edm::ESInputTag("", "PropagatorWithMaterialParabolicMf")))
  ,ttbToken_(esConsumes(edm::ESInputTag("", "TransientTrackBuilder")))
  {
  inputTagClusters       = conf.getParameter<edm::InputTag>("siStripClustersTag");
  vertexToken_ = consumes<reco::VertexCollection>(conf.getParameter<edm::InputTag>("vertex"));
  clusterToken           = consumes<edmNew::DetSetVector<SiStripCluster>>(inputTagClusters);
  tracksToken_           = consumes<reco::TrackCollection>(conf.getParameter<edm::InputTag>("tracks"));
  hlttracksToken_           = consumes<reco::TrackCollection>(conf.getParameter<edm::InputTag>("hlttracks"));
  hltPixeltracksToken_           = consumes<reco::TrackCollection>(conf.getParameter<edm::InputTag>("hltPixeltracks"));
  stripCPEToken_         = esConsumes<StripClusterParameterEstimator, TkStripCPERecord>(edm::ESInputTag("", "StripCPEfromTrackAngle"));
  tTopoToken_ = esConsumes<TrackerTopology, TrackerTopologyRcd>();
  tkGeomToken_ = esConsumes();
  usesResource("TFileService");

  beamSpot_ = conf.getParameter<edm::InputTag>("beamSpot");
  beamSpotToken_ = consumes<reco::BeamSpot>(beamSpot_);

  stripNoiseToken_ = esConsumes();

  hist_sig = fs->make<TH2F>("adc_idx_sig","", 50,0,50,260,0,260);
  hist_bkg = fs->make<TH2F>("adc_idx_bkg","", 50,0,50,260,0,260);
  create_tree();
}

nn_tupleProducer_raw::~nn_tupleProducer_raw() = default;

void nn_tupleProducer_raw::analyze(const edm::Event& event, const edm::EventSetup& es) {
  edm::Handle<edmNew::DetSetVector<SiStripCluster>> clusterCollection = event.getHandle(clusterToken);
  const auto& tracksHandle = event.getHandle(tracksToken_);
  const auto& hlttracksHandle = event.getHandle(hlttracksToken_);
  const auto& hltPixeltracksHandle = event.getHandle(hltPixeltracksToken_);
  const auto* stripCPE    = &es.getData(stripCPEToken_);
  const auto* magField    = &es.getData(magFieldToken_);

  const auto& theNoise_ = &es.getData(stripNoiseToken_);

  const Propagator* thePropagator = &es.getData(propagatorToken_);

  theTTrackBuilder = &es.getData(ttbToken_);
  using namespace edm;

  const auto& vertexHandle = event.getHandle(vertexToken_);

  if (!tracksHandle.isValid()) {
    edm::LogError("flatNtuple_producer") << "No valid track collection found";
    return;
  }

  if (!hlttracksHandle.isValid()) {
	      edm::LogError("flatNtuple_producer") << "No valid track collection found";
	          return;
  }
  if (!hltPixeltracksHandle.isValid()) {
	edm::LogError("flatNtuple_producer") << "No valid track collection found";
	return;
  }
  
  if (!vertexHandle.isValid()) {
	      edm::LogError("flatNtuple_producer") << "No valid vertex collection found";
	   return;
  }

  const reco::TrackCollection* tracks = tracksHandle.product();
  const reco::TrackCollection* hlttracks = hlttracksHandle.product();//.product();
  const reco::TrackCollection* hltPixeltracks = hltPixeltracksHandle.product();
  const reco::VertexCollection vertices = *vertexHandle;
  //if ( tracks.size() != 1 || event.id().event() !=33) return;
  //std::cout << "event " << event.id().event() << std::endl;
  std::map<uint32_t, std::vector<cluster_property>> matched_cluster;
  //int trkcluster = 0;
  for(unsigned int i=0; i<tracks->size(); i++) {
     auto& trk = tracks->at(i);
     for (auto ih = trk.recHitsBegin(); ih != trk.recHitsEnd(); ih++) {
         const SiStripCluster* strip=NULL;
         const TrackingRecHit& hit = **ih;
         const DetId detId((hit).geographicalId());
         if (detId.det() == DetId::Tracker) { 
           if (detId.subdetId() == kBPIX || detId.subdetId() == kFPIX) continue;  // pixel is always 2D
           else {        // should be SiStrip now
		   //std::cout << "subdet " << detId.subdetId() << std::endl;
               if (dynamic_cast<const SiStripRecHit1D *>(&hit)) {
		   //std::cout << " found SiStripRecHit1D " << std::endl;
                   strip = dynamic_cast<const SiStripRecHit1D *>(&hit)->cluster().get();
               }
               else if ( dynamic_cast<const SiStripRecHit2D *>(&hit)) {
                 //std::cout << "found SiStripRecHit2D " << std::endl;
                 strip = dynamic_cast<const SiStripRecHit2D *>(&hit)->cluster().get();
               }
               else if (dynamic_cast<const SiStripMatchedRecHit2D *>(&hit)) {
                 // std::cout << "found SiStripMatchedRecHit2D " << std::endl;
                  strip = &(dynamic_cast<const SiStripMatchedRecHit2D *>(&hit))->monoCluster();
              }
           }
           if(strip) {
		   //trkcluster += 1;
	       //std::cout << "strip " << strip << std::endl;
               bool low_pt_trk = trk.pt() < 1.;
               matched_cluster[detId].emplace_back(
                      low_pt_trk, !low_pt_trk, strip->barycenter(),
                      strip->size(), strip->firstStrip(), strip->endStrip(),
                      strip->charge(),
                      trk.algo()
               );
         }
        }
    }
  }    
  const auto& tkGeom = &es.getData(tkGeomToken_);
  const auto tkDets = tkGeom->dets();
  const TrackerTopology& tTopo = es.getData(tTopoToken_);

  //int cluster = 0;
  for (const auto& detSiStripClusters : *clusterCollection) {
    isTIB = isTOB = isTID = isTEC = isStereo = 0;
    eventN = event.id().event();
    //if (eventN != 24061779) continue;
    runN   = (int) event.id().run();
    lumi   = (int) event.id().luminosityBlock();
    detId  = detSiStripClusters.id();
    SiStripNoises::Range detNoiseRange = theNoise_->getRange(detId);
    uint32_t subdet = DetId(detId).subdetId();
    if (subdet == SiStripSubdetector::TIB) isTIB = 1;
    else if (subdet == SiStripSubdetector::TOB) isTOB = 1;
    else if (subdet == SiStripSubdetector::TID) isTID = 1;
    else if (subdet == SiStripSubdetector::TEC) isTEC = 1;
    layer = tTopo.layer(DetId(detId));
    isStereo = tTopo.isStereo(DetId(detId));
    isglued = tTopo.glued(DetId(detId));
    isstacked = tTopo.stack(DetId(detId));
    const auto& _detId = detId; // for the capture clause in the lambda function
    auto det = std::find_if(tkDets.begin(), tkDets.end(), [_detId](auto& elem) -> bool {
        return (elem->geographicalId().rawId() == _detId);
    });
    const StripTopology& p = dynamic_cast<const StripGeomDetUnit*>(*det)->specificTopology();
    std::vector<cluster_property> track_clusters = {};
    if ( matched_cluster.find(detId) != matched_cluster.end() ) track_clusters = matched_cluster[detId];

    std::map<reco::TrackRef, TrajectoryStateOnSurface> recotrk_tsosCache;
    std::map<reco::TrackRef, TrajectoryStateOnSurface> hlttrk_tsosCache;
    std::map<reco::TrackRef, TrajectoryStateOnSurface> pixeltrk_tsosCache;

    const GeomDetUnit* geomDet = tkGeom->idToDetUnit(detId);
    const StripGeomDetUnit* stripDet = dynamic_cast<const StripGeomDetUnit*>(geomDet);
    for (const auto& stripCluster : detSiStripClusters) {
    //  cluster += 1;
      initialize_vars();
      firstStrip = stripCluster.firstStrip();
      endStrip   = stripCluster.endStrip();
      barycenter = stripCluster.barycenter();
      size       = stripCluster.size();
      charge     = stripCluster.charge();

      std::vector<int>adcs;
      std::vector<float>noises;
      
      for (int strip = firstStrip; strip < endStrip; ++strip)
      {
        GlobalPoint gp = (tkGeom->idToDet(detId))->surface().toGlobal(p.localPosition((float) strip));

        hitX   [strip - firstStrip] = gp.x();
        hitY   [strip - firstStrip] = gp.y();
        hitZ   [strip - firstStrip] = gp.z();
        channel[strip - firstStrip] = strip;
        adc    [strip - firstStrip] = stripCluster[strip - firstStrip];
	adcs.push_back(stripCluster[strip - firstStrip]);
	noises.push_back(theNoise_->getNoise(strip, detNoiseRange));
	if (adc    [strip - firstStrip] >= 254) n_saturated += 1;
        if (adc [strip - firstStrip] > max_adc) {
                max_adc = adc[strip - firstStrip];
                max_adc_idx = strip - firstStrip;
		max_adc_x   = hitX[strip - firstStrip];
		max_adc_y   = hitY[strip - firstStrip];
		max_adc_z   = hitZ[strip - firstStrip];
        }	
      }
      double mean = std::accumulate(adc, adc+size,0.0) / size;
      adc_std = std::sqrt(std::accumulate(adc, adc+size,0.0, [mean](double acc, double x) {
			      return acc + (x - mean) * (x - mean);
			      }) / size );
      noise_max_adc = adc[max_adc_idx] - noises[max_adc_idx];
      if (max_adc_idx >=1) {
	      diff_adc_mone = adc[max_adc_idx] - adc[max_adc_idx-1];
	      noise_diff_adc_mone = abs(adc[max_adc_idx-1] - noises[max_adc_idx-1]);
	      noise_adc_mone = noises[max_adc_idx-1];
	      adc_mone = adc[max_adc_idx-1];
      }
      if (max_adc_idx >=2) {
	      diff_adc_mtwo = adc[max_adc_idx] - adc[max_adc_idx-2];
	      noise_diff_adc_mtwo = abs(adc[max_adc_idx-2] - noises[max_adc_idx-2]);
              noise_adc_mtwo = noises[max_adc_idx-2];
	      adc_mtwo = adc[max_adc_idx-2];
      }
      if (max_adc_idx >=3) {
              diff_adc_mthree = adc[max_adc_idx] - adc[max_adc_idx-3];
	      noise_diff_adc_mthree = abs(adc[max_adc_idx-3] - noises[max_adc_idx-3]);
	      noise_adc_mthree = noises[max_adc_idx-3];
              adc_mthree = adc[max_adc_idx-3];
      }
      if (((size-1)-max_adc_idx) >=1) {
	      diff_adc_pone = adc[max_adc_idx] - adc[max_adc_idx+1];
	      noise_diff_adc_pone = abs(adc[max_adc_idx+1] - noises[max_adc_idx+1]);
	      noise_adc_pone = noises[max_adc_idx+1];
	      adc_pone = adc[max_adc_idx+1];
      }
      if (((size-1)-max_adc_idx) >=2) {
	      diff_adc_ptwo = adc[max_adc_idx] - adc[max_adc_idx+2];
	      noise_diff_adc_ptwo = abs(adc[max_adc_idx+2] - noises[max_adc_idx+2]);
	      noise_adc_ptwo = noises[max_adc_idx+2];
	      adc_ptwo = adc[max_adc_idx+2];
      }
      if (((size-1)-max_adc_idx) >=3) {
              diff_adc_pthree = adc[max_adc_idx] - adc[max_adc_idx+3];
	      noise_diff_adc_pthree = abs(adc[max_adc_idx+3] - noises[max_adc_idx+3]);
	      noise_adc_pthree = noises[max_adc_idx+3];
              adc_pthree = adc[max_adc_idx+3];
      }
      
      for(auto& trk_cluster_property: track_clusters)
        {
           if (trk_cluster_property.barycenter == barycenter)
           {
               assert( (size == trk_cluster_property.size)
                      && (firstStrip == trk_cluster_property.firstStrip)
                      && (endStrip == trk_cluster_property.endStrip)
                      && (charge == trk_cluster_property.charge)
               );
               low_pt_trk_cluster = trk_cluster_property.low_pt_trk_cluster;
               high_pt_trk_cluster = trk_cluster_property.high_pt_trk_cluster;
               trk_algo           = trk_cluster_property.trk_algo;
	       target = 1;
	       break;
           }
        }

      pixeltrk_dr_min = 99.;
      
      const reco::Track* recotrk = NULL;
      hlttrk_dr_min = 99;
      recotrk_dr_min = 99;
      const reco::Track* pixeltrk = NULL;
      for ( size_t i=0; i<hltPixeltracks->size(); i++) {
	    reco::TrackRef trackRef(hltPixeltracks, i);
            auto it = pixeltrk_tsosCache.find(trackRef);
	    TrajectoryStateOnSurface tsos;
	    if (it != pixeltrk_tsosCache.end()) tsos = it->second;
	    else {
              reco::TransientTrack tkTT = theTTrackBuilder->build(*trackRef);
	      if(!tkTT.impactPointState().isValid()) continue;
              tsos = thePropagator->propagate(tkTT.impactPointState(), geomDet->surface());
	      pixeltrk_tsosCache[trackRef] = tsos;
	    }
            if (!tsos.isValid()) continue;
            auto localValues = stripCPE->localParameters(stripCluster, *stripDet, tsos);
            LocalPoint clusterLocal = localValues.first;
            LocalPoint trackLocal = geomDet->surface().toLocal(tsos.globalPosition());
            float dr = abs(trackLocal.x() - clusterLocal.x());
            if (dr < pixeltrk_dr_min) {
                  pixeltrk_dr_min = dr;
		  pixeltrk = &(*trackRef);
             }
      }

      if (pixeltrk) {
        pixeltrk_pt = pixeltrk->pt();
        pixeltrk_pterr = pixeltrk->ptError();
        pixeltrk_eta = pixeltrk->eta();
        pixeltrk_phi = pixeltrk->phi();
        pixeltrk_dz = pixeltrk->dz(vertices.at(0).position());
        pixeltrk_dxy = pixeltrk->dxy(vertices.at(0).position());
        pixeltrk_validhits = pixeltrk->numberOfValidHits();
        pixeltrk_chi2 = pixeltrk->normalizedChi2();
        pixeltrk_d0sigma = sqrt(pixeltrk->d0Error() * pixeltrk->d0Error() + vertices.at(0).xError() * vertices.at(0).yError());
        pixeltrk_dzsigma = sqrt(pixeltrk->dzError() * pixeltrk->dzError() + vertices.at(0).zError() * vertices.at(0).zError());
        pixeltrk_qoverp = pixeltrk->qoverp();
        pixeltrk_qoverperror = pixeltrk->qoverpError();
      }
     
      for ( size_t i=0; i<tracks->size(); i++) {
          reco::TrackRef trackRef(tracks, i);
	  auto it = recotrk_tsosCache.find(trackRef);
	  TrajectoryStateOnSurface tsos;
	   if (it != recotrk_tsosCache.end()) tsos = it->second;
	    else {
	       for (auto const& hit : trackRef->recHits()) {
	         if (!hit->isValid()) continue;
                 if (hit->geographicalId() != detId) continue;
		 reco::TransientTrack tkTT = theTTrackBuilder->build(*trackRef);
	         tsos = thePropagator->propagate(tkTT.impactPointState(), geomDet->surface());							                  
		 recotrk_tsosCache[trackRef] = tsos;
	      }
              if (!tsos.isValid()) continue;
              auto localValues = stripCPE->localParameters(stripCluster, *stripDet, tsos);
              LocalPoint clusterLocal = localValues.first;

              LocalPoint trackLocal = geomDet->surface().toLocal(tsos.globalPosition());
              float dr = abs(trackLocal.x() - clusterLocal.x()); //std::sqrt( pow( (trackLocal.x() - clusterLocal.x()), 2 ) +
              if (dr < recotrk_dr_min) {
                recotrk_dr_min = dr;
		recotrk = &(*trackRef);
                }
              }
      }

      if (recotrk) {
        recotrk_pt = recotrk->pt();
        recotrk_pterr = recotrk->ptError();
        recotrk_eta = recotrk->eta();
        recotrk_phi = recotrk->phi();
        recotrk_dz = recotrk->dz(vertices.at(0).position());
        recotrk_dxy = recotrk->dxy(vertices.at(0).position());
        recotrk_validhits = recotrk->numberOfValidHits();
        recotrk_chi2 = recotrk->normalizedChi2();
        recotrk_d0sigma = sqrt(recotrk->d0Error() * recotrk->d0Error() + vertices.at(0).xError() * vertices.at(0).yError());
        recotrk_dzsigma = sqrt(recotrk->dzError() * recotrk->dzError() + vertices.at(0).zError() * vertices.at(0).zError());
        recotrk_qoverp = recotrk->qoverp();
        recotrk_qoverperror = recotrk->qoverpError();
      }
      
      //regression->Fill();
      //continue;
      const reco::Track* hlttrk = NULL;
      for ( size_t i=0; i<hlttracks->size(); i++) {
	  reco::TrackRef trackRef(hlttracks, i);
          auto it = hlttrk_tsosCache.find(trackRef);
	  TrajectoryStateOnSurface tsos;
	  if (it != hlttrk_tsosCache.end()) tsos = it->second;
          else {
            for (auto const& hit : trackRef->recHits()) {
	       if (!hit->isValid()) continue;
	       if (hit->geographicalId() != detId) continue;
		reco::TransientTrack tkTT = theTTrackBuilder->build(*trackRef);
		tsos = thePropagator->propagate(tkTT.impactPointState(), geomDet->surface());
		hlttrk_tsosCache[trackRef] = tsos;
	     }
             if (!tsos.isValid()) continue;
             auto localValues = stripCPE->localParameters(stripCluster, *stripDet, tsos);
             LocalPoint clusterLocal = localValues.first;

             LocalPoint trackLocal = geomDet->surface().toLocal(tsos.globalPosition());
             float dr = abs(trackLocal.x() - clusterLocal.x()); //std::sqrt( pow( (trackLocal.x() - clusterLocal.x()), 2 ) +
             if (dr < hlttrk_dr_min) {
		vrtx_xy = trackRef->dxy(vertices.at(0).position());
		vrtx_z = trackRef->dz(vertices.at(0).position());
                hlttrk_dr_min = dr;
		hlttrk = &(*trackRef);
                }
	      }
      }

      if (hlttrk) {
        hlttrk_pt = hlttrk->pt();
        hlttrk_pterr = hlttrk->ptError();
        hlttrk_eta = hlttrk->eta();
        hlttrk_phi = hlttrk->phi();
        hlttrk_dz = hlttrk->dz(vertices.at(0).position());
        hlttrk_dxy = hlttrk->dxy(vertices.at(0).position());
        hlttrk_validhits = hlttrk->numberOfValidHits();
        hlttrk_chi2 = hlttrk->normalizedChi2();
        hlttrk_d0sigma = sqrt(hlttrk->d0Error() * hlttrk->d0Error() + vertices.at(0).xError() * vertices.at(0).yError());
        hlttrk_dzsigma = sqrt(hlttrk->dzError() * hlttrk->dzError() + vertices.at(0).zError() * vertices.at(0).zError());
        hlttrk_qoverp = hlttrk->qoverp();
        hlttrk_qoverperror = hlttrk->qoverpError();
      }

      tree->Fill();
      if (target) {
	      for (unsigned int i = 0; i < adcs.size()-1; ++i) hist_sig->Fill(i, adcs[i]); // Fill 2D histogram
      }
      else {
	      for (unsigned int i = 0; i < adcs.size()-1; ++i) hist_bkg->Fill(i, adcs[i]); // Fill 2D histogram
      }
    }
  }
}

void nn_tupleProducer_raw::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("siStripClustersTag", edm::InputTag("siStripClusters"));
  desc.add<edm::InputTag>("tracks", edm::InputTag("generalTracks","","reRECO"));
  desc.add<edm::InputTag>("hlttracks", edm::InputTag("hltTracks","","HLTX"));
  desc.add<edm::InputTag>("hltPixeltracks", edm::InputTag("hltPixelTracks","","HLTX"));
  desc.add<edm::InputTag>("beamSpot", edm::InputTag("offlineBeamSpot"));
  desc.add<edm::InputTag>("vertex", edm::InputTag("vertex"));
  descriptions.add("nn_tupleProducer_raw", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(nn_tupleProducer_raw);
