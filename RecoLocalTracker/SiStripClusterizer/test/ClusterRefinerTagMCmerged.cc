#include "RecoLocalTracker/SiStripClusterizer/test/ClusterRefinerTagMCmerged.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "FWCore/Framework/interface/Event.h"

ClusterRefinerTagMCmerged::
ClusterRefinerTagMCmerged(const edm::ParameterSet& conf) 
  : inputTag( conf.getParameter<edm::InputTag>("UntaggedClusterProducer") ),
    confClusterRefiner_(conf.getParameter<edm::ParameterSet>("ClusterRefiner")),
    useAssociateHit_(!confClusterRefiner_.getParameter<bool>("associateRecoTracks")),
    trackerHitAssociatorConfig_(confClusterRefiner_, consumesCollector()) {
  produces< edmNew::DetSetVector<SiStripCluster> > ();
  inputToken = consumes< edmNew::DetSetVector<SiStripCluster> >(inputTag);
}

void ClusterRefinerTagMCmerged::
produce(edm::Event& event, const edm::EventSetup& es)  {

  std::auto_ptr< edmNew::DetSetVector<SiStripCluster> > output(new edmNew::DetSetVector<SiStripCluster>());
  output->reserve(10000,4*10000);

  associator_.reset(new TrackerHitAssociator(event, trackerHitAssociatorConfig_));
  edm::Handle< edmNew::DetSetVector<SiStripCluster> >     input;  

  if ( findInput(inputToken, input, event) ) refineCluster(input, output);
  else edm::LogError("Input Not Found") << "[ClusterRefinerTagMCmerged::produce] ";// << inputTag;

  LogDebug("Output") << output->dataSize() << " clusters from " 
		     << output->size()     << " modules";
  output->shrink_to_fit();
  event.put(output);
}

void  ClusterRefinerTagMCmerged::
refineCluster(const edm::Handle< edmNew::DetSetVector<SiStripCluster> >& input,
	      std::auto_ptr< edmNew::DetSetVector<SiStripCluster> >& output) {

  for (edmNew::DetSetVector<SiStripCluster>::const_iterator det=input->begin(); det!=input->end(); det++) {
    // DetSetVector filler to receive the clusters we produce
    edmNew::DetSetVector<SiStripCluster>::FastFiller outFill(*output, det->id());
    uint32_t detId = det->id();
    int ntk = 0;
    int NtkAll = 0;
    for (edmNew::DetSet<SiStripCluster>::iterator clust = det->begin(); clust != det->end(); clust++) {
      std::vector<uint8_t> amp = clust->amplitudes();
      SiStripCluster* newCluster = new SiStripCluster(clust->firstStrip(), amp.begin(), amp.end());
      if (associator_ != 0) {
        std::vector<SimHitIdpr> simtrackid;
	if (useAssociateHit_) {
	  std::vector<PSimHit> simhit;
	  associator_->associateCluster(clust, DetId(detId), simtrackid, simhit);
	  NtkAll = simtrackid.size();
	  ntk = 0;
	  if (simtrackid.size() > 1) {
	    for (auto const& it : simtrackid) {
	      int NintimeHit = 0;
	      for (auto const& ih : simhit) {
		// std::cout << "  hit(tk, evt) trk(tk, evt) bunch ("
		// 	  << ih.trackId() << ", " << ih.eventId().rawId() << ") ("
		// 	  << it.first << ", " << it.second.rawId() << ") "
		// 	  << ih.eventId().bunchCrossing()
		// 	  << std::endl;
		if (ih.trackId() == it.first && ih.eventId() == it.second && ih.eventId().bunchCrossing() == 0) ++NintimeHit;
	      }
	      if (NintimeHit > 0) ++ntk;
	    }
	  }
	} else {
	  associator_->associateSimpleRecHitCluster(clust, DetId(detId), simtrackid);
	  ntk = NtkAll = simtrackid.size();
	}
	if (ntk > 1) {
	  newCluster->setMerged(true);
	} else {
	  newCluster->setMerged(false);
	}
      }
      outFill.push_back(*newCluster);
      // std::cout << "t_m_w " << " " << NtkAll << " " << newCluster->isMerged()
      // 		<< " " << clust->amplitudes().size()
      // 		<< std::endl;
    } // traverse clusters
  }  // traverse sensors
}

template<class T>
inline
bool ClusterRefinerTagMCmerged::
findInput(const edm::EDGetTokenT<T>& tag, edm::Handle<T>& handle, const edm::Event& e) {
    e.getByToken( tag, handle);
    return handle.isValid();
}
