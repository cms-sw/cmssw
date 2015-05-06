//
// Use MC truth to identify merged strip clusters, i.e., those associated with more
// than one SimTrack, and split them into their true components.
//
// Author:  Bill Ford (wtford)  17 March 2015
//

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/StripDigiSimLink.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#include <memory>

// Class declaration

class ClusterMCsplitStrips : public edm::stream::EDProducer<>  {

public:

  explicit ClusterMCsplitStrips(const edm::ParameterSet& conf);
  virtual void produce(edm::Event&, const edm::EventSetup&);

private:

  template<class T> bool findInput(const edm::EDGetTokenT<T>&, edm::Handle<T>&, const edm::Event&);
  void refineCluster(const edm::Handle< edmNew::DetSetVector<SiStripCluster> >& input,
		     std::auto_ptr< edmNew::DetSetVector<SiStripCluster> >& output);

  const edm::InputTag inputTag;
  typedef edm::EDGetTokenT< edmNew::DetSetVector<SiStripCluster> > token_t;
  token_t inputToken;
  edm::ParameterSet confClusterRefiner_;
  edm::EDGetTokenT< edm::DetSetVector<StripDigiSimLink> > stripdigisimlinkToken;
  edm::Handle< edm::DetSetVector<StripDigiSimLink> > stripdigisimlink_;
  edm::ESHandle<TrackerTopology> tTopoHandle_;

  std::vector<std::string> moduleTypeStrings_;
  std::vector<int> moduleTypeCodes_;

};


// Class implementation

ClusterMCsplitStrips::
ClusterMCsplitStrips(const edm::ParameterSet& conf) 
  : inputTag( conf.getParameter<edm::InputTag>("UnsplitClusterProducer") ),
    confClusterRefiner_(conf.getParameter<edm::ParameterSet>("ClusterRefiner")) {
  produces< edmNew::DetSetVector<SiStripCluster> > ();
  inputToken = consumes< edmNew::DetSetVector<SiStripCluster> >(inputTag);
  stripdigisimlinkToken = consumes< edm::DetSetVector<StripDigiSimLink> >(edm::InputTag("simSiStripDigis"));
  moduleTypeStrings_ = confClusterRefiner_.getParameter<std::vector<std::string> >("moduleTypes");
  for (auto mod = moduleTypeStrings_.begin(); mod != moduleTypeStrings_.end(); ++mod) {
    if (*mod == "IB1") moduleTypeCodes_.push_back(SiStripDetId::IB1);
    if (*mod == "IB2") moduleTypeCodes_.push_back(SiStripDetId::IB2);
    if (*mod == "OB1") moduleTypeCodes_.push_back(SiStripDetId::OB1);
    if (*mod == "OB2") moduleTypeCodes_.push_back(SiStripDetId::OB2);
    if (*mod == "W1A") moduleTypeCodes_.push_back(SiStripDetId::W1A);
    if (*mod == "W2A") moduleTypeCodes_.push_back(SiStripDetId::W2A);
    if (*mod == "W3A") moduleTypeCodes_.push_back(SiStripDetId::W3A);
    if (*mod == "W1B") moduleTypeCodes_.push_back(SiStripDetId::W1B);
    if (*mod == "W2B") moduleTypeCodes_.push_back(SiStripDetId::W2B);
    if (*mod == "W3B") moduleTypeCodes_.push_back(SiStripDetId::W3B);
    if (*mod == "W4") moduleTypeCodes_.push_back(SiStripDetId::W4);
    if (*mod == "W5") moduleTypeCodes_.push_back(SiStripDetId::W5);
    if (*mod == "W6") moduleTypeCodes_.push_back(SiStripDetId::W6);
    if (*mod == "W7") moduleTypeCodes_.push_back(SiStripDetId::W7);
  }
}

void ClusterMCsplitStrips::
produce(edm::Event& event, const edm::EventSetup& evSetup)  {

  //Retrieve tracker topology from geometry
  evSetup.get<TrackerTopologyRcd>().get(tTopoHandle_);

  std::auto_ptr< edmNew::DetSetVector<SiStripCluster> > output(new edmNew::DetSetVector<SiStripCluster>());
  output->reserve(10000,4*10000);

  event.getByToken(stripdigisimlinkToken, stripdigisimlink_);

  edm::Handle< edmNew::DetSetVector<SiStripCluster> >     input;  

  if ( findInput(inputToken, input, event) ) refineCluster(input, output);
  else edm::LogError("Input Not Found") << "[ClusterMCsplitStrips::produce] ";// << inputTag;

  LogDebug("Output") << output->dataSize() << " clusters from " 
		     << output->size()     << " modules";
  output->shrink_to_fit();
  event.put(output);
}

void  ClusterMCsplitStrips::
refineCluster(const edm::Handle< edmNew::DetSetVector<SiStripCluster> >& input,
	      std::auto_ptr< edmNew::DetSetVector<SiStripCluster> >& output) {

  const TrackerTopology* const tTopo = tTopoHandle_.product();

  for (edmNew::DetSetVector<SiStripCluster>::const_iterator det = input->begin(); det != input->end(); det++) {

    uint32_t detID = det->id();
    DetId theDet(detID);
    // int subDet = theDet.subdetId();

    edm::DetSetVector<StripDigiSimLink>::const_iterator isearch = stripdigisimlink_->find(detID);
    if (isearch == stripdigisimlink_->end()) continue;  // This sensor has no simlinks;
                                                       // Any clusters must be from noise.
    // DetSetVector filler to receive the clusters we produce
    edmNew::DetSetVector<SiStripCluster>::FastFiller outFill(*output, det->id());

    // Consider clusters of selected module types for splitting; else just push original cluster to output.
    if (std::find(moduleTypeCodes_.begin(), moduleTypeCodes_.end(), tTopo->moduleGeometry(theDet)) == moduleTypeCodes_.end()) {
      for (auto clust = det->begin(); clust != det->end(); clust++)
	outFill.push_back(*clust);
      continue;  // On to the next sensor.
    }
    // Traverse the clusters for this sensor.
    for (edmNew::DetSet<SiStripCluster>::iterator clust = det->begin(); clust != det->end(); clust++) {
      int clusiz = clust->amplitudes().size();
      int first  = clust->firstStrip();
      int last   = first + clusiz;
      edm::DetSet<StripDigiSimLink> link_detset = (*isearch);

      //  First pass to count simTracks and set this cluster's range in the simlink vector
      edm::DetSet<StripDigiSimLink>::const_iterator
      	firstlink = link_detset.data.begin(),
      	lastlink = link_detset.data.end();
      bool firstlinkInit = false;

      std::vector<unsigned int> trackID;
      for (edm::DetSet<StripDigiSimLink>::const_iterator linkiter = link_detset.data.begin();
	    linkiter != link_detset.data.end(); linkiter++) {
        // DigiSimLinks are ordered first by channel; there can be > 1 track, and > 1 simHit for each track
	int thisChannel = linkiter->channel();
	if (thisChannel < first) continue;
	if (thisChannel >= first && !firstlinkInit) {
	  firstlinkInit = true;
	  firstlink = linkiter;
	}
	if (thisChannel >= last) break;
	lastlink = linkiter;  lastlink++;
	auto const& thisSimTrk = linkiter->SimTrackId();
	if(std::find(trackID.begin(), trackID.end(), thisSimTrk) == trackID.end()) {
	  trackID.push_back(thisSimTrk);
	}
      }  // end initial loop over this sensor's simlinks

      size_t NsimTrk = trackID.size();
      if (NsimTrk < 2) {
	if (NsimTrk == 1) outFill.push_back(*clust);  // Unmerged cluster:  push it to the output.

	//  (If NsimTrk = 0, cluster has no matched simlinks; abandon it.)
	continue;  // On to the next cluster
      }

      // std::cout << "subDet " << DetId(detID).subdetId() << ", det " << detID << ": " << NsimTrk << " simTracks:";
      // for (unsigned int i=0; i<NsimTrk; i++) std::cout << " " << trackID[i];
      // std::cout << std::endl;

      //  This cluster matches more than one simTrack, so we proceed to split it.
      auto const& amp = clust->amplitudes();
      std::vector<int> TKfirstStrip(NsimTrk, -1);
      std::vector< std::vector<uint16_t> > TKampl(NsimTrk);
      std::vector<int> prevStrip(NsimTrk,-1);

      for (edm::DetSet<StripDigiSimLink>::const_iterator linkiter = firstlink; linkiter != lastlink; linkiter++) {
	int stripIdx = (int)linkiter->channel()-first;

	uint16_t rawAmpl = (uint16_t)(amp[stripIdx]);
	uint16_t thisAmpl;
	if (rawAmpl < 254) {
	  thisAmpl = std::min( uint16_t(253), std::max(uint16_t(0), (uint16_t)(rawAmpl*linkiter->fraction()+0.5)) );
	} else {
	  thisAmpl = rawAmpl;
	}

	unsigned int thisSimTrk = linkiter->SimTrackId();
	auto const& TKiter = std::find(trackID.begin(), trackID.end(), thisSimTrk);
	unsigned int TKidx = TKiter - trackID.begin();

	if (TKfirstStrip[TKidx] == -1) TKfirstStrip[TKidx] = linkiter->channel();
	if (stripIdx != prevStrip[TKidx]) {
	  prevStrip[TKidx] = stripIdx;
	  TKampl[TKidx].push_back(thisAmpl);
	} else {
	  if (rawAmpl < 254)
	    (TKampl[TKidx])[linkiter->channel() - TKfirstStrip[TKidx]] += thisAmpl;
	}
      }
      
      for (unsigned int TKidx = 0; TKidx < NsimTrk; ++TKidx) {
	if (std::accumulate(TKampl[TKidx].begin(), TKampl[TKidx].end(), 0) > 0) {

	  // std::cout << "SimTrackID, 1st: ampl " << trackID[TKidx] << ", " << TKfirstStrip[TKidx] << ":";
	  // for (unsigned i=0; i<TKampl[TKidx].size(); ++i) std::cout << " " << TKampl[TKidx][i];
	  // std::cout << std::endl;

	  outFill.push_back( SiStripCluster( (uint16_t)TKfirstStrip[TKidx], TKampl[TKidx].begin(), TKampl[TKidx].end()) );
	}
      }
    }  // end loop over original clusters
      
  }  // end loop over sensors

}

template<class T>
inline
bool ClusterMCsplitStrips::
findInput(const edm::EDGetTokenT<T>& tag, edm::Handle<T>& handle, const edm::Event& e) {
    e.getByToken( tag, handle);
    return handle.isValid();
}

//define this as a plug-in                                                                                                                               
DEFINE_FWK_MODULE(ClusterMCsplitStrips);
