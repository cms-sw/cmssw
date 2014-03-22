#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Provenance/interface/ProductID.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackerRecHit2D/interface/ClusterRemovalInfo.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
//
// class decleration
//

class TopBottomClusterInfoProducer : public edm::EDProducer {
public:
  TopBottomClusterInfoProducer(const edm::ParameterSet& iConfig) ;
  ~TopBottomClusterInfoProducer() ;
  void produce(edm::Event &iEvent, const edm::EventSetup &iSetup) override ;
  
private:
  edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster> > pixelClustersOld_;
  edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster> > pixelClustersNew_;
  edm::EDGetTokenT<edmNew::DetSetVector<SiStripCluster> > stripClustersOld_;
  edm::EDGetTokenT<edmNew::DetSetVector<SiStripCluster> > stripClustersNew_;
};


using namespace std;
using namespace edm;
using namespace reco;

TopBottomClusterInfoProducer::TopBottomClusterInfoProducer(const ParameterSet& iConfig)
{
  pixelClustersOld_ = consumes<edmNew::DetSetVector<SiPixelCluster> >(iConfig.getParameter<edm::InputTag>("stripClustersOld"));
  stripClustersOld_ = consumes<edmNew::DetSetVector<SiStripCluster> >(iConfig.getParameter<edm::InputTag>("pixelClustersOld"));
  pixelClustersNew_ = consumes<edmNew::DetSetVector<SiPixelCluster> >(iConfig.getParameter<edm::InputTag>("stripClustersNew"));
  stripClustersNew_ = consumes<edmNew::DetSetVector<SiStripCluster> >(iConfig.getParameter<edm::InputTag>("pixelClustersNew"));
  produces< ClusterRemovalInfo >();
}


TopBottomClusterInfoProducer::~TopBottomClusterInfoProducer()
{
}

void
TopBottomClusterInfoProducer::produce(Event& iEvent, const EventSetup& iSetup)
{

    Handle<edmNew::DetSetVector<SiPixelCluster> > pixelClustersOld;
    iEvent.getByToken(pixelClustersOld_, pixelClustersOld);
    Handle<edmNew::DetSetVector<SiStripCluster> > stripClustersOld;
    iEvent.getByToken(stripClustersOld_, stripClustersOld);

    Handle<edmNew::DetSetVector<SiPixelCluster> > pixelClustersNew;
    iEvent.getByToken(pixelClustersNew_, pixelClustersNew);
    Handle<edmNew::DetSetVector<SiStripCluster> > stripClustersNew;
    iEvent.getByToken(stripClustersNew_, stripClustersNew);

    auto_ptr<ClusterRemovalInfo> cri(new ClusterRemovalInfo(pixelClustersOld, stripClustersOld));
    ClusterRemovalInfo::Indices& pixelInd = cri->pixelIndices();
    ClusterRemovalInfo::Indices& stripInd = cri->stripIndices();
    stripInd.reserve(stripClustersNew->size()); 
    pixelInd.reserve(pixelClustersNew->size()); 

    //const SiStripCluster * firstOffsetStripNew = & stripClustersNew->data().front();
    for (edmNew::DetSetVector<SiStripCluster>::const_iterator itdetNew = stripClustersNew->begin(); itdetNew != stripClustersNew->end(); ++itdetNew) {
      edmNew::DetSet<SiStripCluster> oldDSstripNew = *itdetNew;
      if (oldDSstripNew.empty()) continue; // skip empty detsets 
      for (edmNew::DetSet<SiStripCluster>::const_iterator clNew = oldDSstripNew.begin(); clNew != oldDSstripNew.end(); ++clNew) {
	uint16_t firstStripNew = clNew->firstStrip();
	uint32_t idStripNew = itdetNew->id();
	//uint32_t keyNew = ((&*clNew) - firstOffsetStripNew);
	//cout << "new strip index=" << keyNew << endl;
	uint32_t keyOld=99999;
	
	const SiStripCluster * firstOffsetStripOld = & stripClustersOld->data().front();
        edmNew::DetSetVector<SiStripCluster>::const_iterator itdetOld = stripClustersOld->find(itdetNew->id());
        if (itdetOld != stripClustersOld->end()) {
	  edmNew::DetSet<SiStripCluster> oldDSstripOld = *itdetOld;
	  if (oldDSstripOld.empty()) continue; // skip empty detsets 
	  for (edmNew::DetSet<SiStripCluster>::const_iterator clOld = oldDSstripOld.begin(); clOld != oldDSstripOld.end(); ++clOld) {
	    uint16_t firstStripOld = clOld->firstStrip();
	    uint32_t idStripOld = itdetOld->id();
	    if (idStripNew==idStripOld && firstStripNew==firstStripOld) {
	      keyOld = ((&*clOld) - firstOffsetStripOld);
	      //cout << "old strip index=" << keyOld << endl;
	      break;
	    }
	  }
	}
	//assert(keyOld!=99999);
	//cout << "push back strip index=" << keyOld << endl;
	stripInd.push_back(keyOld);
      }	 
    }


    //const SiPixelCluster * firstOffsetPixelNew = & pixelClustersNew->data().front();
    for (edmNew::DetSetVector<SiPixelCluster>::const_iterator itdetNew = pixelClustersNew->begin(); itdetNew != pixelClustersNew->end(); ++itdetNew) {
      edmNew::DetSet<SiPixelCluster> oldDSpixelNew = *itdetNew;
      if (oldDSpixelNew.empty()) continue; // skip empty detsets 
      for (edmNew::DetSet<SiPixelCluster>::const_iterator clNew = oldDSpixelNew.begin(); clNew != oldDSpixelNew.end(); ++clNew) {
	int minPixelRowNew = clNew->minPixelRow();
	//uint32_t keyNew = ((&*clNew) - firstOffsetPixelNew);
	//cout << "new pixel index=" << keyNew << endl;
	uint32_t keyOld=99999;
	
	const SiPixelCluster * firstOffsetPixelOld = & pixelClustersOld->data().front();
        edmNew::DetSetVector<SiPixelCluster>::const_iterator itdetOld = pixelClustersOld->find(oldDSpixelNew.detId());
        if (itdetOld != pixelClustersOld->end()) {
	  edmNew::DetSet<SiPixelCluster> oldDSpixelOld = *itdetOld;
	  if (oldDSpixelOld.empty()) continue; // skip empty detsets 
	  for (edmNew::DetSet<SiPixelCluster>::const_iterator clOld = oldDSpixelOld.begin(); clOld != oldDSpixelOld.end(); ++clOld) {
	    int minPixelRowOld = clOld->minPixelRow();
	    if (minPixelRowNew==minPixelRowOld) {
	      keyOld = ((&*clOld) - firstOffsetPixelOld);
	      //cout << "old pixel index=" << keyOld << endl;
	      break;
	    }
	  }
	}
	assert(keyOld!=99999);
	//cout << "push back pixel index=" << keyOld << endl;
	pixelInd.push_back(keyOld);
      }	 
    }
    
    //cout << "pixelInd size" << pixelInd.size() << endl; 
    //cout << "stripInd size" << stripInd.size() << endl; 

    cri->setNewPixelClusters(edm::OrphanHandle<SiPixelClusterCollectionNew>(pixelClustersNew.product(),pixelClustersNew.id()));
    cri->setNewStripClusters(edm::OrphanHandle<edmNew::DetSetVector<SiStripCluster> >(stripClustersNew.product(),stripClustersNew.id()));

    iEvent.put(cri);
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TopBottomClusterInfoProducer);
