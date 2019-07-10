// -*- C++ -*-
//
// Package:    RecoBTag/PixelCluster
// Class:      PixelClusterTagInfoProducer
// 
/**\class PixelClusterTagInfoProducer PixelCluster RecoBTag/PixelCluster/plugins/PixelClusterTagInfoProducer.cc

 Description: Produces a vector of PixelClusterTagInfo objects.

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Alberto Zucchetta (UniZ) [zucchett]
//         Created:  Wed, 03 Jul 2019 12:37:30 GMT
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

// TagInfo
#include "DataFormats/BTauReco/interface/PixelClusterTagInfo.h"

// For vertices
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

// For jet
#include "DataFormats/JetReco/interface/PFJet.h" 
#include "DataFormats/JetReco/interface/PFJetCollection.h"

// For pixel clusters
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

// Pixel topology
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"

// Geometry
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"



//
// class declaration
//

class PixelClusterTagInfoProducer : public edm::stream::EDProducer<> {
   public:
      explicit PixelClusterTagInfoProducer(const edm::ParameterSet&);
      ~PixelClusterTagInfoProducer();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:

      virtual void beginStream(edm::StreamID) override;
      virtual void produce(edm::Event&, const edm::EventSetup&) override;
      virtual void endStream() override;

      //virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
      //virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
      //virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
      //virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

      // ----------member data ---------------------------
      edm::ParameterSet iConfig;
      
      edm::EDGetTokenT< edm::View<reco::Jet> >                 m_jetsAK4;
      edm::EDGetTokenT< edm::View<reco::Jet> >                 m_jetsAK8;
      edm::EDGetTokenT< reco::VertexCollection >               m_vertices;
      edm::EDGetTokenT< edmNew::DetSetVector<SiPixelCluster> > m_pixelhit;
      unsigned int m_nLayers;
      unsigned int m_minADC;
      double m_minJetPt;
      bool isPhase1;
};

//
// constants, enums and typedefs
//


//
// static data member definitions
//

//
// constructors and destructor
//
PixelClusterTagInfoProducer::PixelClusterTagInfoProducer(const edm::ParameterSet& iConfig):
    m_jetsAK4  ( consumes<edm::View<reco::Jet>                >  (iConfig.getParameter<edm::InputTag>("jetsAK4")) ),
    m_jetsAK8  ( consumes<edm::View<reco::Jet>                >  (iConfig.getParameter<edm::InputTag>("jetsAK8")) ),
    m_vertices ( consumes<reco::VertexCollection              >  (iConfig.getParameter<edm::InputTag>("primaryVertex")) ),
    m_pixelhit ( consumes<edmNew::DetSetVector<SiPixelCluster> > (iConfig.getParameter<edm::InputTag>("pixels")) ),
    m_nLayers  ( iConfig.getParameter<unsigned int>("pixelLayers") ),
    m_minADC   ( iConfig.getParameter<unsigned int>("minAdcCount") ),
    m_minJetPt ( iConfig.getParameter<double>("minJetPtCut") )
{
   //register your products
/* Examples
   produces<ExampleData2>();

   //if do put with a label
   produces<ExampleData2>("label");
 
   //if you want to put into the Run
   produces<ExampleData2,InRun>();
*/
  produces<reco::PixelClusterTagInfoCollection>();
  
  isPhase1 = (m_nLayers == 4);
}


PixelClusterTagInfoProducer::~PixelClusterTagInfoProducer()
{
 
   // do anything here that needs to be done at destruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
PixelClusterTagInfoProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    //  using namespace edm;
    // This is an event example
    //Read 'ExampleData' from the Event
    //edm::Handle<reco::PixelClusterTagInfo> tagInfoCollection;
    // Declare produced collection
    auto pixelTagInfo = std::make_unique<reco::PixelClusterTagInfoCollection>();

  /* this is an EventSetup example
     //Read SetupData from the SetupRecord in the EventSetup
     ESHandle<SetupData> pSetup;
     iSetup.get<SetupRecord>().get(pSetup);
  */
    // Get handles
    
    edm::Handle<edm::View<reco::Jet> > collectionAK4;
    iEvent.getByToken( m_jetsAK4, collectionAK4);
    
    edm::Handle<edm::View<reco::Jet> > collectionAK8;
    iEvent.getByToken( m_jetsAK8, collectionAK8);

    edm::Handle<reco::VertexCollection> collectionPVs;
    iEvent.getByToken( m_vertices, collectionPVs);
    
    int nJets(0);
    for(auto jetIt = collectionAK4->begin(); jetIt != collectionAK4->end(); ++jetIt) {
        if(jetIt->pt() > m_minJetPt) nJets++;
    }
    if(nJets <= 0) {
        iEvent.put(std::move(pixelTagInfo));
        return;
    }
    
    edm::Handle<edmNew::DetSetVector<SiPixelCluster> > collectionClusters;
    iEvent.getByToken( m_pixelhit, collectionClusters);

    edm::ESHandle<TrackerGeometry> geom;
    iSetup.get<TrackerDigiGeometryRecord>().get( geom );
    const TrackerGeometry& theTracker(*geom);

    // Retrieve tracker topology from geometry
    edm::ESHandle<TrackerTopology> tTopoH;
    iSetup.get<TrackerTopologyRcd>().get(tTopoH);
    const TrackerTopology* tTopo = tTopoH.product();

    std::vector<reco::Jet> jets;
    std::vector<reco::PixelClusterProperties> clusters;
    
    std::cout << std::endl << std::endl << collectionClusters->size() << std::endl;
    
    // Get vector of detunit ids and loop
    for(edmNew::DetSetVector<SiPixelCluster>::const_iterator detUnit = collectionClusters->begin(); detUnit != collectionClusters->end(); ++detUnit) {
        if(detUnit->size() <= 0) continue;
        unsigned int detid = detUnit->detId();
        DetId detId = DetId(detid);            // Get the Detid object
        unsigned int detType = detId.det();    // det type, pixel=1
        if(detType != 1) continue;               // look only at pixels
        unsigned int subid = detId.subdetId(); //subdetector type, pix barrel=1, forward=2
        // Subdet id, pix barrel=1, forward=2
        
        // Get the geom-detector
        const PixelGeomDetUnit* theGeomDet = dynamic_cast<const PixelGeomDetUnit*>(theTracker.idToDet(detId));
        const PixelTopology* topol = &(theGeomDet->specificTopology());
        double detX = theGeomDet->surface().position().x();
        double detY = theGeomDet->surface().position().y();
        double detZ = theGeomDet->surface().position().z();
        int layer = 0; // 1-4
        
        if(subid==1) {  // barrel
            PixelBarrelName pbn(detid, tTopo, isPhase1);
            layer = pbn.layerName();
        }
//        else if(subid==2) {  // forward
//            PixelEndcapName pen(detid, tTopo, isPhase1);
//            layer = pen.bladeName();
//        }
        if(layer == 0 || layer > 4) continue;
//        std::cout << layer << "\t" << detX << " : " << detUnit->x() << "\t" << detY << " : " << detUnit->y() << "\t" << detZ << " : " << detUnit->z() << std::endl;
        
        for(edmNew::DetSet<SiPixelCluster>::const_iterator clustIt = detUnit->begin(); clustIt != detUnit->end(); ++clustIt) {
            // get global position of the cluster
            LocalPoint lp = topol->localPosition(MeasurementPoint(clustIt->x(), clustIt->y()));
            GlobalPoint clustgp = theGeomDet->surface().toGlobal( lp );
            reco::PixelClusterProperties cp = { clustgp.x(), clustgp.y(), clustgp.z(), clustIt->charge() };
//            clusterX.push_back(clustgp.x());
//            clusterY.push_back(clustgp.y());
//            clusterZ.push_back(clustgp.z());
//            clusterC.push_back(clustIt->charge());
            clusters.push_back(cp);
        }
//    if( numberOfClusters < 1.) continue;
//    
//    detUnit_layer .push_back( layer ); 
//    detUnit_X         .push_back(detX     );
//    detUnit_Y         .push_back(detY     );
//    detUnit_Z         .push_back(detZ     );
//    
//    nClusters.push_back(numberOfClusters);
//    cluster_globalz  .push_back(_cluster_globalz);
//    cluster_globalx  .push_back(_cluster_globalx);
//    cluster_globaly  .push_back(_cluster_globaly);
//    cluster_charge   .push_back(_cluster_charge);
    
    }
//    std::cout << clusterC.size() << std::endl;
   
    // Loop over jets
    for(unsigned int i=0, nj=collectionAK4->size(); i<nj; i++) {
        if(collectionAK4->at(i).pt() < m_minJetPt) continue;

        edm::RefToBase<reco::Jet> jetRef = collectionAK4->refAt(i);
        //std::cout << jetRef->pt() << std::endl;
        
        reco::PixelClusterTagInfo tagInfo;
        tagInfo.setJetRef(jetRef);
        
        

        pixelTagInfo->push_back(tagInfo);
    }
    
    // Put the TagInfo collection in the event
    iEvent.put(std::move(pixelTagInfo));
}

// ------------ method called once each stream before processing any runs, lumis or events  ------------
void
PixelClusterTagInfoProducer::beginStream(edm::StreamID)
{
}

// ------------ method called once each stream after processing all runs, lumis and events  ------------
void
PixelClusterTagInfoProducer::endStream() {
}

// ------------ method called when starting to processes a run  ------------
/*
void
PixelClusterTagInfoProducer::beginRun(edm::Run const&, edm::EventSetup const&)
{
}
*/
 
// ------------ method called when ending the processing of a run  ------------
/*
void
PixelClusterTagInfoProducer::endRun(edm::Run const&, edm::EventSetup const&)
{
}
*/
 
// ------------ method called when starting to processes a luminosity block  ------------
/*
void
PixelClusterTagInfoProducer::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/
 
// ------------ method called when ending the processing of a luminosity block  ------------
/*
void
PixelClusterTagInfoProducer::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/
 
// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
PixelClusterTagInfoProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(PixelClusterTagInfoProducer);
