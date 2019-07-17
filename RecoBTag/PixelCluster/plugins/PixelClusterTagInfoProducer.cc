
#include "RecoBTag/PixelCluster/interface/PixelClusterTagInfoProducer.h"


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
    m_jets     ( consumes<edm::View<reco::Jet> >                 (iConfig.getParameter<edm::InputTag>("jets")) ),
    m_vertices ( consumes<reco::VertexCollection >               (iConfig.getParameter<edm::InputTag>("vertices")) ),
    m_pixelhit ( consumes<edmNew::DetSetVector<SiPixelCluster> > (iConfig.getParameter<edm::InputTag>("pixelhit")) ),
    m_isPhase1 ( iConfig.getParameter<bool>("isPhase1") ),
    m_addFPIX  ( iConfig.getParameter<bool>("addForward") ),
    m_minADC   ( iConfig.getParameter<int>("minAdcCount") ),
    m_minJetPt ( iConfig.getParameter<double>("minJetPtCut") ),
    m_maxJetEta( iConfig.getParameter<double>("maxJetEtaCut") )
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
  
  m_nLayers = (m_isPhase1 ? 4 : 3);
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
    
    edm::Handle<edm::View<reco::Jet> > collectionJets;
    iEvent.getByToken(m_jets, collectionJets);

    int nJets(0);
    for(auto jetIt = collectionJets->begin(); jetIt != collectionJets->end(); ++jetIt) {
        if(jetIt->pt() > m_minJetPt) nJets++;
    }
    
    edm::Handle<reco::VertexCollection> collectionPVs;
    iEvent.getByToken(m_vertices, collectionPVs);
    reco::VertexCollection::const_iterator firstPV = collectionPVs->begin();

    for(reco::VertexCollection::const_iterator vtxIt = collectionPVs->begin(); vtxIt != collectionPVs->end(); ++vtxIt) {
        firstPV = vtxIt;
        break;
    }

    // If no suitable Jet and PV is available, skip the event without opening pixel collection
    if(collectionPVs->size() <= 0 || nJets <= 0) {
        iEvent.put(std::move(pixelTagInfo));
        return;
    }
    
    edm::Handle<edmNew::DetSetVector<SiPixelCluster> > collectionClusters;
    iEvent.getByToken(m_pixelhit, collectionClusters);

    edm::ESHandle<TrackerGeometry> geom;
    iSetup.get<TrackerDigiGeometryRecord>().get( geom );
    const TrackerGeometry& theTracker(*geom);

    // Retrieve tracker topology from geometry
    edm::ESHandle<TrackerTopology> tTopoH;
    iSetup.get<TrackerTopologyRcd>().get(tTopoH);
    const TrackerTopology* tTopo = tTopoH.product();

    std::vector<reco::PixelClusterProperties> clusters;
    
    std::cout << std::endl << "Event " << iEvent.eventAuxiliary().event() << std::endl;
//    std::cout << std::endl << std::endl << collectionClusters->size() << std::endl;
    
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
//        double detX = theGeomDet->surface().position().x();
//        double detY = theGeomDet->surface().position().y();
//        double detZ = theGeomDet->surface().position().z();
        int layer = 0; // 1-4
        
        if(subid==1) {  // barrel
            PixelBarrelName pbn(detid, tTopo, m_isPhase1);
            layer = pbn.layerName();
        }
        else if(m_addFPIX && subid==2) {  // forward
            PixelEndcapName pen(detid, tTopo, m_isPhase1);
            layer = pen.diskName();
        }
        if(layer == 0 || layer > m_nLayers) continue;
//        std::cout << layer << "\t" << detX << " : " << detUnit->x() << "\t" << detY << " : " << detUnit->y() << "\t" << detZ << " : " << detUnit->z() << std::endl;
        
        for(edmNew::DetSet<SiPixelCluster>::const_iterator clustIt = detUnit->begin(); clustIt != detUnit->end(); ++clustIt) {
            // get global position of the cluster
            LocalPoint lp = topol->localPosition(MeasurementPoint(clustIt->x(), clustIt->y()));
            GlobalPoint clustgp = theGeomDet->surface().toGlobal( lp );
            if(m_minADC > 0  and clustIt->charge() < m_minADC) continue;
            reco::PixelClusterProperties cp = { clustgp.x(), clustgp.y(), clustgp.z(), clustIt->charge(), layer };
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
    for(unsigned int j = 0, nj = collectionJets->size(); j < nj; j++) {
        if(collectionJets->at(j).pt() < m_minJetPt) continue;

        edm::RefToBase<reco::Jet> jetRef = collectionJets->refAt(j);
//        reco::PixelClusterData* data = new reco::PixelClusterData();
//        reco::PixelClusterTagInfo* tagInfo = new reco::PixelClusterTagInfo();

        reco::PixelClusterData data = {}; // Initialize new Data to 0
        reco::PixelClusterTagInfo tagInfo = {};

        
        for(auto cluIt = clusters.begin(); cluIt != clusters.end(); ++cluIt) {
            TVector3 c3(cluIt->x - firstPV->x(), cluIt->y - firstPV->y(), cluIt->z - firstPV->z());
            TVector3 j3(jetRef->px(), jetRef->py(), jetRef->pz());
            float dR = j3.DeltaR(c3);
            float sC = 12. * 2. / (jetRef->pt()); // * jetRef->correctedP4(0).pt()
//            fillData(data, dR, sC, cluIt->layer, cluIt->charge);
//            if(cluIt->layer == 1) {
//                if(dR < 0.04) data->L1_R004++;
//                if(dR < 0.06) data->L1_R006++;
//                if(dR < 0.08) data->L1_R008++;
//                if(dR < 0.10) data->L1_R010++;
//                if(dR < 0.16) data->L1_R016++;
//                if(dR < sC)   data->L1_RVAR++;
//                if(dR < sC)   data->L1_RVWT += cluIt->charge;
//            }
//            if(cluIt->layer == 2) {
//                if(dR < 0.04) data->L2_R004++;
//                if(dR < 0.06) data->L2_R006++;
//                if(dR < 0.08) data->L2_R008++;
//                if(dR < 0.10) data->L2_R010++;
//                if(dR < 0.16) data->L2_R016++;
//                if(dR < sC)   data->L2_RVAR++;
//                if(dR < sC)   data->L2_RVWT += cluIt->charge;
//            }
//            if(cluIt->layer == 3) {
//                if(dR < 0.04) data->L3_R004++;
//                if(dR < 0.06) data->L3_R006++;
//                if(dR < 0.08) data->L3_R008++;
//                if(dR < 0.10) data->L3_R010++;
//                if(dR < 0.16) data->L3_R016++;
//                if(dR < sC)   data->L3_RVAR++;
//                if(dR < sC)   data->L3_RVWT += cluIt->charge;
//            }
//            if(cluIt->layer == 4) {
//                if(dR < 0.04) data->L4_R004++;
//                if(dR < 0.06) data->L4_R006++;
//                if(dR < 0.08) data->L4_R008++;
//                if(dR < 0.10) data->L4_R010++;
//                if(dR < 0.16) data->L4_R016++;
//                if(dR < sC)   data->L4_RVAR++;
//                if(dR < sC)   data->L4_RVWT += cluIt->charge;
//            }





            if(cluIt->layer == 1) {
                if(dR < 0.04) data.L1_R004++;
                if(dR < 0.06) data.L1_R006++;
                if(dR < 0.08) data.L1_R008++;
                if(dR < 0.10) data.L1_R010++;
                if(dR < 0.16) data.L1_R016++;
                if(dR < sC)   data.L1_RVAR++;
                if(dR < sC)   data.L1_RVWT += cluIt->charge;
            }
            if(cluIt->layer == 2) {
                if(dR < 0.04) data.L2_R004++;
                if(dR < 0.06) data.L2_R006++;
                if(dR < 0.08) data.L2_R008++;
                if(dR < 0.10) data.L2_R010++;
                if(dR < 0.16) data.L2_R016++;
                if(dR < sC)   data.L2_RVAR++;
                if(dR < sC)   data.L2_RVWT += cluIt->charge;
            }
            if(cluIt->layer == 3) {
                if(dR < 0.04) data.L3_R004++;
                if(dR < 0.06) data.L3_R006++;
                if(dR < 0.08) data.L3_R008++;
                if(dR < 0.10) data.L3_R010++;
                if(dR < 0.16) data.L3_R016++;
                if(dR < sC)   data.L3_RVAR++;
                if(dR < sC)   data.L3_RVWT += cluIt->charge;
            }
            if(cluIt->layer == 4) {
                if(dR < 0.04) data.L4_R004++;
                if(dR < 0.06) data.L4_R006++;
                if(dR < 0.08) data.L4_R008++;
                if(dR < 0.10) data.L4_R010++;
                if(dR < 0.16) data.L4_R016++;
                if(dR < sC)   data.L4_RVAR++;
                if(dR < sC)   data.L4_RVWT += cluIt->charge;
            }

        }
//std::cout << jetRef->pt() << ", " << jetRef->eta() << " : " << (int)data->L1_R010 << ", " << (int)data->L2_R010 << ", " << (int)data->L3_R010 << ", " << (int)data->L4_R010 << std::endl;
//        std::cout << jetRef->pt() << ", " << jetRef->eta() << " : " << (int)(tagInfo->data().L1_R010) << ", " << (int)(tagInfo->data().L2_R010) << ", " << (int)(tagInfo->data().L3_R010) << ", " << (int)(tagInfo->data().L4_R010) << std::endl;

//        
//        pixelTagInfo->push_back(*tagInfo);
//        
//        delete data;
//        delete tagInfo;

        




        tagInfo.setJetRef(jetRef);
        tagInfo.setData(data);

        pixelTagInfo->push_back(tagInfo);

        std::cout << jetRef->pt() << ", " << jetRef->eta() << " : " << (int)data.L1_R010 << ", " << (int)data.L2_R010 << ", " << (int)data.L3_R010 << ", " << (int)data.L4_R010 << std::endl;

        std::cout << jetRef->pt() << ", " << jetRef->eta() << " : " << (int)tagInfo.data().L1_R010 << ", " << (int)tagInfo.data().L2_R010 << ", " << (int)tagInfo.data().L3_R010 << ", " << (int)tagInfo.data().L4_R010 << std::endl;
    }
    
    // Put the TagInfo collection in the event
    iEvent.put(std::move(pixelTagInfo));
}


//void PixelClusterTagInfoProducer::fillData(reco::PixelClusterData* data, float dR, float sC, int layer, int charge) {
//    
//}

/*
// ------------ method called once each stream before processing any runs, lumis or events  ------------
void
PixelClusterTagInfoProducer::beginStream(edm::StreamID)
{
}

// ------------ method called once each stream after processing all runs, lumis and events  ------------
void
PixelClusterTagInfoProducer::endStream() {
}
*/
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
/*
// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
PixelClusterTagInfoProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}
*/
//define this as a plug-in
DEFINE_FWK_MODULE(PixelClusterTagInfoProducer);

