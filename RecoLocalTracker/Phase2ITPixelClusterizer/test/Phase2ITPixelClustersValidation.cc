#include <map>
#include <vector>
#include <algorithm>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Phase2ITPixelCluster/interface/Phase2ITPixelCluster.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"

#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLink.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "CommonTools/Utils/interface/TFileDirectory.h"

#include <TH2F.h>
#include <TH1F.h>
#include <THStack.h>


struct ClusterHistos {
    TH1F* numberClusterPixel;

    TH1F* clusterSizePixel;

    TH1F* clusterCharge;

    TH2F* globalPosXY[2];
    TH2F* localPosXY[2];
    TH1F* localPosX[2];
    TH1F* localPosY[2];
    TH1F* localCol[2];
    TH1F* localRow[2];
  
    TH1F* deltaXClusterSimHits[2];
    TH1F* deltaYClusterSimHits[2];

    TH1F* deltaXClusterSimHits_P[2];
    TH1F* deltaYClusterSimHits_P[2];

    TH1F* primarySimHits;
    TH1F* otherSimHits;
};

class Phase2ITPixelClustersValidation : public edm::one::EDAnalyzer<> {

    public:

        typedef std::map< unsigned int, std::vector< PSimHit > > SimHitsMap;
        typedef std::map< unsigned int, SimTrack > SimTracksMap;

        explicit Phase2ITPixelClustersValidation(const edm::ParameterSet&);
        ~Phase2ITPixelClustersValidation();
        void beginJob() override;
        void endJob() override;
        void analyze(const edm::Event&, const edm::EventSetup&) override;

    private:

        std::map< unsigned int, ClusterHistos >::iterator createLayerHistograms(unsigned int);
        unsigned int getLayerNumber(const DetId&, const TrackerTopology*);
        unsigned int getSimTrackId(const edm::Handle< edm::DetSetVector< PixelDigiSimLink > >&, const DetId&, unsigned int);

        edm::EDGetTokenT< Phase2ITPixelClusterCollectionNew > tokenClusters_;
        edm::EDGetTokenT< edm::DetSetVector< PixelDigiSimLink > > tokenLinks_;
        edm::EDGetTokenT< edm::PSimHitContainer > tokenSimHits_;
        edm::EDGetTokenT< edm::SimTrackContainer> tokenSimTracks_;
        edm::EDGetTokenT< edm::SimVertexContainer > tokenSimVertices_;

        TH2F* trackerLayout_;
        TH2F* trackerLayoutXY_;
        TH2F* trackerLayoutXYBar_;
        TH2F* trackerLayoutXYEC_;

        std::map< unsigned int, ClusterHistos > histograms_;

};

    Phase2ITPixelClustersValidation::Phase2ITPixelClustersValidation(const edm::ParameterSet& conf) { 
        tokenClusters_ = consumes< Phase2ITPixelClusterCollectionNew >(conf.getParameter<edm::InputTag>("src"));
        tokenLinks_ = consumes< edm::DetSetVector< PixelDigiSimLink> >(conf.getParameter<edm::InputTag>("links"));
        tokenSimHits_ = consumes< edm::PSimHitContainer >(edm::InputTag("g4SimHits", "TrackerHitsPixelBarrelLowTof"));
        tokenSimTracks_ = consumes< edm::SimTrackContainer >(edm::InputTag("g4SimHits"));
        tokenSimVertices_ = consumes< edm::SimVertexContainer >(edm::InputTag("g4SimHits")); 
    }

    Phase2ITPixelClustersValidation::~Phase2ITPixelClustersValidation() { }

    void Phase2ITPixelClustersValidation::beginJob() {
        edm::Service<TFileService> fs; 
        fs->file().cd("/");
        TFileDirectory td = fs->mkdir("Common");
        // Create common histograms
        trackerLayout_ = td.make< TH2F >("RVsZ", "R vs. z position", 6000, -300.0, 300.0, 1200, 0.0, 120.0);
        trackerLayoutXY_ = td.make< TH2F >("XVsY", "x vs. y position", 2400, -120.0, 120.0, 2400, -120.0, 120.0);
        trackerLayoutXYBar_ = td.make< TH2F >("XVsYBar", "x vs. y position", 2400, -120.0, 120.0, 2400, -120.0, 120.0);
        trackerLayoutXYEC_ = td.make< TH2F >("XVsYEC", "x vs. y position", 2400, -120.0, 120.0, 2400, -120.0, 120.0);

    }

void Phase2ITPixelClustersValidation::endJob() { }

void Phase2ITPixelClustersValidation::analyze(const edm::Event& event, const edm::EventSetup& eventSetup) {

    /*
     * Get the needed objects
     */

    // Get the clusters
    edm::Handle< Phase2ITPixelClusterCollectionNew > clusters;
    event.getByToken(tokenClusters_, clusters);

    // Get the PixelDigiSimLinks
    edm::Handle< edm::DetSetVector< PixelDigiSimLink > > pixelSimLinks;
    event.getByToken(tokenLinks_, pixelSimLinks);

    // Get the SimHits
    edm::Handle< edm::PSimHitContainer > simHitsRaw;
    event.getByToken(tokenSimHits_, simHitsRaw);

    // Get the SimTracks
    edm::Handle< edm::SimTrackContainer > simTracksRaw;
    event.getByToken(tokenSimTracks_, simTracksRaw);

    // Get the SimVertex
    edm::Handle< edm::SimVertexContainer > simVertices;
    event.getByToken(tokenSimVertices_, simVertices);

    // Get the geometry
    edm::ESHandle< TrackerGeometry > geomHandle;
    eventSetup.get< TrackerDigiGeometryRecord >().get(geomHandle);
    const TrackerGeometry* tkGeom = &(*geomHandle);


    edm::ESHandle< TrackerTopology > tTopoHandle;
    eventSetup.get< TrackerTopologyRcd >().get(tTopoHandle);
    const TrackerTopology* tTopo = tTopoHandle.product();

    /*
     * Rearrange the simTracks
     */

    // Rearrange the simTracks for ease of use <simTrackID, simTrack>
    SimTracksMap simTracks;
    for (edm::SimTrackContainer::const_iterator simTrackIt(simTracksRaw->begin()); simTrackIt != simTracksRaw->end(); ++simTrackIt) simTracks.insert(std::pair< unsigned int, SimTrack >(simTrackIt->trackId(), *simTrackIt));

    /*
     * Rearrange the simHits by detUnit
     */

    // Rearrange the simHits for ease of use 
    SimHitsMap simHitsDetUnit;
    SimHitsMap simHitsTrackId;
    for (edm::PSimHitContainer::const_iterator simHitIt(simHitsRaw->begin()); simHitIt != simHitsRaw->end(); ++simHitIt) {
        SimHitsMap::iterator simHitsDetUnitIt(simHitsDetUnit.find(simHitIt->detUnitId()));
        if (simHitsDetUnitIt == simHitsDetUnit.end()) {
            std::pair< SimHitsMap::iterator, bool > newIt(simHitsDetUnit.insert(std::pair< unsigned int, std::vector< PSimHit > >(simHitIt->detUnitId(), std::vector< PSimHit >())));
            simHitsDetUnitIt = newIt.first;
        }
        simHitsDetUnitIt->second.push_back(*simHitIt);

        SimHitsMap::iterator simHitsTrackIdIt(simHitsTrackId.find(simHitIt->trackId()));
        if (simHitsTrackIdIt == simHitsTrackId.end()) {
            std::pair< SimHitsMap::iterator, bool > newIt(simHitsTrackId.insert(std::pair< unsigned int, std::vector< PSimHit > >(simHitIt->trackId(), std::vector< PSimHit >())));
            simHitsTrackIdIt = newIt.first;
        }
        simHitsTrackIdIt->second.push_back(*simHitIt);
    }

    /*
     * Validation   
     */

    // Loop over modules
    for (Phase2ITPixelClusterCollectionNew::const_iterator DSViter = clusters->begin(); DSViter != clusters->end(); ++DSViter) {

        // Get the detector unit's id
        unsigned int rawid(DSViter->detId()); 
        DetId detId(rawid);
        unsigned int layer(getLayerNumber(detId, tTopo));
        // Get the geometry of the tracker
        const GeomDetUnit* geomDetUnit(tkGeom->idToDetUnit(detId));
        const PixelGeomDetUnit* theGeomDet = dynamic_cast< const PixelGeomDetUnit* >(geomDetUnit);
        const PixelTopology& topol = theGeomDet->specificTopology();

        if (!geomDetUnit) break;

        // Create histograms for the layer if they do not yet exist
        std::map< unsigned int, ClusterHistos >::iterator histogramLayer(histograms_.find(layer));
        if (histogramLayer == histograms_.end()) histogramLayer = createLayerHistograms(layer);

        // Number of clusters
        unsigned int nClustersPixel(0);

        // Loop over the clusters in the detector unit
        for (edmNew::DetSet< Phase2ITPixelCluster >::const_iterator clustIt = DSViter->begin(); clustIt != DSViter->end(); ++clustIt) {
            /*
             * Cluster related variables
             */


	    MeasurementPoint mpClu(clustIt->x(), clustIt->y());
            Local3DPoint localPosClu = geomDetUnit->topology().localPosition(mpClu);
            Global3DPoint globalPosClu = geomDetUnit->surface().toGlobal(localPosClu);

            // Fill the position histograms
            trackerLayout_->Fill(globalPosClu.z(), globalPosClu.perp());
            trackerLayoutXY_->Fill(globalPosClu.x(), globalPosClu.y());
            if (layer < 100) trackerLayoutXYBar_->Fill(globalPosClu.x(), globalPosClu.y());
            else trackerLayoutXYEC_->Fill(globalPosClu.x(), globalPosClu.y());

            histogramLayer->second.localPosXY[0]->Fill(localPosClu.x(), localPosClu.y());
            histogramLayer->second.globalPosXY[0]->Fill(globalPosClu.x(), globalPosClu.y());

            histogramLayer->second.localPosX[0]->Fill(localPosClu.x());
            histogramLayer->second.localPosY[0]->Fill(localPosClu.y());

            // Pixel module
            if (topol.ncolumns() > 32) {



	      histogramLayer->second.localCol[1]->Fill(clustIt->minPixelCol());
	      histogramLayer->second.localCol[1]->Fill(clustIt->maxPixelCol());
	      histogramLayer->second.localRow[1]->Fill(clustIt->minPixelRow());
	      histogramLayer->second.localRow[1]->Fill(clustIt->maxPixelRow());

 
 	        histogramLayer->second.localPosX[1]->Fill(localPosClu.x());
	        histogramLayer->second.localPosY[1]->Fill(localPosClu.y());

                histogramLayer->second.localPosXY[1]->Fill(localPosClu.x(), localPosClu.y());
                histogramLayer->second.globalPosXY[1]->Fill(globalPosClu.x(), globalPosClu.y());
                histogramLayer->second.clusterSizePixel->Fill(clustIt->size());
                histogramLayer->second.clusterCharge->Fill(clustIt->charge()*1.e-3);  // *1.e-3 conversion to ke
                ++nClustersPixel;
            }


            /*
             * Digis related variables
             */

            std::vector< unsigned int > clusterSimTrackIds;

	    // ATricomi + EMigliore: skip for now 
	    // Phase2ITPixelCluster::size: should return an unsigned int instead of a int?

            // // Get all the simTracks that form the cluster
            // for (unsigned int i(0); i < clustIt->size(); ++i) {
            //     unsigned int channel(PixelDigi::pixelToChannel(clustIt->firstRow() + i, clustIt->column())); // Here we have to use the old pixelToChannel function (not Phase2ITPixelDigi but PixelDigi), change this when using new Digis
            //     unsigned int simTrackId(getSimTrackId(pixelSimLinks, detId, channel));
            //     clusterSimTrackIds.push_back(simTrackId);
            // }

            /*
             * SimHits related variables
             */


            unsigned int primarySimHits(0);
            unsigned int otherSimHits(0);

            for (edm::PSimHitContainer::const_iterator hitIt(simHitsRaw->begin()); hitIt != simHitsRaw->end(); ++hitIt) {
                if (rawid == hitIt->detUnitId() and std::find(clusterSimTrackIds.begin(), clusterSimTrackIds.end(), hitIt->trackId()) != clusterSimTrackIds.end()) {
                    Local3DPoint localPosHit(hitIt->localPosition());

                    histogramLayer->second.deltaXClusterSimHits[0]->Fill(localPosClu.x() - localPosHit.x());
                    histogramLayer->second.deltaYClusterSimHits[0]->Fill(localPosClu.y() - localPosHit.y());

                    // Pixel module
                    if (topol.ncolumns() > 32) {
                        histogramLayer->second.deltaXClusterSimHits[1]->Fill(localPosClu.x() - localPosHit.x());
                        histogramLayer->second.deltaYClusterSimHits[1]->Fill(localPosClu.y() - localPosHit.y());
                    }


                    ++otherSimHits;

                    std::map< unsigned int, SimTrack >::const_iterator simTrackIt(simTracks.find(hitIt->trackId()));
                    if (simTrackIt == simTracks.end()) continue;

                    // Primary particles only
                    unsigned int processType(hitIt->processType());
                    if (simTrackIt->second.vertIndex() == 0 and (processType == 2 || processType == 7 || processType == 9 || processType == 11 || processType == 13 || processType == 15)) {
                        histogramLayer->second.deltaXClusterSimHits_P[0]->Fill(localPosClu.x() - localPosHit.x());
                        histogramLayer->second.deltaYClusterSimHits_P[0]->Fill(localPosClu.y() - localPosHit.y());

                        // Pixel module
                        if (topol.ncolumns() > 32) {
                            histogramLayer->second.deltaXClusterSimHits_P[1]->Fill(localPosClu.x() - localPosHit.x());
                            histogramLayer->second.deltaYClusterSimHits_P[1]->Fill(localPosClu.y() - localPosHit.y());
                        } 

                        ++primarySimHits;
                    }
                }
            }

            otherSimHits -= primarySimHits;

            histogramLayer->second.primarySimHits->Fill(primarySimHits);
            histogramLayer->second.otherSimHits->Fill(otherSimHits);
        }

        if (nClustersPixel) histogramLayer->second.numberClusterPixel->Fill(nClustersPixel);
    }
}

// Create the histograms
std::map< unsigned int, ClusterHistos >::iterator Phase2ITPixelClustersValidation::createLayerHistograms(unsigned int ival) {
    std::ostringstream fname1, fname2;

    edm::Service<TFileService> fs;
    fs->file().cd("/");

    std::string tag;
    unsigned int id;
    if (ival < 100) {
        id = ival;
        fname1 << "Barrel";
        fname2 << "Layer_" << id;
        tag = "_layer_";
    }
    else {
        int side = ival / 100;
        id = ival - side * 100;
        fname1 << "EndCap_Side_" << side;
        fname2 << "Disc_" << id;
        tag = "_disc_";
    }

    TFileDirectory td1 = fs->mkdir(fname1.str().c_str());
    TFileDirectory td = td1.mkdir(fname2.str().c_str());

    ClusterHistos local_histos;

    std::ostringstream histoName;

    /*
     * Number of clusters
     */

    histoName.str(""); histoName << "Number_Clusters_Pixel" << tag.c_str() <<  id;
    local_histos.numberClusterPixel = td.make< TH1F >(histoName.str().c_str(), histoName.str().c_str(), 20, 0., 20.);


    /*
     * Cluster size
     */

    histoName.str(""); histoName << "Cluster_Size_Pixel" << tag.c_str() <<  id;
    local_histos.clusterSizePixel = td.make< TH1F >(histoName.str().c_str(), histoName.str().c_str(), 20, -0.5, 19.5);

    /*
     * Cluster charge [ke]
     */

    histoName.str(""); histoName << "Cluster_Charge" << tag.c_str() <<  id;
    local_histos.clusterCharge = td.make< TH1F >(histoName.str().c_str(), histoName.str().c_str(), 250, 0., 500.);


    /*
     * Local and Global positions
     */

    histoName.str(""); histoName << "Local_Position_XY_Mixed" << tag.c_str() <<  id;
    local_histos.localPosXY[0] = td.make< TH2F >(histoName.str().c_str(), histoName.str().c_str(), 2000, 0., 0., 2000, 0., 0.);

    histoName.str(""); histoName << "Local_Position_XY_Pixel" << tag.c_str() <<  id;
    local_histos.localPosXY[1] = td.make< TH2F >(histoName.str().c_str(), histoName.str().c_str(), 2000, 0., 0., 2000, 0., 0.);

    histoName.str(""); histoName << "Local_Position_X_Mixed" << tag.c_str() <<  id;
    local_histos.localPosX[0] = td.make< TH1F >(histoName.str().c_str(), histoName.str().c_str(), 2000, 0., 0.);

    histoName.str(""); histoName << "Local_Position_X_Pixel" << tag.c_str() <<  id;
    local_histos.localPosX[1] = td.make< TH1F >(histoName.str().c_str(), histoName.str().c_str(), 2000, 0., 0.);

    histoName.str(""); histoName << "Local_Position_Y_Mixed" << tag.c_str() <<  id;
    local_histos.localPosY[0] = td.make< TH1F >(histoName.str().c_str(), histoName.str().c_str(), 2000, 0., 0.);

    histoName.str(""); histoName << "Local_Position_Y_Pixel" << tag.c_str() <<  id;
    local_histos.localPosY[1] = td.make< TH1F >(histoName.str().c_str(), histoName.str().c_str(), 2000, 0., 0.);

    histoName.str(""); histoName << "Local_Position_Col_Mixed" << tag.c_str() <<  id;
    local_histos.localCol[0] = td.make< TH1F >(histoName.str().c_str(), histoName.str().c_str(), 2000, 0., 0.);

    histoName.str(""); histoName << "Local_Position_Col_Pixel" << tag.c_str() <<  id;
    local_histos.localCol[1] = td.make< TH1F >(histoName.str().c_str(), histoName.str().c_str(), 2000, 0., 0.);

    histoName.str(""); histoName << "Local_Position_Row_Mixed" << tag.c_str() <<  id;
    local_histos.localRow[0] = td.make< TH1F >(histoName.str().c_str(), histoName.str().c_str(), 2000, 0., 0.);

    histoName.str(""); histoName << "Local_Position_Row_Pixel" << tag.c_str() <<  id;
    local_histos.localRow[1] = td.make< TH1F >(histoName.str().c_str(), histoName.str().c_str(), 2000, 0., 0.);

    histoName.str(""); histoName << "Global_Position_XY_Mixed" << tag.c_str() <<  id;
    local_histos.globalPosXY[0] = td.make< TH2F >(histoName.str().c_str(), histoName.str().c_str(), 2400, -120.0, 120.0, 2400, -120.0, 120.0);

    histoName.str(""); histoName << "Global_Position_XY_Pixel" << tag.c_str() <<  id;
    local_histos.globalPosXY[1] = td.make< TH2F >(histoName.str().c_str(), histoName.str().c_str(), 1200, -20.0, 20.0, 1200, -20.0, 20.0); 


    /*
     * Delta positions with SimHits
     */

    histoName.str(""); histoName << "Delta_X_Cluster_SimHits_Mixed" << tag.c_str() <<  id;
    local_histos.deltaXClusterSimHits[0] = td.make< TH1F >(histoName.str().c_str(), histoName.str().c_str(), 100, 0., 0.);

    histoName.str(""); histoName << "Delta_X_Cluster_SimHits_Pixel" << tag.c_str() <<  id;
    local_histos.deltaXClusterSimHits[1] = td.make< TH1F >(histoName.str().c_str(), histoName.str().c_str(), 100, 0., 0.);

    histoName.str(""); histoName << "Delta_Y_Cluster_SimHits_Mixed" << tag.c_str() <<  id;
    local_histos.deltaYClusterSimHits[0] = td.make< TH1F >(histoName.str().c_str(), histoName.str().c_str(), 100, 0., 0.);

    histoName.str(""); histoName << "Delta_Y_Cluster_SimHits_Pixel" << tag.c_str() <<  id;
    local_histos.deltaYClusterSimHits[1] = td.make< TH1F >(histoName.str().c_str(), histoName.str().c_str(), 100, 0., 0.);

    /*
     * Delta position with simHits for primary tracks only
     */

    histoName.str(""); histoName << "Delta_X_Cluster_SimHits_Mixed_P" << tag.c_str() <<  id;
    local_histos.deltaXClusterSimHits_P[0] = td.make< TH1F >(histoName.str().c_str(), histoName.str().c_str(), 100, 0., 0.);

    histoName.str(""); histoName << "Delta_X_Cluster_SimHits_Pixel_P" << tag.c_str() <<  id;
    local_histos.deltaXClusterSimHits_P[1] = td.make< TH1F >(histoName.str().c_str(), histoName.str().c_str(), 100, 0., 0.);

    histoName.str(""); histoName << "Delta_Y_Cluster_SimHits_Mixed_P" << tag.c_str() <<  id;
    local_histos.deltaYClusterSimHits_P[0] = td.make< TH1F >(histoName.str().c_str(), histoName.str().c_str(), 100, 0., 0.);

    histoName.str(""); histoName << "Delta_Y_Cluster_SimHits_Pixel_P" << tag.c_str() <<  id;
    local_histos.deltaYClusterSimHits_P[1] = td.make< TH1F >(histoName.str().c_str(), histoName.str().c_str(), 100, 0., 0.);

    /*
     * Information on the Digis per cluster
     */

    histoName.str(""); histoName << "Primary_Digis" << tag.c_str() <<  id;
    local_histos.primarySimHits= td.make< TH1F >(histoName.str().c_str(), histoName.str().c_str(), 10, 0., 10.);

    histoName.str(""); histoName << "Other_Digis" << tag.c_str() <<  id;
    local_histos.otherSimHits= td.make< TH1F >(histoName.str().c_str(), histoName.str().c_str(), 10, 0., 10.);

    /*
     * End
     */

    std::pair< std::map< unsigned int, ClusterHistos >::iterator, bool > insertedIt(histograms_.insert(std::make_pair(ival, local_histos)));
    fs->file().cd("/");

    return insertedIt.first;
}

unsigned int Phase2ITPixelClustersValidation::getLayerNumber(const DetId& detid, const TrackerTopology* topo) {
    if (detid.det() == DetId::Tracker) {
        if (detid.subdetId() == PixelSubdetector::PixelBarrel) return (topo->pxbLayer(detid));
        else if (detid.subdetId() == PixelSubdetector::PixelEndcap) return (100 * topo->pxfSide(detid) + topo->pxfDisk(detid));
        else return 999;
    }
    return 999;
}

unsigned int Phase2ITPixelClustersValidation::getSimTrackId(const edm::Handle< edm::DetSetVector< PixelDigiSimLink > >& pixelSimLinks, const DetId& detId, unsigned int channel) {
    edm::DetSetVector< PixelDigiSimLink >::const_iterator DSViter(pixelSimLinks->find(detId));
    if (DSViter == pixelSimLinks->end()) return 0;
    for (edm::DetSet< PixelDigiSimLink >::const_iterator it = DSViter->data.begin(); it != DSViter->data.end(); ++it) {
        if (channel == it->channel()) return it->SimTrackId();
    }
    return 0;
}

DEFINE_FWK_MODULE(Phase2ITPixelClustersValidation);
