#include <iostream>
#include <map>
#include <vector>
#include <algorithm>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

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
#include "DataFormats/Phase2TrackerCluster/interface/Phase2TrackerCluster1D.h"
#include "DataFormats/TrackerRecHit2D/interface/Phase2TrackerRecHit1D.h"

#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLink.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "CommonTools/Utils/interface/TFileDirectory.h"

#include <TH2D.h>
#include <TH1D.h>
#include <THStack.h>


struct RecHitHistos {
    THStack* numberRecHitsMixed;
    TH1D* numberRecHitsPixel;
    TH1D* numberRecHitsStrip;

    TH2D* globalPosXY[3];
    TH2D* localPosXY[3];

    TH1F* deltaXRecHitsSimHits[3];
    TH1F* deltaYRecHitsSimHits[3];

    TH1F* deltaXRecHitsSimHits_P[3];
    TH1F* deltaYRecHitsSimHits_P[3];

    TH1F* primarySimHits;
    TH1F* otherSimHits;
};


class Phase2TrackerRecHitsValidation : public edm::EDAnalyzer {

    public:

        typedef std::map< unsigned int, std::vector< PSimHit > > SimHitsMap;
        typedef std::map< unsigned int, SimTrack > SimTracksMap;

        explicit Phase2TrackerRecHitsValidation(const edm::ParameterSet&);
        ~Phase2TrackerRecHitsValidation();
        void beginJob() override;
        void endJob() override;
        void analyze(const edm::Event&, const edm::EventSetup&) override;

    private:

        std::map< unsigned int, RecHitHistos >::iterator createLayerHistograms(unsigned int);
        unsigned int getSimTrackId(const edm::Handle< edm::DetSetVector< PixelDigiSimLink > >&, const DetId&, unsigned int);

        edm::EDGetTokenT< Phase2TrackerRecHit1DCollectionNew > tokenRecHits_;
        edm::EDGetTokenT< Phase2TrackerCluster1DCollectionNew > tokenClusters_;
        edm::EDGetTokenT< edm::DetSetVector< PixelDigiSimLink > > tokenLinks_;
        edm::EDGetTokenT< edm::PSimHitContainer > tokenSimHitsB_;
        edm::EDGetTokenT< edm::PSimHitContainer > tokenSimHitsE_;
        edm::EDGetTokenT< edm::SimTrackContainer> tokenSimTracks_;
        //edm::EDGetTokenT< edm::SimVertexContainer > tokenSimVertices_;

        TH2D* trackerLayout_;
        TH2D* trackerLayoutXY_;
        TH2D* trackerLayoutXYBar_;
        TH2D* trackerLayoutXYEC_;

        std::map< unsigned int, RecHitHistos > histograms_;

};


Phase2TrackerRecHitsValidation::Phase2TrackerRecHitsValidation(const edm::ParameterSet& conf) :
    tokenRecHits_  (consumes< Phase2TrackerRecHit1DCollectionNew    >(conf.getParameter<edm::InputTag>("src"))),
    tokenClusters_ (consumes< Phase2TrackerCluster1DCollectionNew   >(conf.getParameter<edm::InputTag>("clusters"))),
    tokenLinks_    (consumes< edm::DetSetVector< PixelDigiSimLink > >(conf.getParameter<edm::InputTag>("links"))),
    tokenSimHitsB_ (consumes< edm::PSimHitContainer                 >(conf.getParameter<edm::InputTag>("simhitsbarrel"))),
    tokenSimHitsE_ (consumes< edm::PSimHitContainer                 >(conf.getParameter<edm::InputTag>("simhitsendcap"))),
    tokenSimTracks_(consumes< edm::SimTrackContainer                >(conf.getParameter<edm::InputTag>("simtracks"))) {
}


Phase2TrackerRecHitsValidation::~Phase2TrackerRecHitsValidation() { }


void Phase2TrackerRecHitsValidation::beginJob() {
    edm::Service<TFileService> fs; 
    fs->file().cd("/");
    TFileDirectory td = fs->mkdir("Common");
    // Create common histograms
    trackerLayout_ = td.make< TH2D >("RVsZ", "R vs. z position", 6000, -300.0, 300.0, 1200, 0.0, 120.0);
    trackerLayoutXY_ = td.make< TH2D >("XVsY", "x vs. y position", 2400, -120.0, 120.0, 2400, -120.0, 120.0);
    trackerLayoutXYBar_ = td.make< TH2D >("XVsYBar", "x vs. y position", 2400, -120.0, 120.0, 2400, -120.0, 120.0);
    trackerLayoutXYEC_ = td.make< TH2D >("XVsYEC", "x vs. y position", 2400, -120.0, 120.0, 2400, -120.0, 120.0);
}


void Phase2TrackerRecHitsValidation::endJob() { }


void Phase2TrackerRecHitsValidation::analyze(const edm::Event& event, const edm::EventSetup& eventSetup) {

    /*
     * Get the needed objects
     */

    // Get the RecHits
    edm::Handle< Phase2TrackerRecHit1DCollectionNew > rechits;
    event.getByToken(tokenRecHits_, rechits);

    // Get the Clusters
    edm::Handle< Phase2TrackerCluster1DCollectionNew > clusters;
    event.getByToken(tokenClusters_, clusters);

    // Get the PixelDigiSimLinks
    edm::Handle< edm::DetSetVector< PixelDigiSimLink > > pixelSimLinks;
    event.getByToken(tokenLinks_, pixelSimLinks);

    // Get the SimHits
    edm::Handle< edm::PSimHitContainer > simHitsRaw[2];
    event.getByToken(tokenSimHitsB_, simHitsRaw[0]);
    event.getByToken(tokenSimHitsE_, simHitsRaw[1]);

    // Get the SimTracks
    edm::Handle< edm::SimTrackContainer > simTracksRaw;
    event.getByToken(tokenSimTracks_, simTracksRaw);

    // Get the SimVertex
    //edm::Handle< edm::SimVertexContainer > simVertices;
    //event.getByToken(tokenSimVertices_, simVertices);

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
    for (unsigned int simhitidx = 0; simhitidx < 2; ++simhitidx) {
      for (edm::PSimHitContainer::const_iterator simHitIt(simHitsRaw[simhitidx]->begin()); simHitIt != simHitsRaw[simhitidx]->end(); ++simHitIt) {
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
    }

    /*
     * Validation   
     */


    // Loop over modules
    for (Phase2TrackerRecHit1DCollectionNew::const_iterator DSViter = rechits->begin(); DSViter != rechits->end(); ++DSViter) {

        // Get the detector unit's id
        unsigned int rawid(DSViter->detId()); 
        DetId detId(rawid);
        unsigned int layer = tTopo->side(detId)*100 + tTopo->layer(detId);
	TrackerGeometry::ModuleType mType = tkGeom->getDetectorType(detId);

        // Get the geomdet
        const GeomDetUnit* geomDetUnit(tkGeom->idToDetUnit(detId));
        if (!geomDetUnit) break;

        // Create histograms for the layer if they do not yet exist
        std::map< unsigned int, RecHitHistos >::iterator histogramLayer(histograms_.find(layer));
        if (histogramLayer == histograms_.end()) histogramLayer = createLayerHistograms(layer);

        // Number of rechits
        unsigned int nRecHitsPixel(0), nRecHitsStrip(0);

        // Loop over the rechits in the detector unit
        for (edmNew::DetSet< Phase2TrackerRecHit1D >::const_iterator rechitIt = DSViter->begin(); rechitIt != DSViter->end(); ++rechitIt) {

            /*
             * Rechit related variables
             */

            LocalPoint localPosClu = rechitIt->localPosition();
            Global3DPoint globalPosClu = geomDetUnit->surface().toGlobal(localPosClu);

            // Fill the position histograms
            trackerLayout_->Fill(globalPosClu.z(), globalPosClu.perp());
            trackerLayoutXY_->Fill(globalPosClu.x(), globalPosClu.y());
            if (layer < 100) trackerLayoutXYBar_->Fill(globalPosClu.x(), globalPosClu.y());
            else trackerLayoutXYEC_->Fill(globalPosClu.x(), globalPosClu.y());

            histogramLayer->second.localPosXY[0]->Fill(localPosClu.x(), localPosClu.y());
	    if (layer<100)
              histogramLayer->second.globalPosXY[0]->Fill(globalPosClu.z(), globalPosClu.perp());
	    else
              histogramLayer->second.globalPosXY[0]->Fill(globalPosClu.x(), globalPosClu.y());

            // Pixel module
            if (mType == TrackerGeometry::ModuleType::Ph2PSP) {
                histogramLayer->second.localPosXY[1]->Fill(localPosClu.x(), localPosClu.y());
		if (layer<100)
                  histogramLayer->second.globalPosXY[1]->Fill(globalPosClu.z(), globalPosClu.perp());
		else
                  histogramLayer->second.globalPosXY[1]->Fill(globalPosClu.x(), globalPosClu.y());
                ++nRecHitsPixel;
            }
            // Strip module
            else if (mType == TrackerGeometry::ModuleType::Ph2PSS || mType == TrackerGeometry::ModuleType::Ph2SS) {
                histogramLayer->second.localPosXY[2]->Fill(localPosClu.x(), localPosClu.y());
		if (layer<100)
                  histogramLayer->second.globalPosXY[2]->Fill(globalPosClu.z(), globalPosClu.perp());
		else
                  histogramLayer->second.globalPosXY[2]->Fill(globalPosClu.x(), globalPosClu.y());
                ++nRecHitsStrip;
            }

            /*
	     * Get the cluster from the rechit
	     */
	    
	    const Phase2TrackerCluster1D * clustIt = &*rechitIt->cluster();

            /*
             * Digis related variables
             */

            std::vector< unsigned int > clusterSimTrackIds;

            // Get all the simTracks that form the cluster
            for (unsigned int i(0); i < clustIt->size(); ++i) {
                unsigned int channel(Phase2TrackerDigi::pixelToChannel(clustIt->firstRow() + i, clustIt->column())); 
                unsigned int simTrackId(getSimTrackId(pixelSimLinks, detId, channel));
                clusterSimTrackIds.push_back(simTrackId);
            }

            /*
             * SimHits related variables
             */

            unsigned int primarySimHits(0);
            unsigned int otherSimHits(0);

            for (unsigned int simhitidx = 0; simhitidx < 2; ++simhitidx) {
              for (edm::PSimHitContainer::const_iterator hitIt(simHitsRaw[simhitidx]->begin()); hitIt != simHitsRaw[simhitidx]->end(); ++hitIt) {
                if (rawid == hitIt->detUnitId() and std::find(clusterSimTrackIds.begin(), clusterSimTrackIds.end(), hitIt->trackId()) != clusterSimTrackIds.end()) {
                    Local3DPoint localPosHit(hitIt->localPosition());

                    histogramLayer->second.deltaXRecHitsSimHits[0]->Fill(localPosClu.x() - localPosHit.x());
                    histogramLayer->second.deltaYRecHitsSimHits[0]->Fill(localPosClu.y() - localPosHit.y());
                    // Pixel module
                    if (mType == TrackerGeometry::ModuleType::Ph2PSP) {
                        histogramLayer->second.deltaXRecHitsSimHits[1]->Fill(localPosClu.x() - localPosHit.x());
                        histogramLayer->second.deltaYRecHitsSimHits[1]->Fill(localPosClu.y() - localPosHit.y());
                    }
                    // Strip module
                    else if (mType == TrackerGeometry::ModuleType::Ph2PSS || mType == TrackerGeometry::ModuleType::Ph2SS) {
                        histogramLayer->second.deltaXRecHitsSimHits[2]->Fill(localPosClu.x() - localPosHit.x());
                        histogramLayer->second.deltaYRecHitsSimHits[2]->Fill(localPosClu.y() - localPosHit.y());
                    }

                    ++otherSimHits;

                    std::map< unsigned int, SimTrack >::const_iterator simTrackIt(simTracks.find(hitIt->trackId()));
                    if (simTrackIt == simTracks.end()) continue;

                    // Primary particles only
                    unsigned int processType(hitIt->processType());
                    if (simTrackIt->second.vertIndex() == 0 and (processType == 2 || processType == 7 || processType == 9 || processType == 11 || processType == 13 || processType == 15)) {
                        histogramLayer->second.deltaXRecHitsSimHits_P[0]->Fill(localPosClu.x() - localPosHit.x());
                        histogramLayer->second.deltaYRecHitsSimHits_P[0]->Fill(localPosClu.y() - localPosHit.y());

                        // Pixel module
                        if (mType == TrackerGeometry::ModuleType::Ph2PSP) {
                            histogramLayer->second.deltaXRecHitsSimHits_P[1]->Fill(localPosClu.x() - localPosHit.x());
                            histogramLayer->second.deltaYRecHitsSimHits_P[1]->Fill(localPosClu.y() - localPosHit.y());
                        } 
                        // Strip module
                        else if (mType == TrackerGeometry::ModuleType::Ph2PSS || mType == TrackerGeometry::ModuleType::Ph2SS) {
                            histogramLayer->second.deltaXRecHitsSimHits_P[2]->Fill(localPosClu.x() - localPosHit.x());
                            histogramLayer->second.deltaYRecHitsSimHits_P[2]->Fill(localPosClu.y() - localPosHit.y());
                        }

                        ++primarySimHits;
                    }
                }
	      }
            }

            otherSimHits -= primarySimHits;

            histogramLayer->second.primarySimHits->Fill(primarySimHits);
            histogramLayer->second.otherSimHits->Fill(otherSimHits);

        }

        if (nRecHitsPixel) histogramLayer->second.numberRecHitsPixel->Fill(nRecHitsPixel);
        if (nRecHitsStrip) histogramLayer->second.numberRecHitsStrip->Fill(nRecHitsStrip);

    }
}

// Create the histograms
std::map< unsigned int, RecHitHistos >::iterator Phase2TrackerRecHitsValidation::createLayerHistograms(unsigned int ival) {
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
//        fname1 << "EndCap_Side_" << side;
        fname1 << "EndCap";
        fname2 << "Ring_" << id;
        tag = "_ring_";
    }

    TFileDirectory td1 = fs->mkdir(fname1.str().c_str());
    TFileDirectory td = td1.mkdir(fname2.str().c_str());

    RecHitHistos local_histos;

    std::ostringstream histoName;

    /*
     * Number of rechits
     */

    histoName.str(""); histoName << "Number_RecHits_Pixel" << tag.c_str() <<  id;
    local_histos.numberRecHitsPixel = td.make< TH1D >(histoName.str().c_str(), histoName.str().c_str(), 20, 0., 20.);
    local_histos.numberRecHitsPixel->SetFillColor(kAzure + 7);

    histoName.str(""); histoName << "Number_RecHits_Strip" << tag.c_str() <<  id;
    local_histos.numberRecHitsStrip = td.make< TH1D >(histoName.str().c_str(), histoName.str().c_str(), 20, 0., 20.);
    local_histos.numberRecHitsStrip->SetFillColor(kOrange - 3);

    histoName.str(""); histoName << "Number_RecHitss_Mixed" << tag.c_str() <<  id;
    local_histos.numberRecHitsMixed = td.make< THStack >(histoName.str().c_str(), histoName.str().c_str());
    local_histos.numberRecHitsMixed->Add(local_histos.numberRecHitsPixel);
    local_histos.numberRecHitsMixed->Add(local_histos.numberRecHitsStrip);

    /*
     * Local and Global positions
     */

    histoName.str(""); histoName << "Local_Position_XY_Mixed" << tag.c_str() <<  id;
    local_histos.localPosXY[0] = td.make< TH2D >(histoName.str().c_str(), histoName.str().c_str(), 2000, 0., 0., 2000, 0., 0.);

    histoName.str(""); histoName << "Local_Position_XY_Pixel" << tag.c_str() <<  id;
    local_histos.localPosXY[1] = td.make< TH2D >(histoName.str().c_str(), histoName.str().c_str(), 2000, 0., 0., 2000, 0., 0.);

    histoName.str(""); histoName << "Local_Position_XY_Strip" << tag.c_str() <<  id;
    local_histos.localPosXY[2] = td.make< TH2D >(histoName.str().c_str(), histoName.str().c_str(), 2000, 0., 0., 2000, 0., 0.);

    histoName.str(""); histoName << "Global_Position_XY_Mixed" << tag.c_str() <<  id;
    local_histos.globalPosXY[0] = td.make< TH2D >(histoName.str().c_str(), histoName.str().c_str(), 2400, 0., 0., 2400, 0., 0.);

    histoName.str(""); histoName << "Global_Position_XY_Pixel" << tag.c_str() <<  id;
    local_histos.globalPosXY[1] = td.make< TH2D >(histoName.str().c_str(), histoName.str().c_str(), 2400, 0., 0., 2400, 0., 0.); 

    histoName.str(""); histoName << "Global_Position_XY_Strip" << tag.c_str() <<  id;
    local_histos.globalPosXY[2] = td.make< TH2D >(histoName.str().c_str(), histoName.str().c_str(), 2400, 0., 0., 2400, 0., 0.); 

    /*
     * Delta positions with SimHits
     */

    histoName.str(""); histoName << "Delta_X_RecHit_SimHits_Mixed" << tag.c_str() <<  id;
    local_histos.deltaXRecHitsSimHits[0] = td.make< TH1F >(histoName.str().c_str(), histoName.str().c_str(), 100, -0.02, 0.02);

    histoName.str(""); histoName << "Delta_X_RecHit_SimHits_Pixel" << tag.c_str() <<  id;
    local_histos.deltaXRecHitsSimHits[1] = td.make< TH1F >(histoName.str().c_str(), histoName.str().c_str(), 100, -0.02, 0.02);

    histoName.str(""); histoName << "Delta_X_RecHit_SimHits_Strip" << tag.c_str() <<  id;
    local_histos.deltaXRecHitsSimHits[2] = td.make< TH1F >(histoName.str().c_str(), histoName.str().c_str(), 100, -0.02, 0.02);

    histoName.str(""); histoName << "Delta_Y_RecHit_SimHits_Mixed" << tag.c_str() <<  id;
    local_histos.deltaYRecHitsSimHits[0] = td.make< TH1F >(histoName.str().c_str(), histoName.str().c_str(), 100, 0., 0.);

    histoName.str(""); histoName << "Delta_Y_RecHit_SimHits_Pixel" << tag.c_str() <<  id;
    local_histos.deltaYRecHitsSimHits[1] = td.make< TH1F >(histoName.str().c_str(), histoName.str().c_str(), 100, -0.2, 0.2);

    histoName.str(""); histoName << "Delta_Y_RecHit_SimHits_Strip" << tag.c_str() <<  id;
    local_histos.deltaYRecHitsSimHits[2] = td.make< TH1F >(histoName.str().c_str(), histoName.str().c_str(), 100, 0., 0.);

    /*
     * Delta position with simHits for primary tracks only
     */

    histoName.str(""); histoName << "Delta_X_RecHit_SimHits_Mixed_P" << tag.c_str() <<  id;
    local_histos.deltaXRecHitsSimHits_P[0] = td.make< TH1F >(histoName.str().c_str(), histoName.str().c_str(), 100, -0.02, 0.02);

    histoName.str(""); histoName << "Delta_X_RecHit_SimHits_Pixel_P" << tag.c_str() <<  id;
    local_histos.deltaXRecHitsSimHits_P[1] = td.make< TH1F >(histoName.str().c_str(), histoName.str().c_str(), 100, -0.02, 0.02);

    histoName.str(""); histoName << "Delta_X_RecHit_SimHits_Strip_P" << tag.c_str() <<  id;
    local_histos.deltaXRecHitsSimHits_P[2] = td.make< TH1F >(histoName.str().c_str(), histoName.str().c_str(), 100, -0.02, 0.02);

    histoName.str(""); histoName << "Delta_Y_RecHit_SimHits_Mixed_P" << tag.c_str() <<  id;
    local_histos.deltaYRecHitsSimHits_P[0] = td.make< TH1F >(histoName.str().c_str(), histoName.str().c_str(), 100, 0., 0.);

    histoName.str(""); histoName << "Delta_Y_RecHit_SimHits_Pixel_P" << tag.c_str() <<  id;
    local_histos.deltaYRecHitsSimHits_P[1] = td.make< TH1F >(histoName.str().c_str(), histoName.str().c_str(), 100, -0.2, 0.2);

    histoName.str(""); histoName << "Delta_Y_RecHit_SimHits_Strip_P" << tag.c_str() <<  id;
    local_histos.deltaYRecHitsSimHits_P[2] = td.make< TH1F >(histoName.str().c_str(), histoName.str().c_str(), 100, 0., 0.);

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

    std::pair< std::map< unsigned int, RecHitHistos >::iterator, bool > insertedIt(histograms_.insert(std::make_pair(ival, local_histos)));
    fs->file().cd("/");

    return insertedIt.first;
}


unsigned int Phase2TrackerRecHitsValidation::getSimTrackId(const edm::Handle< edm::DetSetVector< PixelDigiSimLink > >& pixelSimLinks, const DetId& detId, unsigned int channel) {
    edm::DetSetVector< PixelDigiSimLink >::const_iterator DSViter(pixelSimLinks->find(detId));
    if (DSViter == pixelSimLinks->end()) return 0;
    for (edm::DetSet< PixelDigiSimLink >::const_iterator it = DSViter->data.begin(); it != DSViter->data.end(); ++it) {
        if (channel == it->channel()) return it->SimTrackId();
    }
    return 0;
}


DEFINE_FWK_MODULE(Phase2TrackerRecHitsValidation);
