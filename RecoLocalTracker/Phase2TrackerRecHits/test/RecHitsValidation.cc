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
#include "DataFormats/Phase2TrackerRecHit/interface/Phase2TrackerRecHit1D.h"

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
};

class Phase2TrackerRecHitsValidation : public edm::EDAnalyzer {

    public:

        explicit Phase2TrackerRecHitsValidation(const edm::ParameterSet&);
        ~Phase2TrackerRecHitsValidation();
        void beginJob() override;
        void endJob() override;
        void analyze(const edm::Event&, const edm::EventSetup&) override;

    private:

        std::map< unsigned int, RecHitHistos >::iterator createLayerHistograms(unsigned int);
        unsigned int getLayerNumber(const DetId&, const TrackerTopology*);

        edm::EDGetTokenT< Phase2TrackerRecHit1DCollectionNew > tokenRecHits_;
        
        TH2D* trackerLayout_;
        TH2D* trackerLayoutXY_;
        TH2D* trackerLayoutXYBar_;
        TH2D* trackerLayoutXYEC_;

        std::map< unsigned int, RecHitHistos > histograms_;

};

Phase2TrackerRecHitsValidation::Phase2TrackerRecHitsValidation(const edm::ParameterSet& conf) :
    tokenRecHits_(consumes< Phase2TrackerRecHit1DCollectionNew >(conf.getParameter<edm::InputTag>("src"))) {
        std::cout << "RecHits Validation" << std::endl;
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

    std::cout << "PROCESSING RECHIT EVENT" << std::endl;

    /*
     * Get the needed objects
     */

    // Get the RecHitss
    edm::Handle< Phase2TrackerRecHit1DCollectionNew > rechits;
    event.getByToken(tokenRecHits_, rechits);

    // Get the geometry
    edm::ESHandle< TrackerGeometry > geomHandle;
    eventSetup.get< TrackerDigiGeometryRecord >().get(geomHandle);
    const TrackerGeometry* tkGeom = &(*geomHandle);

    edm::ESHandle< TrackerTopology > tTopoHandle;
    eventSetup.get< IdealGeometryRecord >().get(tTopoHandle);
    const TrackerTopology* tTopo = tTopoHandle.product();

    /*
     * Validation   
     */

    // Loop over modules
    for (Phase2TrackerRecHit1DCollectionNew::const_iterator DSViter = rechits->begin(); DSViter != rechits->end(); ++DSViter) {

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
        std::map< unsigned int, RecHitHistos >::iterator histogramLayer(histograms_.find(layer));
        if (histogramLayer == histograms_.end()) histogramLayer = createLayerHistograms(layer);

        // Number of clusters
        unsigned int nRecHitsPixel(0), nRecHitsStrip(0);

        // Loop over the clusters in the detector unit
        for (edmNew::DetSet< Phase2TrackerRecHit1D >::const_iterator rechitIt = DSViter->begin(); rechitIt != DSViter->end(); ++rechitIt) {

            /*
             * Cluster related variables
             */

            //LocalPoint localPosClu = rechitIt->localPosition();
            LocalPoint localPosClu(1, 2, 3);
            //Global3DPoint globalPosClu = geomDetUnit->surface().toGlobal(localPosClu);
            Global3DPoint globalPosClu(0, 0, 0); 

            // Fill the position histograms
            trackerLayout_->Fill(globalPosClu.z(), globalPosClu.perp());
            trackerLayoutXY_->Fill(globalPosClu.x(), globalPosClu.y());
            if (layer < 100) trackerLayoutXYBar_->Fill(globalPosClu.x(), globalPosClu.y());
            else trackerLayoutXYEC_->Fill(globalPosClu.x(), globalPosClu.y());

            histogramLayer->second.localPosXY[0]->Fill(localPosClu.x(), localPosClu.y());
            histogramLayer->second.globalPosXY[0]->Fill(globalPosClu.z(), globalPosClu.perp());

            // Pixel module
            if (topol.ncolumns() == 32) {
                histogramLayer->second.localPosXY[1]->Fill(localPosClu.x(), localPosClu.y());
                histogramLayer->second.globalPosXY[1]->Fill(globalPosClu.z(), globalPosClu.perp());
                ++nRecHitsPixel;
            }
            // Strip module
            else if (topol.ncolumns() == 2) {
                histogramLayer->second.localPosXY[2]->Fill(localPosClu.x(), localPosClu.y());
                histogramLayer->second.globalPosXY[2]->Fill(globalPosClu.z(), globalPosClu.perp());
                ++nRecHitsStrip;
            }
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
        fname1 << "EndCap_Side_" << side;
        fname2 << "Disc_" << id;
        tag = "_disc_";
    }

    TFileDirectory td1 = fs->mkdir(fname1.str().c_str());
    TFileDirectory td = td1.mkdir(fname2.str().c_str());

    RecHitHistos local_histos;

    std::ostringstream histoName;

    /*
     * Number of clusters
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
     * End
     */

    std::pair< std::map< unsigned int, RecHitHistos >::iterator, bool > insertedIt(histograms_.insert(std::make_pair(ival, local_histos)));
    fs->file().cd("/");

    return insertedIt.first;
}

unsigned int Phase2TrackerRecHitsValidation::getLayerNumber(const DetId& detid, const TrackerTopology* topo) {
    if (detid.det() == DetId::Tracker) {
        if (detid.subdetId() == PixelSubdetector::PixelBarrel) return (topo->pxbLayer(detid));
        else if (detid.subdetId() == PixelSubdetector::PixelEndcap) return (100 * topo->pxfSide(detid) + topo->pxfDisk(detid));
        else return 999;
    }
    return 999;
}

DEFINE_FWK_MODULE(Phase2TrackerRecHitsValidation);
