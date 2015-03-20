// ----------------------------------------------------------------------
// PCCNTupler
// ---------

#include <memory>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <bitset>

#include "PCCNTupler.h"

#include "CondFormats/Alignment/interface/Definitions.h"
#include "CondFormats/RunInfo/interface/RunInfo.h"

#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include <DataFormats/VertexReco/interface/VertexFwd.h>

#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"


#include <TROOT.h>
#include <TSystem.h>
#include <TTree.h>
#include <TFile.h>
#include <TH1F.h>
#include <TH2F.h>

using namespace std;
using namespace edm;
using namespace reco;

// ----------------------------------------------------------------------
PCCNTupler::PCCNTupler(edm::ParameterSet const& iConfig): 
    fVerbose(iConfig.getUntrackedParameter<int>("verbose", 0)),
    fPixelClusterLabel(iConfig.getUntrackedParameter<InputTag>("pixelClusterLabel", edm::InputTag("siPixelClusters"))), 
    fPixelRecHitLabel(iConfig.getUntrackedParameter<InputTag>("pixelRecHitLabel", edm::InputTag("siPixelRecHits"))), 
    fHLTProcessName(iConfig.getUntrackedParameter<string>("HLTProcessName"))
{
    cout << "----------------------------------------------------------------------" << endl;
    cout << "--- PCCNTupler constructor" << endl;

    nPrint = 0;
    edm::Service<TFileService> fs;

    includeVertexInformation = true;
    includeTracks = true;
    includePixels = true;

    tree = fs->make<TTree>("tree","Pixel Cluster Counters");
    tree->Branch("runNo",&runNo,"runNo/I");
    tree->Branch("LSNo",&LSNo,"LSNo/I");

    if(includeVertexInformation){
        tree->Branch("nGoodVtx","map<int,int>",&nGoodVtx); 
        //tree->Branch("x_Vertex_um",&xV,"xV/F"); 
        //tree->Branch("y_Vertex_um",&yV,"yV/F"); 
        //tree->Branch("z_Vertex_mm",&zV,"zV/F");
        tree->Branch("num_Trks",&nTrk,"nTrk/I");
        //tree->Branch("chi_squared",&chi2,"chi2/F");
        //tree->Branch("ndof",&ndof,"ndof/I");
    }

    if(includeTracks){
        tree->Branch("n1GeV",&n1GeV,"n1GeV/I");
        tree->Branch("n2GeV",&n2GeV,"n2GeV/I");
    }

    if(includePixels){
        tree->Branch("BXNo","map<int,int>",&BXNo);
        tree->Branch("nPixelClusters","map<int,int>",&nPixelClusters);
        tree->Branch("nClusters","map<int,int>",&nClusters);
        tree->Branch("layers","map<int,int>",&layers);
        // dead modules
        nDeadModules = 6;
        nDeadPrint = 0;
        deadModules[0] = 302125076; 
        deadModules[1] = 302125060;
        deadModules[2] = 302197516;
        deadModules[3] = 344019460;
        deadModules[4] = 344019464;
        deadModules[5] = 344019468;  
    }
}

// ----------------------------------------------------------------------
PCCNTupler::~PCCNTupler() { }  

// ----------------------------------------------------------------------
void PCCNTupler::endJob() { 
    tree->Fill();
    cout << "==>PCCNTupler> Succesfully gracefully ended job" << endl;
}

// ----------------------------------------------------------------------
void PCCNTupler::beginJob() {
  
}


// ----------------------------------------------------------------------
void  PCCNTupler::beginRun(const Run &run, const EventSetup &iSetup) {
    bool hasChanged;
    fValidHLTConfig = fHltConfig.init(run,iSetup,fHLTProcessName,hasChanged);
}

// ----------------------------------------------------------------------
void PCCNTupler::endRun(Run const&run, EventSetup const&iSetup) {
    fValidHLTConfig = false;
} 



// ----------------------------------------------------------------------
void PCCNTupler::analyze(const edm::Event& iEvent, 
            const edm::EventSetup& iSetup)  {

    using namespace edm;
    using reco::TrackCollection;
    using reco::VertexCollection;

    bool qPrint = (nPrint++ % 10000 == 1);

    bool sameLS = (LSNo==(int)iEvent.getLuminosityBlock().luminosityBlock());
    bool firstEvent = (LSNo==-99);
    
    // when arriving at the new LS the tree must be filled and 
    // branches must be reset for 
    if(!sameLS){ 
        std::cout<<"last LS, this LS "<<LSNo<<"  "<<iEvent.getLuminosityBlock().luminosityBlock()<<std::endl; 
        if(!firstEvent) tree->Fill();
        nVtx = 0;
        nTrk = 0;
        nPixelClusters.clear();
        nClusters.clear();
        layers.clear();
        BXNo.clear();
        nGoodVtx.clear();
    }

    // Get the Run, Lumi Section, and Event numbers, etc.
    runNo = iEvent.id().run();
    LSNo = iEvent.getLuminosityBlock().luminosityBlock();
    int eventNo = iEvent.id().event();
    timeStamp = iEvent.time().unixTime();
    
    if((BXNo.count(iEvent.bunchCrossing())==0||nGoodVtx.count(iEvent.bunchCrossing())==0) && !(BXNo.count(iEvent.bunchCrossing())==0&&nGoodVtx.count(iEvent.bunchCrossing())==0)){
        std::cout<<"BXNo and nGoodVtx should have the same keys but DO NOT!!!"<<std::endl;
    }
    
    if(BXNo.count(iEvent.bunchCrossing())==0){
        BXNo[iEvent.bunchCrossing()]=0;
    }

    if(nGoodVtx.count(iEvent.bunchCrossing())==0){
        nGoodVtx[iEvent.bunchCrossing()]=0;
    }

    BXNo[iEvent.bunchCrossing()]=BXNo[iEvent.bunchCrossing()]+1;
    // add the vertex information

    xV = -9999.; yV = -9999.; zV = -9999.;  chi2 = -9999.;  ndof = -9999;
   
    if(includeVertexInformation){
        edm::Handle<reco::VertexCollection> recVtxs;
        iEvent.getByLabel("offlinePrimaryVertices",recVtxs);
   
        if(recVtxs.isValid()){
            for(reco::VertexCollection::const_iterator v=recVtxs->begin(); v!=recVtxs->end(); ++v){
                int nnTrk = v->tracksSize();
                if(qPrint) { std::cout << "\n ****Event: " << eventNo << " Vertex nnTrk=" << nnTrk << std::endl; }
                int nd = (int)v->ndof();
                if(nd > 4 && v->isValid() && (v->isFake() == 0)){
                    nVtx++;
                    if(nnTrk > 0){
                        nGoodVtx[iEvent.bunchCrossing()]=nGoodVtx[iEvent.bunchCrossing()]+1;
                        if(nnTrk > nTrk){
                            nTrk = nnTrk;                   
                            //FIXME why are we multiplying by 10000 or 10?
                            xV = 10000.*v->x(); 
                            yV = 10000.*v->y(); 
                            zV = 10.*v->z();
                            chi2 = v->chi2();
                            ndof = (int)v->ndof();
                            if(qPrint){ 
                                std::cout 
                                << " nTrk=" << nTrk 
                                << " xV=" <<  xV  
                                << " yV=" << yV 
                                << " zV=" << zV
                                << " chi2=" << chi2 
                                << " ndof=" << ndof << std::endl;
                            } 
                        }
                    }
                }
            }
        }
    }

    // get the generalTracks information
    // for now just do something very simple and count the number of
    // of tracks with pT > 0.25 GeV
    using reco::TrackCollection;

    if(includeTracks){
        edm::Handle<TrackCollection> tracks;
        iEvent.getByLabel("generalTracks",tracks);

        nGeneralTracks = 0;
        n1GeV = 0;
        n2GeV = 0;
        for(TrackCollection::const_iterator itTrack = tracks->begin(); itTrack != tracks->end(); ++itTrack) {
            double pT = itTrack->pt();
            if(pT > 0.25)nGeneralTracks++;
            double eta = itTrack->eta();
            if(eta > -0.8 && eta < 0.8){
                if(pT > 1.0)n1GeV++;
                if(pT > 2.0)n2GeV++;
            }
        }
    }


    // -- Does this belong into beginJob()?
    ESHandle<TrackerGeometry> TG;
    iSetup.get<TrackerDigiGeometryRecord>().get(TG);

    // -- FED
    edm::eventsetup::EventSetupRecordKey recordKey(edm::eventsetup::EventSetupRecordKey::TypeTag::findType("RunInfoRcd"));
    if (recordKey.type() == edm::eventsetup::EventSetupRecordKey::TypeTag()) {
        //record not found
        cout << "Record \"RunInfoRcd" << "\" does not exist " << endl;
        fFED1 = 0; 
        fFED2 = (0x1 << 12) ; 
    } else {
        edm::ESHandle<RunInfo> runInfoHandle;
        iSetup.get<RunInfoRcd>().get(runInfoHandle);
    }
    // -- Pixel cluster
    edm::Handle< edmNew::DetSetVector<SiPixelCluster> > hClusterColl;
    iEvent.getByLabel(fPixelClusterLabel, hClusterColl);
    const edmNew::DetSetVector<SiPixelCluster> clustColl = *(hClusterColl.product());
    // -- Pixel RecHit
    edm::Handle<SiPixelRecHitCollection> hRecHitColl;
    iEvent.getByLabel(fPixelRecHitLabel, hRecHitColl); 

    // ----------------------------------------------------------------------
    // -- Clusters without tracks

    for (TrackerGeometry::DetContainer::const_iterator it = TG->dets().begin(); it != TG->dets().end(); it++){
        if (dynamic_cast<PixelGeomDetUnit*>((*it)) != 0){ 
            DetId detId = (*it)->geographicalId();


            // -- clusters on this det
            edmNew::DetSetVector<SiPixelCluster>::const_iterator isearch = clustColl.find(detId);
            // -- rechits on this det
            SiPixelRecHitCollection::const_iterator dsmatch = hRecHitColl->find(detId);
            SiPixelRecHitCollection::DetSet rhRange;
            if (dsmatch != hRecHitColl->end()) { 
                rhRange = *dsmatch;
            }
            if (isearch != clustColl.end()) {  // Not an empty iterator
                edmNew::DetSet<SiPixelCluster>::const_iterator  di;
                for (di = isearch->begin(); di != isearch->end(); ++di) {
                    if(nPixelClusters.count(detId())==0){
                        nPixelClusters[detId()]=0;
                    }
                    nPixelClusters[detId()] = nPixelClusters[detId()]+1;
                }

                int nCluster = isearch->size();
                if(nClusters.count(detId())==0){
                    nClusters[detId()]=0;
                }
                nClusters[detId()] += nCluster;

                if (detId.subdetId() == PixelSubdetector::PixelBarrel) {
                    PixelBarrelName detName = PixelBarrelName(detId);
                    int layer = detName.layerName();
                    if(layers.count(detId())==0){
                        layers[detId()]=layer;
                    }
                } else {
                    assert(detId.subdetId() == PixelSubdetector::PixelEndcap);
                    PixelEndcapName detName = PixelEndcapName(detId);
                    int disk = detName.diskName();
                    if(layers.count(detId())==0){
                        layers[detId()]=disk+3;
                    }
                }
            }
        }
    }
} 

// define this as a plug-in
DEFINE_FWK_MODULE(PCCNTupler);


