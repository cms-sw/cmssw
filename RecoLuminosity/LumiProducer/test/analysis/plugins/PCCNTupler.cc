// ----------------------------------------------------------------------
// PCCNTupler
// ---------
// Summary: The full pixel information, including tracks and cross references
//          A lot has been copied from 
//            DPGAnalysis/SiPixelTools/plugins/PixelNtuplizer_RealData.cc
//            DQM/SiPixelMonitorTrack/src/SiPixelTrackResidualSource.cc
//
// ----------------------------------------------------------------------
// Send all questions, wishes and complaints to the 
//
// Author:  Urs Langenegger (PSI)
// ----------------------------------------------------------------------

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

  tree = fs->make<TTree>("tree","HF Info");
  tree->Branch("runNo",&runNo,"runNo/I");
  tree->Branch("LSNo",&LSNo,"LSNo/I");
  tree->Branch("eventNo",&eventNo,"eventNo/I");
  tree->Branch("BXNo",&BXNo,"BXNo/I");
  tree->Branch("timeStamp",&timeStamp,"timeStamp/i");  
  tree->Branch("orbitN",&orbitN,"orbitN/i");  
  
  if(includeVertexInformation){
     tree->Branch("nGoodVtx",&nGoodVtx,"nGoodVtx/I"); 
     tree->Branch("x_Vertex_um",&xV,"xV/F"); 
     tree->Branch("y_Vertex_um",&yV,"yV/F"); 
     tree->Branch("z_Vertex_mm",&zV,"zV/F");
     tree->Branch("num_Trks",&nTrk,"nTrk/I");
     tree->Branch("chi_squared",&chi2,"chi2/F");
     tree->Branch("ndof",&ndof,"ndof/I");
  }

  if(includeTracks){
     tree->Branch("n1GeV",&n1GeV,"n1GeV/I");
     tree->Branch("n2GeV",&n2GeV,"n2GeV/I");
  }

  if(includePixels){
     tree->Branch("nPixelClusters",&nPixelClusters,"nPixelClusters/I");
     tree->Branch("nB1",&nB1,"nB1/I");
     tree->Branch("nB2",&nB2,"nB2/I");
     tree->Branch("nB3",&nB3,"nB3/I");
     tree->Branch("nF1",&nF1,"nF1/I");
     tree->Branch("nF2",&nF2,"nF2/I");
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

   bool qPrint = (nPrint++ % 200 == 1);

   // Get the Run, Lumi Section, and Event numbers, etc.
   runNo = iEvent.id().run();
   LSNo = iEvent.getLuminosityBlock().luminosityBlock();
   eventNo = iEvent.id().event();
   BXNo = iEvent.bunchCrossing();
   timeStamp = iEvent.time().unixTime();
   orbitN = iEvent.orbitNumber();

   // add the vertex information

   nVtx = 0;
   nGoodVtx = 0;
   nTrk = 0;
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
		  nGoodVtx++;
                  if(nnTrk > nTrk){
                     nTrk = nnTrk;                   
		     xV = 10000.*v->x(); yV = 10000.*v->y(); zV = 10.*v->z();
		     chi2 = v->chi2();
		     ndof = (int)v->ndof();
		     if(qPrint){ 
			std::cout << "   Good nVtx=" << nVtx << " nGoodVtx=" << nGoodVtx << " nTrk=" << nTrk << " xV=" <<  xV  << " yV=" << yV << " zV=" << zV
				  << " chi2=" << chi2 << " ndof=" << ndof << std::endl;
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

  nPixelClusters = 0;
  nB1 = 0;
  nB2 = 0;
  nB3 = 0;
  nF1 = 0;
  nF2 = 0;
  int nC2 = 0;

  for (TrackerGeometry::DetContainer::const_iterator it = TG->dets().begin(); it != TG->dets().end(); it++){
    if (dynamic_cast<PixelGeomDetUnit*>((*it)) != 0){ 
      DetId detId = (*it)->geographicalId();

      // Skip this module if it's on the list of modules to be ignored.
      bool qDead = false;
      for(int i=0; i<nDeadModules; i++){
	 if(detId() == deadModules[i]){
	    if(nDeadPrint++ < 10)cout << " deadModule found detID=" << detId() << endl;
	    qDead = true;
	    break;
	 }
      }
      if(qDead)continue;

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
	  nPixelClusters++;
	}

        int nCluster = isearch->size();
        nC2 += nCluster;

	if (detId.subdetId() == PixelSubdetector::PixelBarrel) {
           PixelBarrelName detName = PixelBarrelName(detId);
//           PixelBarrelName::Shell shell = detName.shell();
           int layer = detName.layerName();
	   if(layer == 1)nB1 += nCluster;
	   if(layer == 2)nB2 += nCluster;
	   if(layer == 3)nB3 += nCluster;
	} else {
           // DEBUG DEBUG DEBUG
           assert(detId.subdetId() == PixelSubdetector::PixelEndcap);
           // DEBUG DEBUG DEBUG end
           PixelEndcapName detName = PixelEndcapName(detId);
//           PixelEndcapName::HalfCylinder halfCylinder = detName.halfCylinder();
           int disk = detName.diskName();
	   if(disk == 1)nF1 += nCluster;
	   if(disk == 2)nF2 += nCluster;
	}
      }
    }
  }
  if(qPrint)cout << "nPixelClusters=" << nPixelClusters << " nC2=" << nC2 << 
	       " nB1=" << nB1 << " nB2=" << nB2 << " nB3=" << nB3 << " nF1=" << nF1 << " nF2=" << nF2 << endl;

  tree->Fill();
} 

// define this as a plug-in
DEFINE_FWK_MODULE(PCCNTupler);


