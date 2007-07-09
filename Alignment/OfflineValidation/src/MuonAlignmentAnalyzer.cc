/** \class MuonAlignmentAnalyzer
 *  MuonAlignment offline Monitor Analyzer 
 *  Makes histograms of high level Muon objects/quantities
 *  for Alignment Scenarios/DB comparison
 *
 *  $Date: 2007/07/09 15:54:57 $
 *  $Revision: 1.2 $
 *  \author J. Fernandez - IFCA (CSIC-UC) <Javier.Fernandez@cern.ch>
 */

#include "Alignment/OfflineValidation/interface/MuonAlignmentAnalyzer.h"

// Collaborating Class Header
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"

#include "SimDataFormats/Track/interface/SimTrackContainer.h"

#include "DataFormats/TrackingRecHit/interface/RecSegment.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4D.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "DataFormats/CSCRecHit/interface/CSCSegment.h"
#include "DataFormats/CSCRecHit/interface/CSCSegmentCollection.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"

#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TLorentzVector.h"
#include <stdio.h>
#include <fstream>

using namespace std;
using namespace edm;

/// Constructor
MuonAlignmentAnalyzer::MuonAlignmentAnalyzer(const ParameterSet& pset){
  theSTAMuonTag = pset.getParameter<edm::InputTag>("StandAloneTrackCollectionTag");
  theGLBMuonTag = pset.getParameter<edm::InputTag>("GlobalMuonTrackCollectionTag");

  theRecHits4DLabelDT = pset.getUntrackedParameter<string>("RecHits4DDTCollectionLabel");
  theRecHits2DLabelCSC = pset.getUntrackedParameter<string>("RecHits2DCSCCollectionLabel");
  
  theRootFileName = pset.getUntrackedParameter<string>("rootFileName");

  theDataType = pset.getUntrackedParameter<string>("DataType");

  doSAplots = pset.getUntrackedParameter<bool>("doSAplots");
  doGBplots = pset.getUntrackedParameter<bool>("doGBplots");
  doResplots = pset.getUntrackedParameter<bool>("doResplots");
 
  if(theDataType != "RealData" && theDataType != "SimData")
    cout<<"Error in Data Type!!"<<endl;

  numberOfSimTracks=0;
  numberOfSARecTracks=0;
  numberOfGBRecTracks=0;
}

/// Destructor
MuonAlignmentAnalyzer::~MuonAlignmentAnalyzer(){
}

void MuonAlignmentAnalyzer::beginJob(const EventSetup& eventSetup){
  // Create the root file
  theFile = new TFile(theRootFileName.c_str(), "RECREATE");
  theFile->cd();

// Define and book histograms for SA and GB muon quantities/objects

//# muons per event
  hNmuonsGB = new TH1F("GBNmuons","Nmuons",10,0,10);
  hNmuonsSA = new TH1F("SANmuons","Nmuons",10,0,10);
  hNmuonsSim = new TH1F("SimNmuons","Nmuons",10,0,10);

// ############     pt    ####################
  hGBPTRec = new TH1F("GBpTRec","p_{T}^{rec}",300,0,300);
  hSAPTRec = new TH1F("SApTRec","p_{T}^{rec}",300,0,300);
  hPTSim = new TH1F("pTSim","p_{T}^{gen} ",300,0,300);

  hGBPTvsEta = new TH2F("GBPTvsEta","p_{T}^{rec} VS #eta",100,-2.5,2.5,300,0,300);
  hGBPTvsPhi = new TH2F("GBPTvsPhi","p_{T}^{rec} VS #phi",100,-6,6,300,0,300);

  hSAPTvsEta = new TH2F("SAPTvsEta","p_{T}^{rec} VS #eta",100,-2.5,2.5,300,0,300);
  hSAPTvsPhi = new TH2F("SAPTvsPhi","p_{T}^{rec} VS #phi",100,-6,6,300,0,300);

  hSimPTvsEta = new TH2F("SimPTvsEta","p_{T}^{gen} VS #eta",100,-2.5,2.5,300,0,300);
  hSimPTvsPhi = new TH2F("SimPTvsPhi","p_{T}^{gen} VS #phi",100,-6,6,300,0,300);

//pT rec - pT gen
  hSAPTDiff = new TH1F("SApTDiff","p_{T}^{rec} - p_{T}^{gen} ",250,-120,120);
  hGBPTDiff = new TH1F("GBpTDiff","p_{T}^{rec} - p_{T}^{gen} ",250,-120,120);

  hSAPTDiffvsEta = new TH2F("SAPTDiffvsEta","p_{T}^{rec} - p_{T}^{gen} VS #eta",100,-2.5,2.5,250,-120,120);
  hSAPTDiffvsPhi = new TH2F("SAPTDiffvsPhi","p_{T}^{rec} - p_{T}^{gen} VS #phi",100,-6,6,250,-120,120);

  hGBPTDiffvsEta = new TH2F("GBPTDiffvsEta","p_{T}^{rec} - p_{T}^{gen} VS #eta",100,-2.5,2.5,250,-120,120);
  hGBPTDiffvsPhi = new TH2F("GBPTDiffvsPhi","p_{T}^{rec} - p_{T}^{gen} VS #phi",100,-6,6,250,-120,120);

// pT resolution
  hSAPTres = new TH1F("SApTRes","pT Resolution",100,-2,2);
  hSAinvPTres = new TH1F("SAinvPTRes","1/pT Resolution",100,-2,2);
  hGBPTres = new TH1F("GBpTRes","pT Resolution",100,-2,2);
  hGBinvPTres = new TH1F("GBinvPTRes","1/pT Resolution",100,-2,2);
  
  hGBinvPTvsEta = new TH2F("GBinvPTvsEta","1/p_{T}^{rec} VS #eta",100,-2.5,2.5,100,-2,2);
  hGBinvPTvsPhi = new TH2F("GBinvPTvsPhi","1/p_{T}^{rec} VS #phi",100,-6,6,100,-2,2);

  hSAinvPTvsEta = new TH2F("SAinvPTvsEta","1/p_{T}^{rec} VS #eta",100,-2.5,2.5,100,-2,2);
  hSAinvPTvsPhi = new TH2F("SAinvPTvsPhi","1/p_{T}^{rec} VS #phi",100,-6,6,100,-2,2);


// #######Invariant mass #############
// invariant mass for dimuons
  hGBInvM = new TH1F("GBInvM","M_{inv}^{rec}",200,0,200);
  hSAInvM = new TH1F("SAInvM","M_{inv}^{rec}",200,0,200);
  hSimInvM = new TH1F("SimInvM","M_{inv}^{gen} ",200,0,200);

// Invariant Mass distributions in Barrel (eta<1.04) region
  hGBInvM_Barrel = new TH1F("GBInvM_Barrel","M_{inv}^{rec}",200,0,200);
  hSAInvM_Barrel = new TH1F("SAInvM_Barrel","M_{inv}^{rec}",200,0,200);
  hSimInvM_Barrel = new TH1F("SimInvM_Barrel","M_{inv}^{rec}",200,0,200);

 // Chi2 of Track
  hGBChi2 = new TH1F("GBChi2","Chi2",200,0,200);
  hSAChi2 = new TH1F("SAChi2","Chi2",200,0,200);

// All DT and CSC chambers 
  hResidualRPhiDT = new TH1F("hResidualRPhiDT","hResidualRPhiDT",200,-10,10);
  hResidualPhiDT = new TH1F("hResidualPhiDT","hResidualPhiDT",100,-1,1);
  hResidualZDT = new TH1F("hResidualZDT","hResidualZDT",200,-10,10);
  hResidualRPhiCSC = new TH1F("hResidualRPhiCSC","hResidualRPhiCSC",200,-10,10);
  hResidualPhiCSC = new TH1F("hResidualPhiCSC","hResidualPhiCSC",100,-1,1);
  hResidualZCSC = new TH1F("hResidualZCSC","hResidualZCSC",200,-10,10);

}

void MuonAlignmentAnalyzer::endJob(){
  // Write the histos to file
  theFile->cd();

  if(theDataType == "SimData"){
    cout << endl << endl << "Number of Sim tracks: " << numberOfSimTracks << endl;
  hNmuonsSim->Write();
  hPTSim->Write();
  hSimInvM->Write();
  hSimInvM_Barrel->Write();
  hSimPTvsEta->Write();
  hSimPTvsPhi->Write();
  }

  if(doSAplots){
    cout << "Number of SA Reco tracks: " << numberOfSARecTracks << endl << endl;
  hNmuonsSA->Write();
  if(theDataType == "SimData"){
  hSAPTres->Write();
  hSAinvPTres->Write();
  hSAinvPTvsEta->Write();
  hSAinvPTvsPhi->Write();
  hSAPTDiff->Write();
  hSAPTDiffvsEta->Write();
  hSAPTDiffvsPhi->Write();
	}
  hSAPTvsEta->Write();
  hSAPTvsPhi->Write();
  hSAPTRec->Write();
  hSAInvM->Write();
  hSAInvM_Barrel->Write();
  hSAChi2->Write();
	}

  if(doGBplots){
  cout << "Number of GB Reco tracks: " << numberOfGBRecTracks << endl << endl;
  hNmuonsGB->Write();
  if(theDataType == "SimData"){
  hGBPTres->Write();
  hGBPTDiff->Write();
  hGBPTDiffvsEta->Write();
  hGBPTDiffvsPhi->Write();
  hGBinvPTres->Write();
  hGBinvPTvsEta->Write();
  hGBinvPTvsPhi->Write();
	}
  hGBPTRec->Write();
  hGBPTvsEta->Write();
  hGBPTvsPhi->Write();
  hGBInvM->Write();
  hGBInvM_Barrel->Write();
  hGBChi2->Write();
   }

  if(doResplots){
    for(std::vector<TH1F *>::iterator myIt = unitsRPhi.begin(); myIt != unitsRPhi.end(); myIt++) {
      (*myIt)->Write();
    } 
    for(std::vector<TH1F *>::iterator myIt = unitsPhi.begin(); myIt != unitsPhi.end(); myIt++) {
      (*myIt)->Write();
    } 
    for(std::vector<TH1F *>::iterator myIt = unitsZ.begin(); myIt != unitsZ.end(); myIt++) {
      (*myIt)->Write();
    } 

  hResidualRPhiDT->Write(); 
  hResidualPhiDT->Write(); 
  hResidualZDT->Write(); 
  hResidualRPhiCSC->Write(); 
  hResidualPhiCSC->Write(); 
  hResidualZCSC->Write(); 
}

  theFile->Close();
}
 

void MuonAlignmentAnalyzer::analyze(const Event & event, const EventSetup& eventSetup){
  
  cout << "Run: " << event.id().run() << " Event: " << event.id().event() << endl;
  
  ESHandle<MagneticField> theMGField;
  eventSetup.get<IdealMagneticFieldRecord>().get(theMGField);

  ESHandle<GlobalTrackingGeometry> theTrackingGeometry;
  eventSetup.get<GlobalTrackingGeometryRecord>().get(theTrackingGeometry);
  
  GlobalVector p1,p2;
  std::vector< double > simPar[3] ; //pt,eta,phi
  int i=0;
  

// ######### if data= MC, do Simulation Plots#####
  if(theDataType == "SimData"){
  double simEta=0;
  double simPt=0;
  double simPhi=0;

  // Get the SimTrack collection from the event
    Handle<SimTrackContainer> simTracks;
    event.getByLabel("g4SimHits",simTracks);
  

    SimTrackContainer::const_iterator simTrack;

    cout<<"Simulated muons: "<<simTracks->size() <<endl;
	i=0;
    for (simTrack = simTracks->begin(); simTrack != simTracks->end(); ++simTrack){
      if (abs((*simTrack).type()) == 13) {
	i++;
//	cout<<"Sim pT: "<<(*simTrack).momentum().perp()<<endl;
	simPt=(*simTrack).momentum().perp();
//	cout<<"Sim Eta: "<<(*simTrack).momentum().eta()<<endl;
	simEta=(*simTrack).momentum().eta();
//	cout<<"Sim Phi: "<<(*simTrack).momentum().phi()<<endl;
	simPhi=(*simTrack).momentum().phi();
	numberOfSimTracks++;
	hPTSim->Fill(simPt);
	hSimPTvsEta->Fill(simEta,simPt);
	hSimPTvsPhi->Fill(simPhi,simPt);

	simPar[0].push_back(simPt);
	simPar[1].push_back(simEta);
	simPar[2].push_back(simPhi);

	
//	Save the muon pair
        if(i==1)  p1=GlobalVector((*simTrack).momentum().x(),(*simTrack).momentum().y(),(*simTrack).momentum().z());
    	if(i==2)  p2=GlobalVector((*simTrack).momentum().x(),(*simTrack).momentum().y(),(*simTrack).momentum().z());
     }    
    }
	hNmuonsSim->Fill(i);

  if(i==2){
  TLorentzVector mu1(p1.x(), p1.y(), p1.z(), p1.mag());
  TLorentzVector mu2(p2.x(), p2.y(), p2.z(), p2.mag());
  TLorentzVector pair = mu1 + mu2;
  double Minv = pair.M();
  hSimInvM->Fill(Minv);
  if(abs(p1.eta())<1.04 && abs(p2.eta())<1.04) hSimInvM_Barrel->Fill(Minv);
  }

    cout << endl;
  } //simData
  
  
// ############ Stand Alone Muon plots ###############  
  if(doSAplots){

  double SArecPt=0.;
  double SAeta=0.;
  double SAphi=0.;

  // Get the RecTrack collection from the event
  Handle<reco::TrackCollection> staTracks;
  event.getByLabel(theSTAMuonTag, staTracks);
  numberOfSARecTracks += staTracks->size();

  reco::TrackCollection::const_iterator staTrack;

  cout<<"SA Reconstructed muons: " << staTracks->size() << endl;
  i=0;
  for (staTrack = staTracks->begin(); staTrack != staTracks->end(); ++staTrack){
//    reco::TransientTrack track(*staTrack,&*theMGField,theTrackingGeometry); 
    i++;
    
    
    SArecPt = (*staTrack).pt();
    hSAPTRec->Fill(SArecPt);
    SAeta = (*staTrack).eta();
    SAphi = (*staTrack).phi();
    hSAChi2->Fill((*staTrack).chi2());

// save the muon pair
    if(i==1)  p1=GlobalVector((*staTrack).momentum().x(),(*staTrack).momentum().y(),(*staTrack).momentum().z());
    if(i==2)  p2=GlobalVector((*staTrack).momentum().x(),(*staTrack).momentum().y(),(*staTrack).momentum().z());

   
    if(SArecPt && theDataType == "SimData"){  
	
     for(unsigned int  iSim = 0; iSim <simPar[0].size(); iSim++) {
     double simPt=simPar[0][iSim];
     if(sqrt((SAeta-simPar[1][iSim])*(SAeta-simPar[1][iSim])+(SAphi-simPar[2][iSim])*(SAphi-simPar[2][iSim]))<0.3){
      hSAPTres->Fill( (SArecPt-simPt)/simPt);

      hSAPTDiff->Fill(SArecPt-simPt);

      hSAPTDiffvsEta->Fill(SAeta,SArecPt-simPt);
      hSAPTDiffvsPhi->Fill(SAphi,SArecPt-simPt);

      hSAinvPTres->Fill( ( 1/SArecPt - 1/simPt)/ (1/simPt));

      hSAinvPTvsEta->Fill(SAeta,( 1/SArecPt - 1/simPt)/ (1/simPt));
      hSAinvPTvsPhi->Fill(SAphi,( 1/SArecPt - 1/simPt)/ (1/simPt));
	}}
    }

      hSAPTvsEta->Fill(SAeta,SArecPt);
      hSAPTvsPhi->Fill(SAphi,SArecPt);
}

	hNmuonsSA->Fill(i);

  if(i==2){
  TLorentzVector mu1(p1.x(), p1.y(), p1.z(), p1.mag());
  TLorentzVector mu2(p2.x(), p2.y(), p2.z(), p2.mag());
  TLorentzVector pair = mu1 + mu2;
  double Minv = pair.M();
  hSAInvM->Fill(Minv);
  if(abs(p1.eta())<1.04 && abs(p2.eta())<1.04) hSAInvM_Barrel->Fill(Minv);
  }

  }//end doSAplots



// ############### Global Muons plots ##########

  if(doGBplots){  
  // Get the RecTrack collection from the event
  Handle<reco::TrackCollection> glbTracks;
  event.getByLabel(theGLBMuonTag, glbTracks);
  numberOfGBRecTracks += glbTracks->size();

  double GBrecPt = 0; 
  double GBeta = 0;
  double GBphi = 0;

  reco::TrackCollection::const_iterator glbTrack;
	i=0;
 	cout <<endl<<endl;  
  cout<<"GB Reconstructed muons: " << glbTracks->size() << endl;

  for (glbTrack = glbTracks->begin(); glbTrack != glbTracks->end(); ++glbTrack){
//    reco::TransientTrack track2(*glbTrack,&*theMGField,theTrackingGeometry); 
    i++;

    GBrecPt = (*glbTrack).pt(); 
    GBeta = (*glbTrack).eta();
    GBphi = (*glbTrack).phi();
    
// save the muon pair
    if(i==1)  p1=GlobalVector((*glbTrack).momentum().x(),(*glbTrack).momentum().y(),(*glbTrack).momentum().z());
    if(i==2)  p2=GlobalVector((*glbTrack).momentum().x(),(*glbTrack).momentum().y(),(*glbTrack).momentum().z());

  
    hGBPTRec->Fill(GBrecPt);
    hGBChi2->Fill((*glbTrack).chi2());
   

    if(GBrecPt && theDataType == "SimData"){  
     for(unsigned int  iSim = 0; iSim <simPar[0].size(); iSim++) {
     double simPt=simPar[0][iSim];
     if(sqrt((GBeta-simPar[1][iSim])*(GBeta-simPar[1][iSim])+(GBphi-simPar[2][iSim])*(GBphi-simPar[2][iSim]))<0.3){
	
      hGBPTres->Fill( (GBrecPt-simPt)/simPt);

      hGBPTDiff->Fill(GBrecPt-simPt);

      hGBPTDiffvsEta->Fill(GBeta,GBrecPt-simPt);
      hGBPTDiffvsPhi->Fill(GBphi,GBrecPt-simPt);

      hGBinvPTres->Fill( ( 1/GBrecPt - 1/simPt)/ (1/simPt));
      hGBinvPTvsEta->Fill(GBeta,( 1/GBrecPt - 1/simPt)/ (1/simPt));
      hGBinvPTvsPhi->Fill(GBphi,( 1/GBrecPt - 1/simPt)/ (1/simPt));
	}}
    } 


      hGBPTvsEta->Fill(GBeta,GBrecPt);
      hGBPTvsPhi->Fill(GBphi,GBrecPt);

  }

	hNmuonsGB->Fill(i);

  if(i==2){
  TLorentzVector mu1(p1.x(), p1.y(), p1.z(), p1.mag());
  TLorentzVector mu2(p2.x(), p2.y(), p2.z(), p2.mag());
  TLorentzVector pair = mu1 + mu2;
  double Minv = pair.M();
  hGBInvM->Fill(Minv);
  if(abs(p1.eta())<1.04 && abs(p2.eta())<1.04)   hGBInvM_Barrel->Fill(Minv);
  }
} //end doGBplots


// ############    Residual plots ###################

 if(doResplots){

  // Get the RecTrack collection from the event
  Handle<reco::TrackCollection> staTracks;
  event.getByLabel(theSTAMuonTag, staTracks);

  // Get the 4D DTSegments
  edm::Handle<DTRecSegment4DCollection> all4DSegmentsDT;
  event.getByLabel(theRecHits4DLabelDT, all4DSegmentsDT);
  DTRecSegment4DCollection::const_iterator segmentDT;

  // Get the 2D CSCSegments
  edm::Handle<CSCSegmentCollection> all2DSegmentsCSC;
  event.getByLabel(theRecHits2DLabelCSC, all2DSegmentsCSC);
  CSCSegmentCollection::const_iterator segmentCSC;
  
  //Create the propagator
    Propagator *thePropagator = new SteppingHelixPropagator(&*theMGField, alongMomentum);

   //Vectors used to perform the matching between Segments and hits from Track
  std::vector< std::vector<int> > indexCollectionDT;
  std::vector< std::vector<int> > indexCollectionCSC;
   

  reco::TrackCollection::const_iterator staTrack;
  for (staTrack = staTracks->begin(); staTrack != staTracks->end(); ++staTrack){

	int countPoints   = 0;
    
    reco::TransientTrack track(*staTrack,&*theMGField,theTrackingGeometry); 
    
    std::vector<int> positionDT;
    std::vector<int> positionCSC;
    std::vector<TrackingRecHit *> my4DTrack;
    
    //Loop over the hits of the track
    for(int counter  = 0; counter != staTrack->recHitsSize()-1; ++counter) {
      
      TrackingRecHitRef myRef = staTrack->recHit(counter);
      const TrackingRecHit *rechit = myRef.get();
      const GeomDet* geomDet = theTrackingGeometry->idToDet(rechit->geographicalId());
      
      //It's a DT Hit
      if(geomDet->subDetector() == GeomDetEnumerators::DT) {
	
	//Take the layer associated to this hit
	DTLayerId myLayer(rechit->geographicalId().rawId());
	
	int NumberOfDTSegment = 0;
	//Loop over segments
	for(segmentDT = all4DSegmentsDT->begin(); segmentDT != all4DSegmentsDT->end(); ++segmentDT) {
	  
	  //By default the chamber associated to this Segment is new
	  bool isNewChamber = true;

	  //Loop over segments already included in the vector of segments in the actual track
	  for(std::vector<int>::iterator positionIt = positionDT.begin(); positionIt != positionDT.end(); ++positionIt) {

	    //If this segment has been used before isNewChamber = false
	    if(NumberOfDTSegment == *positionIt) isNewChamber = false;
	  }

	  //Loop over vectors of segments associated to previous tracks
	  for(std::vector<std::vector<int> >::iterator collect = indexCollectionDT.begin(); collect != indexCollectionDT.end(); ++collect) {

	    //Loop over segments associated to a track
	    for(std::vector<int>::iterator positionIt = (*collect).begin(); positionIt != (*collect).end(); positionIt++) {
	      
	      //If this segment was used in a previos track then isNewChamber = false
	      if(NumberOfDTSegment == *positionIt) isNewChamber = false;
	    }
	  }
	  
	  //If the chamber is new
	  if(isNewChamber) {
	    
	    DTChamberId myChamber((*segmentDT).geographicalId().rawId());
	    //If the layer of the hit belongs to the chamber of the 4D Segment
	    if(myLayer.wheel() == myChamber.wheel() && myLayer.station() == myChamber.station() && myLayer.sector() == myChamber.sector()) {
	      
	      //push position of the segment and tracking rechit
	      positionDT.push_back(NumberOfDTSegment);
	      my4DTrack.push_back((TrackingRecHit *) &(*segmentDT));
	    }
	  } //new Chamber if
	  
	  NumberOfDTSegment++;
	} // DTsegment Loop

      } //was a DT if
      
	//In case is a CSC
      else if (geomDet->subDetector() == GeomDetEnumerators::CSC) {
	
      //Take the layer associated to this hit
      CSCDetId myLayer(rechit->geographicalId().rawId());
	
      int NumberOfCSCSegment = 0;
      //Loop over 2Dsegments
      for(segmentCSC = all2DSegmentsCSC->begin(); segmentCSC != all2DSegmentsCSC->end(); segmentCSC++) {

      //By default the chamber associated to the segment is new
        bool isNewChamber = true;

      //Loop over segments in the current track
        for(std::vector<int>::iterator positionIt = positionCSC.begin(); positionIt != positionCSC.end(); positionIt++) {
	    
      //If this segment has been used then newchamber = false
          if(NumberOfCSCSegment == *positionIt) isNewChamber = false;
        }
      //Loop over vectors of segments in previous tracks
        for(std::vector<std::vector<int> >::iterator collect = indexCollectionCSC.begin(); collect != indexCollectionCSC.end(); ++collect) {
      //Loop over segments in a track
      for(std::vector<int>::iterator positionIt = (*collect).begin(); positionIt != (*collect).end(); positionIt++) {
      //If the segment was used in a previous track isNewChamber = false
            if(NumberOfCSCSegment == *positionIt) isNewChamber = false;
          }
        }
      //  //If the chamber is new
        if(isNewChamber) {
      CSCDetId myChamber((*segmentCSC).geographicalId().rawId());
      //If the chambers are the same
          if(myLayer.chamberId() == myChamber.chamberId()) {
      //push
      positionCSC.push_back(NumberOfCSCSegment);
      my4DTrack.push_back((TrackingRecHit *) &(*segmentCSC));
          }
        }
        NumberOfCSCSegment++;
      } //CSC segment loop
    } //was a CSC if
    

// start propagation
    TrajectoryStateOnSurface innerTSOS = track.impactPointState();

    //If the state is valid
    if(innerTSOS.isValid()) {

      //Loop over Associated segments
      for(std::vector<TrackingRecHit *>::iterator rechit = my4DTrack.begin(); rechit != my4DTrack.end(); ++rechit) {
	
	const GeomDet* geomDet = theTrackingGeometry->idToDet((*rechit)->geographicalId());
	//This try block is to catch some exceptions given by the propagator
	try {
 
	  TrajectoryStateOnSurface destiny = thePropagator->propagate(*(innerTSOS.freeState()), geomDet->surface());
          if(!destiny.isValid()) continue;

	double residualRPhi,residualPhi,residualZ;

  residualRPhi = (*rechit)->localPosition().x() - destiny.localPosition().x();
  residualZ = geomDet->toGlobal((*rechit)->localPosition()).z() - destiny.freeState()->position().z();
  residualPhi = atan2(geomDet->toGlobal(((RecSegment *)(*rechit))->localDirection()).y(), 
  geomDet->toGlobal(((RecSegment*)(*rechit))->localDirection()).x())-atan2(destiny.globalDirection().y(), destiny.globalDirection().x());


          const long rawId= (*rechit)->geographicalId().rawId();
	  int position = -1;
	  bool newDetector = true; 
	  //Loop over the AligmentDetectorCollection to see if the detector is new and requires a new entry
	  for(std::vector<long>::iterator myIds = detectorCollection.begin(); myIds != detectorCollection.end(); myIds++) {
		++position;
	    //If matches newDetector = false
	    if(*myIds == rawId) {
	      newDetector = false;
	      break;
	    }
	  }

	  if(newDetector) {
	    
	    //Create an RawIdDetector, fill it and push it into the collection 
	    detectorCollection.push_back(rawId);
	    //This piece of code calculates the range of the residuals
	    DetId myDet(rawId);
	    int station = 0;
	    //If it's a DT
	    if(myDet.subdetId() == 1) {
	      DTChamberId myChamber(rawId);
	      station = myChamber.station();
	    } else {
	      CSCDetId myChamber(rawId);
	      station = myChamber.ring();
	    }
	    double range = 0.5;
	    switch(station) {
	    case 1:
	      range = 0.5;
	      break;
	    case 2:
	      range = 1.0;
	      break;
	    case 3:
	      range = 3.0;
	      break;
	    case 4:
	      range = 10.0;
	      break;
	    default:
	      break;
	    }

	//create histograms

	    char nameOfHistoRPhi[50];
	    char nameOfHistoPhi[50];
	    char nameOfHistoZ[50];
	
	    std::ifstream input("translation.txt");
	    for(int i = 0; i < 790; ++i) {
	      long id; int det,w,mb,sec;
	      input >> id;
	      input >> det;
	      input >> w;
	      input >> mb;
	      input >> sec;
	      if(id==rawId){

/*	    sprintf(nameOfHistoRPhi, "HistoRPhi%ld", rawId);
	    sprintf(nameOfHistoPhi, "HistoPhi%ld", rawId);
	    sprintf(nameOfHistoZ, "HistoZ%ld", rawId);*/
	    
	    if(det==0){ // DT
	    sprintf(nameOfHistoRPhi, "ResidualRPhi_DTW%ldMB%1dS%1d",w,mb,sec );
	    sprintf(nameOfHistoPhi, "ResidualPhi_DTW%ldMB%1dS%1d",w,mb,sec);
	    sprintf(nameOfHistoZ, "ResidualZ_DTW%ldMB%1dS%1d",w,mb,sec);
	    hResidualRPhiDT->Fill(residualRPhi);
	    hResidualPhiDT->Fill(residualPhi);
	    hResidualZDT->Fill(residualZ);	

	    	    }
	    else if(det==1){ //CSC
	    sprintf(nameOfHistoRPhi, "ResidualRPhi_CSCW%ldSt%1dCh%1d",w,mb,sec );
	    sprintf(nameOfHistoPhi, "ResidualPhi_CSCW%ldSt%1dCh%1d",w,mb,sec);
	    sprintf(nameOfHistoZ, "ResidualZ_CSCW%ldSt%1dCh%1d",w,mb,sec);
	    hResidualRPhiCSC->Fill(residualRPhi);
	    hResidualPhiCSC->Fill(residualPhi);
	    hResidualZCSC->Fill(residualZ);	

		}		    
	    }
	    }
		input.close();
	    
	    TH1F *histoRPhi = new TH1F(nameOfHistoRPhi, nameOfHistoRPhi, 100, -2.0*range, 2.0*range);
	    TH1F *histoPhi = new TH1F(nameOfHistoPhi, nameOfHistoPhi, 100, -0.1*range, 0.1*range);
	    TH1F *histoZ = new TH1F(nameOfHistoZ, nameOfHistoZ, 100, -2.0*range, 2.0*range);


	    histoRPhi->Fill(residualRPhi);
	    histoPhi->Fill(residualPhi);
	    histoZ->Fill(residualZ);	
	    //Push them into their respective vectors
	    unitsRPhi.push_back(histoRPhi);
	    unitsPhi.push_back(histoPhi);
	    unitsZ.push_back(histoZ);

	  } else {
	    //If the detector was not new, just fill the histogram
	    unitsRPhi.at(position)->Fill(residualRPhi);
	    unitsPhi.at(position)->Fill(residualPhi);
	    unitsZ.at(position)->Fill(residualZ);
	  }
	  countPoints++;
	
 	  innerTSOS = destiny;


	}catch(...) {
	  cout << "<MuonAlignmentAnalyzer> Exception in propagator catched" << endl;
	  continue;
	}

	} //loop over my4DTrack
	} //TSOS was valid

	} // loop over recHitsSize

	cout << "Number of points considered in this track: "<< countPoints <<endl;
	} //loop over STAtracks


 } //end doResplots
 

  cout<<"---"<<endl;  
}

