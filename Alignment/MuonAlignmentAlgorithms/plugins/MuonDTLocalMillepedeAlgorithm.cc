#include <fstream>

#include "TFile.h"
#include "TTree.h"
#include "TKey.h"
#include "TMatrixD.h"
#include "TF1.h"


#include "Alignment/MuonAlignmentAlgorithms/src/DTMuonSLToSL.cc"
#include "Alignment/MuonAlignmentAlgorithms/src/DTMuonMillepede.cc"


#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"


#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"

#include "Alignment/CommonAlignment/interface/Alignable.h"  
#include "Alignment/CommonAlignment/interface/AlignableDet.h"
#include "Alignment/CommonAlignment/interface/AlignableNavigator.h"  
#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"  
#include "Alignment/CommonAlignment/interface/AlignmentParameters.h"
#include "Alignment/CommonAlignment/interface/SurveyResidual.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParameterStore.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParameterSelector.h"
#include "Alignment/MuonAlignment/interface/AlignableMuon.h"
#include <DataFormats/GeometrySurface/interface/LocalError.h> 
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "Alignment/MuonAlignmentAlgorithms/plugins/MuonDTLocalMillepedeAlgorithm.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentAlgorithmPluginFactory.h"

// Constructor ----------------------------------------------------------------

MuonDTLocalMillepedeAlgorithm::MuonDTLocalMillepedeAlgorithm(const edm::ParameterSet& cfg):
  AlignmentAlgorithmBase( cfg )
{
  
  edm::LogInfo("Alignment") << "[MuonDTLocalMillepedeAlgorithm] constructed.";

  //Parse parameters. In the future this section should be completed with more options
  ntuplePath = cfg.getParameter<std::string>( "ntuplePath" );
  numberOfRootFiles = cfg.getParameter<int>( "numberOfRootFiles" ); 
  ptMax = cfg.getParameter<double>( "ptMax" );
  ptMin = cfg.getParameter<double>( "ptMin" );
  numberOfSigmasX = cfg.getParameter<double>( "numberOfSigmasX" );
  numberOfSigmasDXDZ = cfg.getParameter<double>( "numberOfSigmasDXDZ" );
  numberOfSigmasY = cfg.getParameter<double>( "numberOfSigmasY" );
  numberOfSigmasDYDZ = cfg.getParameter<double>( "numberOfSigmasDYDZ" );
  nPhihits = cfg.getParameter<double>( "nPhihits" );
  nThetahits = cfg.getParameter<double>( "nThetahits" );
  workingmode = cfg.getParameter<int>( "workingMode" );
  nMtxSection = cfg.getParameter<int>( "nMtxSection" );

 
  //The algorithm has three working modes:
  //0.- aligment information is extracted from the events and stored in root files
  //1.- The SLtoSl algorithm
  //2.- The local MuonMillepede algorithm
  if(workingmode == 0) {
    edm::LogInfo("Alignment") << "[MuonDTLocalMillepedeAlgorithm] Running on production mode."; 
    char nameOfFile[200];
    snprintf(nameOfFile, sizeof(nameOfFile), "%s/MyNtupleResidual.root", ntuplePath.c_str());
    f = new TFile(nameOfFile, "RECREATE");
    f->cd();
    setBranchTrees();
  } else if (workingmode == 1) {
    edm::LogInfo("Alignment") << "[MuonDTLocalMillepedeAlgorithm] Running SLToSL algorithm."; 
  } else {
    edm::LogInfo("Alignment") << "[MuonDTLocalMillepedeAlgorithm] Running Local Millepede algorithm."; 
  }

}




// Call at beginning of job ---------------------------------------------------
void 
MuonDTLocalMillepedeAlgorithm::initialize( const edm::EventSetup& setup, 
				    AlignableTracker* tracker, AlignableMuon* muon, 
				    AlignmentParameterStore* store )
{
  
  // accessor Det->AlignableDet
  if ( !muon )
    theAlignableDetAccessor = new AlignableNavigator(tracker);
  else if ( !tracker )
    theAlignableDetAccessor = new AlignableNavigator(muon);
  else 
    theAlignableDetAccessor = new AlignableNavigator(tracker,muon);
  
  // set alignmentParameterStore
  theAlignmentParameterStore=store;
  
  // get alignables
  theAlignables = theAlignmentParameterStore->alignables();
  
}



// Call at end of job ---------------------------------------------------------
void MuonDTLocalMillepedeAlgorithm::terminate(void)
{

  //If workingmode equals 1 or 2, the algorithms are run before saving.   
  if(workingmode == 1) {
    edm::LogInfo("Alignment") << "[MuonDTLocalMillepedeAlgorithm] Starting SLToSL algorithm";
    DTMuonSLToSL mySLToSL(ntuplePath, numberOfRootFiles, ptMax, ptMin, f);
  } else if(workingmode >= 2) {
    edm::LogInfo("Alignment") << "[MuonDTLocalMillepedeAlgorithm] Starting local MuonMillepede algorithm";
    DTMuonMillepede myMillepede(ntuplePath, numberOfRootFiles, ptMax, ptMin, nPhihits, nThetahits, workingmode, nMtxSection);
  } 

  if (workingmode==0) {
    f->Write();
    f->Close();
  }

}



// Run the algorithm on trajectories and tracks -------------------------------
void MuonDTLocalMillepedeAlgorithm::run(const edm::EventSetup& setup, const EventInfo &eventInfo)
//void MuonDTLocalMillepedeAlgorithm::run( const edm::EventSetup& setup, const ConstTrajTrackPairCollection& tracks)
{

  //Only important in the production mode
  if(workingmode != 0) return;
 
  const ConstTrajTrackPairCollection &tracks = eventInfo.trajTrackPairs_;
  for( ConstTrajTrackPairCollection::const_iterator it=tracks.begin();
       it!=tracks.end();it++) {

    const Trajectory *traj = (*it).first;
    const reco::Track *track = (*it).second;
    
    p    = track->p();
    pt    = track->pt();
    eta   = track->eta();
    phi   = track->phi();
    charge = track->charge();
 

    if(pt < ptMin || pt > ptMax) continue;
    
    vector<const TransientTrackingRecHit*> hitvec;
    vector<TrajectoryMeasurement> measurements = traj->measurements();

    //In this loop the measurements and hits are extracted and put into two vectors 
    int ch_muons = 0;
    for (vector<TrajectoryMeasurement>::iterator im=measurements.begin();
	 im!=measurements.end(); im++) {
      TrajectoryMeasurement meas = *im;
      const TransientTrackingRecHit* hit = &(*meas.recHit());
      //We are not very strict at this point
      if (hit->isValid()) {
        if(hit->det()->geographicalId().det() == 2 && hit->det()->geographicalId().subdetId() == 1) {
          hitvec.push_back(hit);
          ch_muons++;
	} 
      }
    }


    vector<const TransientTrackingRecHit*>::const_iterator ihit=hitvec.begin();
    //Information is stored temporally in the myTrack1D object, which is analyzed
    //in the build4DSegments method in order to associate hits to segments. 
    int ch_counter = 0;
    while (ihit != hitvec.end()) 
      {
	const GeomDet* det=(*ihit)->det();
        if(det->geographicalId().det() == 2) {
	  if(det->geographicalId().subdetId() == 1) {
            DTLayerId mLayer(det->geographicalId().rawId());
            DTChamberId mChamber(mLayer.wheel(), mLayer.station(), mLayer.sector());
            AlignableDet *aliDet = theAlignableDetAccessor->alignableDetFromDetId(mChamber);
	    myTrack1D.wh[ch_counter] = mLayer.wheel();
            myTrack1D.st[ch_counter] = mLayer.station();
            myTrack1D.sr[ch_counter] = mLayer.sector();
            myTrack1D.sl[ch_counter] = mLayer.superlayer();
            myTrack1D.la[ch_counter] = mLayer.layer();
            myTrack1D.erx[ch_counter] = (*ihit)->localPositionError().xx();
	    GlobalPoint globhit = det->toGlobal((*ihit)->localPosition());
            LocalPoint seghit = aliDet->surface().toLocal(globhit);
            myTrack1D.xc[ch_counter] = seghit.x();
	    myTrack1D.yc[ch_counter] = seghit.y();
	    myTrack1D.zc[ch_counter] = seghit.z();
	    ch_counter++;
	  }
        }
        ihit++;
      }
    myTrack1D.nhits = ch_counter;
    if(build4DSegments()) ttreeOutput->Fill();
  }
  
}



//This method separates the hits depending on the chamber and builds the 4D segments
//through linear fits.
//It returns true if 4Dsegments were correctly built and false otherwise.  
//--------------------------------------------------------------------------------------------------------------
bool MuonDTLocalMillepedeAlgorithm::build4DSegments() {

  bool saveThis = false;
  
  //Set to 0
  int id[20][5];
  int numlayer[20][12];
  for(int s = 0; s < 20; ++s) {
    for(int k = 0; k < 5; ++k) id[s][k] = 0;
    for(int k = 0; k < 12; ++k) numlayer[s][k] = 0;
  }

  
  int nChambers = 0;
  for(int counter = 0; counter < myTrack1D.nhits; ++counter) {
    bool isNew = true;
    for(int counterCham = 0; counterCham < nChambers; counterCham++) {
      if(myTrack1D.wh[counter] == id[counterCham][0] &&
	 myTrack1D.st[counter] == id[counterCham][1] &&
         myTrack1D.sr[counter] == id[counterCham][2]) {
	if(myTrack1D.sl[counter] == 2) { 
	  id[counterCham][4]++;
	} else {
	  id[counterCham][3]++;
	}
	for (int ila = 1; ila<=4; ila++)
	  if (myTrack1D.la[counter]==ila) {
	    int jla = (myTrack1D.sl[counter]-1)*4 + ila -1;
	    numlayer[counterCham][jla]++;
	  }
	isNew = false;
      } 
    }
    if(isNew) {
      id[nChambers][0] = myTrack1D.wh[counter];
      id[nChambers][1] = myTrack1D.st[counter];
      id[nChambers][2] = myTrack1D.sr[counter];
      if(myTrack1D.sl[counter] == 2) {
        id[nChambers][4]++;
      } else {
        id[nChambers][3]++;
      }
      for (int ila = 1; ila<=4; ila++)
	if (myTrack1D.la[counter]==ila) {
	  int jla = (myTrack1D.sl[counter]-1)*4 + ila -1;
	  numlayer[nChambers][jla]++;
	}
      nChambers++;
    }
  }
  
  for (int iseg = 0; iseg<MAX_SEGMENT; iseg++) 
    for (int ihit = 0; ihit<MAX_HIT_CHAM; ihit++) {
      xc[iseg][ihit] = -250.;
      yc[iseg][ihit] = -250.;
      zc[iseg][ihit] = -250.;
      ex[iseg][ihit] = -250.;
      xcp[iseg][ihit] = -250.;
      ycp[iseg][ihit] = -250.;
      excp[iseg][ihit] = -250.;
      eycp[iseg][ihit] = -250.;
      sl[iseg][ihit] = 0;
      la[iseg][ihit] = 0;
    }

  nseg = 0;
  for(int counter = 0; counter < nChambers; ++counter) {
    
    bool GoodPhiChamber = true, GoodThetaChamber = true;
    for (int ila = 1; ila<=12; ila++) {
      if (numlayer[counter][ila-1]!=1 && (ila<5 || ila>8)) GoodPhiChamber = false;
      if (numlayer[counter][ila-1]!=1 && (ila<9 || ila>4) && id[counter][1]!=4) GoodThetaChamber = false;
    }

    if(id[counter][3] >= nPhihits && (id[counter][4] >= nThetahits || id[counter][1] == 4) &&
       GoodPhiChamber && GoodThetaChamber) {

      TMatrixD phiProjection(2,2);
      TMatrixD thetaProjection(2,2);
      TMatrixD bphiProjection(2,1);
      TMatrixD bthetaProjection(2,1);
      
      TMatrixD phiProjectionSL1(2,2);
      TMatrixD bphiProjectionSL1(2,1);
      TMatrixD phiProjectionSL3(2,2);
      TMatrixD bphiProjectionSL3(2,1);
     
      float SL1_z_ave = 0;
      float SL3_z_ave = 0;

      int numh1 = 0, numh2 = 0, numh3 = 0;
      for(int counterH = 0; counterH < myTrack1D.nhits; ++counterH) {
        if(myTrack1D.wh[counterH] == id[counter][0] && myTrack1D.st[counterH] == id[counter][1] &&
           myTrack1D.sr[counterH] == id[counter][2]) {
	  if(myTrack1D.sl[counterH] == 2) {
	    numh2++;
            thetaProjection(0,0) += 1.0/myTrack1D.erx[counterH];
            thetaProjection(0,1) += myTrack1D.zc[counterH]/myTrack1D.erx[counterH];
            thetaProjection(1,0) += myTrack1D.zc[counterH]/myTrack1D.erx[counterH];
            thetaProjection(1,1) += myTrack1D.zc[counterH]*myTrack1D.zc[counterH]/myTrack1D.erx[counterH];
            bthetaProjection(0,0) += myTrack1D.yc[counterH]/myTrack1D.erx[counterH];
            bthetaProjection(1,0) += myTrack1D.yc[counterH]*myTrack1D.zc[counterH]/myTrack1D.erx[counterH];
          } else {
            phiProjection(0,0) += 1.0/myTrack1D.erx[counterH];
            phiProjection(0,1) += myTrack1D.zc[counterH]/myTrack1D.erx[counterH];
            phiProjection(1,0) += myTrack1D.zc[counterH]/myTrack1D.erx[counterH];
            phiProjection(1,1) += myTrack1D.zc[counterH]*myTrack1D.zc[counterH]/myTrack1D.erx[counterH];
            bphiProjection(0,0) += myTrack1D.xc[counterH]/myTrack1D.erx[counterH];
            bphiProjection(1,0) += myTrack1D.xc[counterH]*myTrack1D.zc[counterH]/myTrack1D.erx[counterH];
	    if(myTrack1D.sl[counterH] == 1) {
	      numh1++;
              phiProjectionSL1(0,0) += 1.0/myTrack1D.erx[counterH];
              phiProjectionSL1(0,1) += myTrack1D.zc[counterH]/myTrack1D.erx[counterH];
              phiProjectionSL1(1,0) += myTrack1D.zc[counterH]/myTrack1D.erx[counterH];
              phiProjectionSL1(1,1) += myTrack1D.zc[counterH]*myTrack1D.zc[counterH]/myTrack1D.erx[counterH];
              bphiProjectionSL1(0,0) += myTrack1D.xc[counterH]/myTrack1D.erx[counterH];
              bphiProjectionSL1(1,0) += myTrack1D.xc[counterH]*myTrack1D.zc[counterH]/myTrack1D.erx[counterH];
              SL1_z_ave += myTrack1D.zc[counterH];
            } else {
	      numh3++;
              phiProjectionSL3(0,0) += 1.0/myTrack1D.erx[counterH];
              phiProjectionSL3(0,1) += myTrack1D.zc[counterH]/myTrack1D.erx[counterH];
              phiProjectionSL3(1,0) += myTrack1D.zc[counterH]/myTrack1D.erx[counterH];
              phiProjectionSL3(1,1) += myTrack1D.zc[counterH]*myTrack1D.zc[counterH]/myTrack1D.erx[counterH];
              bphiProjectionSL3(0,0) += myTrack1D.xc[counterH]/myTrack1D.erx[counterH];
              bphiProjectionSL3(1,0) += myTrack1D.xc[counterH]*myTrack1D.zc[counterH]/myTrack1D.erx[counterH];
              SL3_z_ave += myTrack1D.zc[counterH];
            } 
          }
	}
      }

      SL1_z_ave /= 4.0;
      SL3_z_ave /= 4.0;
      
      if (phiProjection(0,0) != 0 && phiProjectionSL1(0,0) != 0 && phiProjectionSL3(0,0) != 0 && 
	  (thetaProjection(0,0) != 0 || id[counter][1] == 4)) {
	
	wh[nseg] = id[counter][0];
	st[nseg] = id[counter][1];
	sr[nseg] = id[counter][2];
	
	if(thetaProjection(0,0) != 0 && id[counter][1] != 4) { // Already asked (almost)
	  thetaProjection.Invert();
	  TMatrixD solution = thetaProjection*bthetaProjection;
	  ySl[nseg] = solution(0,0);
	  dydzSl[nseg] = solution(1,0);
	  eySl[nseg] = thetaProjection(0,0);
	  edydzSl[nseg] = thetaProjection(1,1);
	  eydydzSl[nseg] = thetaProjection(0,1);
	}
	phiProjection.Invert();
	phiProjectionSL1.Invert();
	phiProjectionSL3.Invert();
	TMatrixD solution = phiProjection*bphiProjection;
	TMatrixD solutionSL1 = phiProjectionSL1*bphiProjectionSL1;
	TMatrixD solutionSL3 = phiProjectionSL3*bphiProjectionSL3;
	xSl[nseg] = solution(0,0);
	dxdzSl[nseg] = solution(1,0);
	exSl[nseg] = phiProjection(0,0);
	edxdzSl[nseg] = phiProjection(1,1);
	exdxdzSl[nseg] = phiProjection(0,0);
	xSlSL1[nseg] = solutionSL1(0,0);
	dxdzSlSL1[nseg] = solutionSL1(1,0);
	exSlSL1[nseg] = phiProjectionSL1(0,0);
	edxdzSlSL1[nseg] = phiProjectionSL1(1,1);
	exdxdzSlSL1[nseg] = phiProjectionSL1(0,0);
        xSL1SL3[nseg] = solutionSL1(0,0) + SL3_z_ave * solutionSL1(1,0);
	xSlSL3[nseg] = solutionSL3(0,0);
	dxdzSlSL3[nseg] = solutionSL3(1,0);
	exSlSL3[nseg] = phiProjectionSL3(0,0);
	edxdzSlSL3[nseg] = phiProjectionSL3(1,1);
	exdxdzSlSL3[nseg] = phiProjectionSL3(0,0);
        xSL3SL1[nseg] = solutionSL3(0,0) + SL1_z_ave * solutionSL3(1,0);
        int hitcounter = 0;
        for(int counterH = 0; counterH < myTrack1D.nhits; ++counterH) {
          if(myTrack1D.wh[counterH] == wh[nseg] && myTrack1D.st[counterH] == st[nseg] &&
             myTrack1D.sr[counterH] == sr[nseg]) {
            xc[nseg][hitcounter] = myTrack1D.xc[counterH];
            yc[nseg][hitcounter] = myTrack1D.yc[counterH];
            zc[nseg][hitcounter] = myTrack1D.zc[counterH];
            ex[nseg][hitcounter] = myTrack1D.erx[counterH];
            xcp[nseg][hitcounter] = xSl[nseg]+dxdzSl[nseg]*myTrack1D.zc[counterH];
            ycp[nseg][hitcounter] = ySl[nseg]+dydzSl[nseg]*myTrack1D.zc[counterH];
            excp[nseg][hitcounter] = exSl[nseg]*exSl[nseg]+ (edxdzSl[nseg]*edxdzSl[nseg])*myTrack1D.zc[counterH];
            eycp[nseg][hitcounter] = eySl[nseg]*eySl[nseg]+ (edydzSl[nseg]*edydzSl[nseg])*myTrack1D.zc[counterH];
            sl[nseg][hitcounter] = myTrack1D.sl[counterH];
            la[nseg][hitcounter] = myTrack1D.la[counterH];
            saveThis = true;
            hitcounter++;
          }
        }
	nphihits[nseg] = id[counter][3];
	nthetahits[nseg] = id[counter][4];
	nhits[nseg] = hitcounter;
	nseg++;
      }
    }
  }
  return saveThis;
}



//The tree structure is defined and the variables associated ---------------------------------
void MuonDTLocalMillepedeAlgorithm::setBranchTrees() {

  ttreeOutput = new TTree("InfoTuple", "InfoTuple");
  
  ttreeOutput->Branch("p", &p, "p/F");
  ttreeOutput->Branch("pt", &pt, "pt/F");
  ttreeOutput->Branch("eta", &eta, "eta/F");
  ttreeOutput->Branch("phi", &phi, "phi/F");
  ttreeOutput->Branch("charge", &charge, "charge/F");
  ttreeOutput->Branch("nseg", &nseg, "nseg/I");
  ttreeOutput->Branch("nphihits", nphihits, "nphihits[nseg]/I");
  ttreeOutput->Branch("nthetahits", nthetahits, "nthetahits[nseg]/I");
  ttreeOutput->Branch("nhits", nhits, "nhits[nseg]/I");
  ttreeOutput->Branch("xSl", xSl, "xSl[nseg]/F");
  ttreeOutput->Branch("dxdzSl", dxdzSl, "dxdzSl[nseg]/F");
  ttreeOutput->Branch("exSl", exSl, "exSl[nseg]/F");
  ttreeOutput->Branch("edxdzSl", edxdzSl, "edxdzSl[nseg]/F");
  ttreeOutput->Branch("exdxdzSl", edxdzSl, "exdxdzSl[nseg]/F");
  ttreeOutput->Branch("ySl", ySl, "ySl[nseg]/F");
  ttreeOutput->Branch("dydzSl", dydzSl, "dydzSl[nseg]/F");
  ttreeOutput->Branch("eySl", eySl, "eySl[nseg]/F");
  ttreeOutput->Branch("edydzSl", edydzSl, "edydzSl[nseg]/F");
  ttreeOutput->Branch("eydydzSl", eydydzSl, "eydydzSl[nseg]/F");
  ttreeOutput->Branch("xSlSL1", xSlSL1, "xSlSL1[nseg]/F");
  ttreeOutput->Branch("dxdzSlSL1", dxdzSlSL1, "dxdzSlSL1[nseg]/F");
  ttreeOutput->Branch("exSlSL1", exSlSL1, "exSlSL1[nseg]/F");
  ttreeOutput->Branch("edxdzSlSL1", edxdzSlSL1, "edxdzSlSL1[nseg]/F");
  ttreeOutput->Branch("xSL1SL3", xSL1SL3, "xSL1SL3[nseg]/F");
  ttreeOutput->Branch("xSlSL3", xSlSL3, "xSlSL3[nseg]/F");
  ttreeOutput->Branch("dxdzSlSL3", dxdzSlSL3, "dxdzSlSL3[nseg]/F");
  ttreeOutput->Branch("exSlSL3", exSlSL3, "exSlSL3[nseg]/F");
  ttreeOutput->Branch("edxdzSlSL3", edxdzSlSL3, "edxdzSlSL3[nseg]/F");
  ttreeOutput->Branch("xSL3SL1", xSL3SL1, "xSL3SL1[nseg]/F");
  ttreeOutput->Branch("xc", xc, "xc[nseg][14]/F");
  ttreeOutput->Branch("yc", yc, "yc[nseg][14]/F");
  ttreeOutput->Branch("zc", zc, "zc[nseg][14]/F");
  ttreeOutput->Branch("ex", ex, "ex[nseg][14]/F");
  ttreeOutput->Branch("xcp", xcp, "xcp[nseg][14]/F");
  ttreeOutput->Branch("ycp", ycp, "ycp[nseg][14]/F");
  ttreeOutput->Branch("excp", excp, "excp[nseg][14]/F");
  ttreeOutput->Branch("eycp", eycp, "eycp[nseg][14]/F");
  ttreeOutput->Branch("wh", wh, "wh[nseg]/I");
  ttreeOutput->Branch("st", st, "st[nseg]/I");
  ttreeOutput->Branch("sr", sr, "sr[nseg]/I");
  ttreeOutput->Branch("sl", sl, "sl[nseg][14]/I");
  ttreeOutput->Branch("la", la, "la[nseg][14]/I");

}


DEFINE_EDM_PLUGIN( AlignmentAlgorithmPluginFactory, MuonDTLocalMillepedeAlgorithm, "MuonDTLocalMillepedeAlgorithm" );

