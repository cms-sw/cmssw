///////////////////////////////////////////////////////////////////////////////
// File: RecFP420Test
// Date: 02.2007
// Description: RecFP420Test for FP420
// Modifications: std::  added wrt OSCAR code 
///////////////////////////////////////////////////////////////////////////////
// system include files
#include <iostream>
#include <iomanip>
#include <cmath>
#include<vector>
//
#include "SimG4Core/Notification/interface/BeginOfEvent.h"
#include "SimG4Core/Notification/interface/EndOfEvent.h"
#include "SimG4Core/Notification/interface/TrackWithHistory.h"
#include "SimG4Core/Notification/interface/TrackInformation.h"
 
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimG4CMS/FP420/interface/FP420NumberingScheme.h"
#include "SimG4CMS/FP420/interface/FP420G4HitCollection.h"
#include "SimG4CMS/FP420/interface/FP420G4Hit.h"
#include "RecoRomanPot/RecoFP420/interface/RecFP420Test.h"

// new:DigitizerFP420.cc
//#include "SimRomanPot/SimFP420/interface/FP420DigiMain.h"

#include "SimRomanPot/SimFP420/interface/HDigiFP420.h"
#include "SimRomanPot/SimFP420/interface/DigitizerFP420.h"
#include "SimRomanPot/SimFP420/interface/DigiCollectionFP420.h"

#include "SimRomanPot/SimFP420/interface/ClusterFP420.h"
#include "RecoRomanPot/RecoFP420/interface/ClusterizerFP420.h"
#include "RecoRomanPot/RecoFP420/interface/ClusterCollectionFP420.h"

#include "RecoRomanPot/RecoFP420/interface/TrackFP420.h"
#include "RecoRomanPot/RecoFP420/interface/TrackerizerFP420.h"
#include "RecoRomanPot/RecoFP420/interface/TrackCollectionFP420.h"

// G4 stuff
#include "G4SDManager.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4VProcess.hh"
#include "G4HCofThisEvent.hh"
#include "G4UserEventAction.hh"
#include "G4TransportationManager.hh"
#include "G4ProcessManager.hh"
//#include "G4EventManager.hh"

#include "CLHEP/Units/SystemOfUnits.h"
#include "CLHEP/Units/PhysicalConstants.h"
#include <stdio.h>
#include <gsl/gsl_fit.h>
//#include <gsl/gsl_cdf.h>


//================================================================
// Root stuff

// Include the standard header <cassert> to effectively include
// the standard header <assert.h> within the std namespace.
#include <cassert>

using namespace edm;
using namespace std;
///////////////////////////////////////////////////////////////////////////////

//#define ddebugprim
//#define ddebug
//#define mydigidebug10

//#define mydebug
//#define myddebug
//#define mmydebug
//#define myTrackdebug10

// DIGI checks:
//     #define mydigidebug10
// Cluster checks
//#define myClusterdebug10



//================================================================


enum ntfp420_elements {
  ntfp420_evt
};




//================================================================
RecFP420Test::RecFP420Test(const edm::ParameterSet & conf):conf_(conf),theDigitizerFP420(new DigitizerFP420(conf)){
  //constructor
  edm::ParameterSet m_Anal = conf.getParameter<edm::ParameterSet>("RecFP420Test");
    verbosity    = m_Anal.getParameter<int>("Verbosity");
  //verbosity    = 1;

    fDataLabel  =   m_Anal.getParameter<std::string>("FDataLabel");
    fOutputFile =   m_Anal.getParameter<std::string>("FOutputFile");
    fRecreateFile = m_Anal.getParameter<std::string>("FRecreateFile");
   
  if (verbosity > 0) {
   std::cout<<"============================================================================"<<std::endl;
   std::cout << "RecFP420Test constructor :: Initialized as observer"<< std::endl;
  }
	

  //zUnit = 8000.; // 2Stations
  zD2 = 1000.;  // dist between centers of 1st and 2nd stations
  zD3 = 8000.;  // dist between centers of 1st and 3rd stations

  z1 = -150. + (118.4+10.)/2; // 10. -arbitrary
  z2 = z1+zD2;
  z3 = z1+zD3;
  z4 = z1+2*zD3;
  //==================================
  if (verbosity > 0) {
   std::cout<<"============================================================================"<<std::endl;
   // std::cout << "RecFP420Test constructor :: Initialized as observer zUnit=" << zUnit << std::endl;
   std::cout << "RecFP420Test constructor :: Initialized as observer zD2=" << zD2 << std::endl;
   std::cout << " zD3=" << zD3 << std::endl;
   std::cout << " z1=" << z1 << " z2=" << z2 << " z3=" << z3 << " z4=" << z4 << std::endl;
  }
  //==================================

 fp420eventntuple = new TNtuple("NTfp420event","NTfp420event","evt");
  //==================================

  whichevent = 0;

  //   fDataLabel      = "defaultData";
  //       fOutputFile     = "TheAnlysis.root";
  //       fRecreateFile   = "RECREATE";

        TheHistManager = new Fp420AnalysisHistManager(fDataLabel);

  //==================================
  if (verbosity > 0) {
   std::cout << "RecFP420Test constructor :: Initialized Fp420AnalysisHistManager"<< std::endl;
  }
  //==================================
  sn0 = 4;// related to  number of station: sn0=3 mean 2 Stations
  pn0 = 9;// related to number of planes: pn0=11 mean 10 Planes
  //-------------------------------------------------
  UseHalfPitchShiftInX_= true;
  //UseHalfPitchShiftInX_= false;
  
  UseHalfPitchShiftInY_= true;
  //UseHalfPitchShiftInY_= false;
  
  //-------------------------------------------------
  //UseThirdPitchShiftInY_ = true;
  UseThirdPitchShiftInY_ = false;
  
  //UseThirdPitchShiftInX_ = true;
  UseThirdPitchShiftInX_ = false;
  
  //-------------------------------------------------
  //UseForthPitchShiftInY_ = true;
  UseForthPitchShiftInY_ = false;
  
  //UseForthPitchShiftInX_ = true;
  UseForthPitchShiftInX_ = false;
  
  //-------------------------------------------------
	ldriftX= 0.050;
	ldriftY= 0.050;// was 0.040
	
	pitchX= 0.050;
	pitchY= 0.050;// was 0.040
	
	numStripsX = 401;  // X plate number of strips:400*0.050=20mm --> 200*0.100=20mm
	//  numStripsY = 251;  // Y plate number of strips:250*0.040=10mm --> 100*0.100=10mm
	numStripsY = 201;  // Y plate number of strips:200*0.050=10mm --> 100*0.100=10mm
	
	dYYconst = 5.;// XSiDet/2.
	//  dXXconst = 12.7+0.05;//(BoxYshft+dYGap) + (YSi - YSiDet)/2. = 12.7+0.05
	dXXconst = 12.7;//(BoxYshft+dYGap) + (YSi - YSiDet)/2. = 12.7
	//ENC = 1800;
	//ENC = 3000;
	ENC = 2160;
	ElectronPerADC =300;
	Thick300 = 0.300;
//
	double zBlade = 5.00;
	double gapBlade = 1.6;
	ZSiPlane=2*(zBlade+gapBlade);

	double ZKapton = 0.1;
	ZSiStep=ZSiPlane+ZKapton;

	double ZBoundDet = 0.020;
	double ZSiElectr = 0.250;
	double ZCeramDet = 0.500;
//
	ZSiDetL = 0.250;
	ZSiDetR = 0.250;
	ZGapLDet= zBlade/2-(ZSiDetL+ZSiElectr+ZBoundDet+ZCeramDet/2);
//

  // Initialization:

	theFP420NumberingScheme = new FP420NumberingScheme();
	theClusterizerFP420 = new ClusterizerFP420(conf_);
	theTrackerizerFP420 = new TrackerizerFP420(conf_);
//
}



RecFP420Test::~RecFP420Test() {
  //  delete UserNtuples;
  delete theFP420NumberingScheme;
  delete theDigitizerFP420;
  delete theClusterizerFP420;
  delete theTrackerizerFP420;

  TFile fp420OutputFile("newntfp420.root","RECREATE");
  std::cout << "RecFP420Test output root file has been created";
  fp420eventntuple->Write();
  std::cout << ", written";
  fp420OutputFile.Close();
  std::cout << ", closed";
  delete fp420eventntuple;
  std::cout << ", and deleted" << std::endl;

        //------->while end

        // Write histograms to file
        TheHistManager->WriteToFile(fOutputFile,fRecreateFile);

  if (verbosity > 0) {
    std::cout << std::endl << "RecFP420Test Destructor  -------->  End of RecFP420Test : "
      << std::cout << std::endl; 
  }

  std::cout<<"RecFP420Test: End of process"<<std::endl;

}

//================================================================

//================================================================
// Histoes:
//-----------------------------------------------------------------------------

Fp420AnalysisHistManager::Fp420AnalysisHistManager(TString managername)
{
        // The Constructor

        fTypeTitle=managername;
        fHistArray = new TObjArray();      // Array to store histos
        fHistNamesArray = new TObjArray(); // Array to store histos's names

        BookHistos();

        fHistArray->Compress();            // Removes empty space
        fHistNamesArray->Compress();

//      StoreWeights();                    // Store the weights

}
//-----------------------------------------------------------------------------

Fp420AnalysisHistManager::~Fp420AnalysisHistManager()
{
        // The Destructor

        if(fHistArray){
                fHistArray->Delete();
                delete fHistArray;
        }

        if(fHistNamesArray){
                fHistNamesArray->Delete();
                delete fHistNamesArray;
        }
}
//-----------------------------------------------------------------------------

void Fp420AnalysisHistManager::BookHistos()
{
        // Book the histograms and add them to the array

  // at Start: (mm)
    HistInit("ntrackscoll","ntrackscoll",10, 0.,  5.);
//
//
}

//-----------------------------------------------------------------------------

void Fp420AnalysisHistManager::WriteToFile(TString fOutputFile,TString fRecreateFile)
{

        //Write to file = fOutputFile

        std::cout <<"================================================================"<<std::endl;
        std::cout <<" Write this Analysis to File "<<fOutputFile<<std::endl;
        std::cout <<"================================================================"<<std::endl;

        TFile* file = new TFile(fOutputFile, fRecreateFile);

        fHistArray->Write();
        file->Close();
}
//-----------------------------------------------------------------------------

void Fp420AnalysisHistManager::HistInit(const char* name, const char* title, Int_t nbinsx, Axis_t xlow, Axis_t xup)
{
        // Add histograms and histograms names to the array

        char* newtitle = new char[strlen(title)+strlen(fTypeTitle)+5];
        strcpy(newtitle,title);
        strcat(newtitle," (");
        strcat(newtitle,fTypeTitle);
        strcat(newtitle,") ");
        fHistArray->AddLast((new TH1F(name, newtitle, nbinsx, xlow, xup)));
        fHistNamesArray->AddLast(new TObjString(name));

}
//-----------------------------------------------------------------------------

void Fp420AnalysisHistManager::HistInit(const char* name, const char* title, Int_t nbinsx, Axis_t xlow, Axis_t xup, Int_t nbinsy, Axis_t ylow, Axis_t yup)
{
        // Add histograms and histograms names to the array

        char* newtitle = new char[strlen(title)+strlen(fTypeTitle)+5];
        strcpy(newtitle,title);
        strcat(newtitle," (");
        strcat(newtitle,fTypeTitle);
        strcat(newtitle,") ");
        fHistArray->AddLast((new TH2F(name, newtitle, nbinsx, xlow, xup, nbinsy, ylow, yup)));
        fHistNamesArray->AddLast(new TObjString(name));

}
//-----------------------------------------------------------------------------

TH1F* Fp420AnalysisHistManager::GetHisto(Int_t Number)
{
        // Get a histogram from the array with index = Number

        if (Number <= fHistArray->GetLast()  && fHistArray->At(Number) != (TObject*)0){

                return (TH1F*)(fHistArray->At(Number));

        }else{

                std::cout << "!!!!!!!!!!!!!!!!!!GetHisto!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
                std::cout << " WARNING ERROR - HIST ID INCORRECT (TOO HIGH) - " << Number << std::endl;
                std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;

                return (TH1F*)(fHistArray->At(0));
        }
}
//-----------------------------------------------------------------------------

TH2F* Fp420AnalysisHistManager::GetHisto2(Int_t Number)
{
        // Get a histogram from the array with index = Number

        if (Number <= fHistArray->GetLast()  && fHistArray->At(Number) != (TObject*)0){

                return (TH2F*)(fHistArray->At(Number));

        }else{

                std::cout << "!!!!!!!!!!!!!!!!GetHisto2!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
                std::cout << " WARNING ERROR - HIST ID INCORRECT (TOO HIGH) - " << Number << std::endl;
                std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;

                return (TH2F*)(fHistArray->At(0));
        }
}
//-----------------------------------------------------------------------------

TH1F* Fp420AnalysisHistManager::GetHisto(const TObjString histname)
{
        // Get a histogram from the array with name = histname

        Int_t index = fHistNamesArray->IndexOf(&histname);
        return GetHisto(index);
}
//-----------------------------------------------------------------------------

TH2F* Fp420AnalysisHistManager::GetHisto2(const TObjString histname)
{
        // Get a histogram from the array with name = histname

        Int_t index = fHistNamesArray->IndexOf(&histname);
        return GetHisto2(index);
}
//-----------------------------------------------------------------------------

void Fp420AnalysisHistManager::StoreWeights()
{
        // Add structure to each histogram to store the weights

        for(int i = 0; i < fHistArray->GetEntries(); i++){
                ((TH1F*)(fHistArray->At(i)))->Sumw2();
        }
}
// Histoes end :



//==================================================================== per JOB
void RecFP420Test::update(const BeginOfJob * job) {
  //job
  std::cout<<"RecFP420Test:beggining of job"<<std::endl;;
}


//==================================================================== per RUN
void RecFP420Test::update(const BeginOfRun * run) {
  //run

 std::cout << std::endl << "RecFP420Test:: Begining of Run"<< std::endl; 
}


void RecFP420Test::update(const EndOfRun * run) {;}



//=================================================================== per EVENT
void RecFP420Test::update(const BeginOfEvent * evt) {
  iev = (*evt)()->GetEventID();
  //  std::cout <<"RecFP420Test:: Event number = " << iev << std::endl;
  whichevent++;
}

//=================================================================== per Track
void RecFP420Test::update(const BeginOfTrack * trk) {
  itrk = (*trk)()->GetTrackID();
  //   std::cout <<" ====================BeginOfTrack number = " << itrk << std::endl;
  if(itrk == 1) {
     SumEnerDeposit = 0.;
     numofpart = 0;
     SumStepl = 0.;
     SumStepc = 0.;
  }
}



//=================================================================== per EndOfTrack
void RecFP420Test::update(const EndOfTrack * trk) {
  itrk = (*trk)()->GetTrackID();
  //   std::cout <<" ==========EndOfTrack number = " << itrk << std::endl;
  if(itrk == 1) {
//    G4double tracklength  = (*trk)()->GetTrackLength();    // Accumulated track length
#ifdef ddebug
    std::cout <<" ===== EndOf Initial Track with number = " << itrk << std::endl;
    std::cout <<" ======sum deposited energy on all steps along primary track =  " << SumEnerDeposit << std::endl;
#endif
 //   G4ThreeVector   vert_mom  = (*trk)()->GetVertexMomentumDirection();
  //  G4ThreeVector   vert_pos  = (*trk)()->GetVertexPosition(); // vertex ,where this track was created
        // last step information
        const G4Step* aStep = (*trk)()->GetStep();
        //   G4int csn = (*trk)()->GetCurrentStepNumber();
        //   G4double sl = (*trk)()->GetStepLength();
         // preStep
         G4StepPoint*      preStepPoint = aStep->GetPreStepPoint();
         lastpo   = preStepPoint->GetPosition();
 
  }
}

// =====================================================================================================

//=================================================================== each STEP
void RecFP420Test::update(const G4Step * aStep) {
// ==========================================================================
  
  // track on aStep:                                                                                         !
  G4Track*     theTrack     = aStep->GetTrack();   
  TrackInformation* trkInfo = dynamic_cast<TrackInformation*> (theTrack->GetUserInformation());
   if (trkInfo == 0) {
     std::cout << "RecFP420Test on aStep: No trk info !!!! abort " << std::endl;
   } 
//  G4int         id             = theTrack->GetTrackID();
//  G4String       particleType   = theTrack->GetDefinition()->GetParticleName();   //   !!!
//  G4int         parentID       = theTrack->GetParentID();   //   !!!
//  G4TrackStatus   trackstatus    = theTrack->GetTrackStatus();   //   !!!
//  G4double       tracklength    = theTrack->GetTrackLength();    // Accumulated track length
//  G4ThreeVector   trackmom       = theTrack->GetMomentum();
//  G4double       entot          = theTrack->GetTotalEnergy();   //   !!! deposited on step
//  G4int         curstepnumber  = theTrack->GetCurrentStepNumber();
//  G4ThreeVector   vert_pos       = theTrack->GetVertexPosition(); // vertex ,where this track was created
//  G4ThreeVector   vert_mom       = theTrack->GetVertexMomentumDirection();
  
//  double costheta =vert_mom.z()/sqrt(vert_mom.x()*vert_mom.x()+vert_mom.y()*vert_mom.y()+vert_mom.z()*vert_mom.z());
//  double theta = acos(min(max(costheta,double(-1.)),double(1.)));
////  float eta = -log(tan(theta/2));
//  double phi = -1000.;
//  if (vert_mom.x() != 0) phi = atan2(vert_mom.y(),vert_mom.x()); 
//  if (phi < 0.) phi += twopi;
//  //  double phigrad = phi*360./twopi;  

////G4double       trackvel       = theTrack->GetVelocity();

////std::cout << " trackstatus= " << trackstatus << " entot= " << entot  << std::endl;
#ifdef ddebug
     std::cout << " =========================================================================" << std::endl;

     std::cout << " ==========================================111111" << std::endl;
     std::cout << "RecFP420Test on aStep: Entered for track ID=" << id 
          << " ID Name= " << particleType
          << " at stepNumber= " << curstepnumber 
          << " ID onCaloSur..= " << trkInfo->getIDonCaloSurface()
          << " CaloID Check= " << trkInfo->caloIDChecked() 
          << " trackstatus= " << trackstatus
          << " trackmom= " << trackmom
          << " entot= " << entot
          << " vert_where_track_created= " << vert_pos
          << " vert_mom= " << vert_mom
       //          << " Accumulated tracklength= " << tracklength
          << " parent ID = " << parentID << std::endl;
  G4ProcessManager* pm   = theTrack->GetDefinition()->GetProcessManager();
  G4ProcessVector* pv = pm->GetProcessList();
 G4int np = pm->GetProcessListLength();
 for(G4int i=0; i<np; i++) {
 std::cout <<"i=   " <<i << "ProcessName = "  << ((*pv)[i])->GetProcessName() << std::endl;
   }
#endif


  //// step points:                                                                                         !
 // G4double        stepl         = aStep->GetStepLength();
 // G4double        EnerDeposit   = aStep->GetTotalEnergyDeposit();
  // pointers:                                                                                         !
//G4VPhysicalVolume*  physvol       = theTrack->GetVolume();
//G4VPhysicalVolume*  nextphysvol   = theTrack->GetNextVolume();
//G4Material*       materialtr     = theTrack->GetMaterial();
//G4Material*       nextmaterialtr = theTrack->GetNextMaterial();

  //// preStep
 // G4StepPoint*      preStepPoint = aStep->GetPreStepPoint(); 
 // G4ThreeVector     preposition   = preStepPoint->GetPosition();	
 // G4ThreeVector     prelocalpoint = theTrack->GetTouchable()->GetHistory()->
 // GetTopTransform().TransformPoint(preposition);
 // G4VPhysicalVolume* currentPV     = preStepPoint->GetPhysicalVolume();
 // G4String         prename       = currentPV->GetName();

//const G4VTouchable*  pre_touch    = preStepPoint->GetTouchable();
//     int          pre_levels   = detLevels(pre_touch);

////     G4String      pre_name1    = detName(pre_touch, pre_levels, 1);
////     G4String      pre_name2    = detName(pre_touch, pre_levels, 2);
////     G4String      pre_name3    = detName(pre_touch, pre_levels, 3);
 //       G4String name1[20]; int copyno1[20];
  //    if (pre_levels > 0) {
  //      detectorLevel(pre_touch, pre_levels, copyno1, name1);
  //    }

//  G4LogicalVolume*   lv            = currentPV->GetLogicalVolume();
//  G4Material*       mat           = lv->GetMaterial();
//  std::string prenameVolume;
//  prenameVolume.assign(prename,0,20);

//   G4double         prebeta          = preStepPoint->GetBeta();
//   G4double         precharge        = preStepPoint->GetCharge();
//  G4double          prerad           = mat->GetRadlen();

//  std::cout << " EneryDeposited = " << EnerDeposit << std::endl;
//  std::cout << " prevolume = "      << prename << std::endl;
////  std::cout << " posvolume = "      << aStep->GetPostStepPoint()->GetPhysicalVolume()->GetName() << std::endl;
//  std::cout << " preposition = "    << preposition << std::endl; 
#ifdef ddebug
      float th_tr     = preposition.theta();
      float eta_tr    = -log(tan(th_tr/2));
      float phi_tr    = preposition.phi();
      if (phi_tr < 0.) phi_tr += twopi;

     std::cout << "============aStep: information:============" << std::endl;
     std::cout << " EneryDeposited = " << EnerDeposit
          << " stepl = "          << stepl << std::endl;

     std::cout << "============preStep: information:============" << std::endl;
     std::cout << " preposition = "    << preposition
          << " prelocalpoint = "  << prelocalpoint
          << " eta_tr = "         << eta_tr
          << " phi_tr = "         << phi_tr*360./twopi
          << " prevolume = "      << prename
//          << " posvolume = "      << aStep->GetPostStepPoint()->GetPhysicalVolume()->GetName()
          << " pre_levels = "     << pre_levels
          << std::endl;
      if (pre_levels > 0) {
        for (int i1=0; i1<pre_levels; i1++) 
          std::cout << "level= " << i1 << "name= " << name1[i1] << "copy= " << copyno1[i1] << std::endl;
      }

#endif

  // ==========================================================================

}
// ==========================================================================
// ==========================================================================
int RecFP420Test::detLevels(const G4VTouchable* touch) const {

  //Return number of levels
  if (touch) 
    return ((touch->GetHistoryDepth())+1);
  else
    return 0;
}
// ==========================================================================

G4String RecFP420Test::detName(const G4VTouchable* touch, int level,
                                    int currentlevel) const {

  //Go down to current level
  if (level > 0 && level >= currentlevel) {
    int ii = level - currentlevel; 
    return touch->GetVolume(ii)->GetName();
  } else {
    return "NotFound";
  }
}

void RecFP420Test::detectorLevel(const G4VTouchable* touch, int& level,
                                      int* copyno, G4String* name) const {

  //Get name and copy numbers
  if (level > 0) {
    for (int ii = 0; ii < level; ii++) {
      int i      = level - ii - 1;
      G4VPhysicalVolume* pv = touch->GetVolume(i);
      if (pv != 0) 
        name[ii] = pv->GetName();
      else
        name[ii] = "Unknown";
      copyno[ii] = touch->GetReplicaNumber(i);
    }
  }
}
// ==========================================================================

//===================================================================   End Of Event
void RecFP420Test::update(const EndOfEvent * evt) {
  // ==========================================================================
  
  // Fill-in ntuple
  fp420eventarray[ntfp420_evt] = (float)whichevent;

  //
 // int trackID = 0;
  G4PrimaryParticle* thePrim=0;
  G4double vz=-99990.;
  G4double vx=-99990.,vy=-99990.;


  // prim.vertex:
  G4int nvertex = (*evt)()->GetNumberOfPrimaryVertex();
  if (nvertex !=1) std::cout << "RecFP420Test:NumberOfPrimaryVertex != 1 --> = " << nvertex <<  std::endl;

#ifdef ddebugprim
    std::cout << "NumberOfPrimaryVertex:" << nvertex << std::endl;
#endif
    int varia= 0,varia2= 0,varia3= 0;   // = 0 -all; =1 - MI; =2 - noMI
    double phi= 0., phigrad= 0., th= 0., eta= 0.; 
    double phi2= 0., phigrad2= 0., th2= 0., eta2= 0.; 
    double phi3= 0., phigrad3= 0., th3= 0., eta3= 0.; 
    double zmilimit= z4;
    //if(zUnit==4000.) zmilimit= z3;
    //if(zUnit==8000.) zmilimit= z2;
    zmilimit= z3;// last variant
    for (int i = 0 ; i<nvertex; ++i) {
      G4PrimaryVertex* avertex = (*evt)()->GetPrimaryVertex(i);
      if (avertex == 0)
	std::cout << "RecFP420Test:  End Of Event ERR: pointer to vertex = 0"
		  << std::endl;
#ifdef ddebugprim
      std::cout << "Vertex number :" <<i << std::endl;
#endif
      G4int npart = avertex->GetNumberOfParticle();
      if (npart !=1)
	std::cout << "RecFP420Test:: My warning: NumberOfPrimaryPart != 1  -->  = " << npart <<  std::endl;
#ifdef ddebugprim
      std::cout << "number of particles for Vertex :" << npart << std::endl;
#endif
      if (npart ==0)
	std::cout << "RecFP420Test: End Of Event ERR: no NumberOfParticle" << std::endl;
      
      // find just primary track:                                                             track pointer: thePrim
      
      for (int i = 0 ; i<npart; ++i) {
	thePrim=avertex->GetPrimary(i);
	G4ThreeVector   mom  = thePrim->GetMomentum();
	if(i==0){
	  phi = mom.phi();
	  if (phi < 0.) phi += twopi;
	  phigrad = phi*180./pi;
	  th     = mom.theta();
	  eta = -log(tan(th/2));
	  //  if(  lastpo.z()< zmilimit || (lastpo.z()>zmilimit && lastpo.perp()> 100.) ) {
  if(  lastpo.z()< zmilimit ) {
	 varia = 1;
  }
  else{
         varia = 2;
  } 

	}
	else if(i==1){
	  phi2= mom.phi();
	  if (phi2< 0.) phi2 += twopi;
	  phigrad2 = phi2*180./pi;
	  th2     = mom.theta();
	  eta2 = -log(tan(th2/2));
	  //  if(  lastpo.z()< zmilimit || (lastpo.z()>zmilimit && lastpo.perp()> 100.) ) {
  if(  lastpo.z()< zmilimit  ) {
	 varia2= 1;
  }
  else{
         varia2= 2;
  } 

	}
	else if(i==2){
	  phi3 = mom.phi();
	  if (phi3 < 0.) phi3 += twopi;
	  phigrad3 = phi3*180./pi;
	  th3     = mom.theta();
	  eta3 = -log(tan(th3/2));
  if(  lastpo.z()< zmilimit || (lastpo.z()>zmilimit && lastpo.perp()> 100.) ) {
	 varia3= 1;
  }
  else{
         varia3= 2;
  } 

	}
	else {
	  std::cout << "RecFP420Test:WARNING i>3" << std::endl; 
	}
	
	// primary vertex:
	vx = avertex->GetX0();
	vy = avertex->GetY0();
	vz = avertex->GetZ0();
	// TheHistManager->GetHisto("VtxX")->Fill(vx);
	// TheHistManager->GetHisto("VtxY")->Fill(vy);
	// TheHistManager->GetHisto("VtxZ")->Fill(vz);
      }
    }// prim.vertex loop end
//                                                                              preparations:
    double zref, xref,  yref,  bxtrue,  bytrue;
    double       xref2, yref2, bxtrue2, bytrue2;
    double       xref3, yref3, bxtrue3, bytrue3;
    
    //zref = vz;
    // zref = 0. + 0.5*zUnit;// center of 1st station is at 0.
    zref = 8000.;// last st.
    
    // 1st primary track 
    bxtrue = tan(th)*cos(phi);
    bytrue = tan(th)*sin(phi);
    xref = vx + (zref-vz)*bxtrue;
    yref = vy + (zref-vz)*bytrue;
    
    // 2st primary track 
    bxtrue2= tan(th2)*cos(phi2);
    bytrue2= tan(th2)*sin(phi2);
    xref2= vx + (zref-vz)*bxtrue2;
    yref2= vy + (zref-vz)*bytrue2;
    
    // 1st primary track 
    bxtrue3= tan(th3)*cos(phi3);
    bytrue3= tan(th3)*sin(phi3);
    xref3= vx + (zref-vz)*bxtrue3;
    yref3= vy + (zref-vz)*bytrue3;
    
    
    //                                                                              .
    //  double dref12 = abs(xref2 - xref);
    // double drefy12 = abs(yref2 - yref);
    //                                                                              .
    //	TheHistManager->GetHisto("xref")->Fill(xref);
    //	TheHistManager->GetHisto("xref2")->Fill(xref2);
    //	TheHistManager->GetHisto("dref12")->Fill(dref12);
    //	TheHistManager->GetHisto("drefy12")->Fill(drefy12);
    //	TheHistManager->GetHisto("yref")->Fill(yref);
    //	TheHistManager->GetHisto("yref2")->Fill(yref2);
    //	TheHistManager->GetHisto("thetaXmrad")->Fill(fabs(bxtrue)*1000.);
    //	TheHistManager->GetHisto("thetaX2mrad")->Fill(fabs(bxtrue2)*1000.);
    //=========================== thePrim != 0 ================================================================================
    //    if (thePrim != 0   && vz < -20.) {
    
    
    //ask 1 tracks	  	  
    
    
    if ( thePrim != 0 
	 &&	varia == 2  
	 && ((xref > -32. && xref < -12.) && (yref > -5. && yref < 5.))  
	 ) {
      
      
      //	     &&	varia == 2  
      //	     && ( fabs(bxtrue)*1000. > 0.1  && fabs(bxtrue)*1000.<0.4 )
      
      // ask 2 tracks		  
      
      
      
      /*	  
	if ( thePrim != 0 
	&& ((xref  > -32. && xref  < -12.) && (yref  > -5. && yref  < 5.))  
	&& ((xref2 > -32. && xref2 < -12.) && (yref2 > -5. && yref2 < 5.))  
	&& dref12 > 1.0 && drefy12 > 1.0       
	) {
      */
      
      
      //  &&	( varia == 2 && varia2 == 2 ) 
      //  && dref12 > 1.       
      //	     && (( fabs(bxtrue)*1000.>0.1)&&( fabs(bxtrue)*1000.<0.4) ) || (( fabs(bxtrue2)*1000. > 0.1)&&( fabs(bxtrue2)*1000.<0.4) )  
      
      
      /////////////////////////////////////////////////////////////////
      //      unsigned int clnumcut=1;// ask 2 tracks
      //  unsigned int clnumcut=0;//ask 1 tracks
      /////////////////////////////////////////////////////////////////
	// TheHistManager->GetHisto("PrimaryEta")->Fill(eta);
	//TheHistManager->GetHisto("PrimaryPhigrad")->Fill(phigrad);
	//TheHistManager->GetHisto("PrimaryTh")->Fill(th*1000.);// mlrad
	//TheHistManager->GetHisto("PrimaryLastpoZ")->Fill(lastpo.z());
	// ==========================================================================
	
	// hit map for FP420
	// ==================================
	
	map<int,float,less<int> > themap;
	map<int,float,less<int> > themap1;
	
	map<int,float,less<int> > themapxystrip;
	map<int,float,less<int> > themapxy;
	map<int,float,less<int> > themapz;
	// access to the G4 hit collections:  -----> this work OK:
	
	G4HCofThisEvent* allHC = (*evt)()->GetHCofThisEvent();
	
	if (verbosity > 0) {
	  std::cout << "RecFP420Test:  accessed all HC" << std::endl;;
	}
	int CAFIid = G4SDManager::GetSDMpointer()->GetCollectionID("FP420SI");
	
	FP420G4HitCollection* theCAFI = (FP420G4HitCollection*) allHC->GetHC(CAFIid);
	if (verbosity > 0) {
	  //std::cout << "FP420Test: theCAFI = " << theCAFI << std::endl;
	  std::cout << "RecFP420Test: theCAFI->entries = " << theCAFI->entries() << std::endl;
	}
	
	varia = 0;
	if( varia == 0  ) {
	  //varia

	  double  totallosenergy= 0.;
	  int AATest[80];
	  for (int j=0; j<80; j++) {
	    AATest[j]=0;
	  }
	  // loop over simhits
	  for (int j=0; j<theCAFI->entries(); j++) {
	    FP420G4Hit* aHit = (*theCAFI)[j];
	    
	    Hep3Vector hitEntryLocalPoint = aHit->getEntryLocalP();
	    Hep3Vector hitExitLocalPoint = aHit->getExitLocalP();
	    Hep3Vector hitPoint = aHit->getEntry();
	    unsigned int unitID = aHit->getUnitID();
	    // int trackIDhit  = aHit->getTrackID();
	    /*
	      double  elmenergy =  aHit->getEM();
	      double  hadrenergy =  aHit->getHadr();
	      double incidentEnergyHit  = aHit->getIncidentEnergy();
	      double   timeslice = aHit->getTimeSlice();     
	      int     timesliceID = aHit->getTimeSliceID();     
	      double  depenergy = aHit->getEnergyDeposit();
	      float   pabs = aHit->getPabs();
	      float   tof = aHit->getTof();
	    */
	    double  losenergy = aHit->getEnergyLoss();
	    /*
	      int   particletype = aHit->getParticleType();
	      float thetaEntry = aHit->getThetaAtEntry();   
	      float phiEntry = aHit->getPhiAtEntry();
	      float xEntry = aHit->getX();
	      float yEntry = aHit->getY();
	      float zEntry = aHit->getZ();
	      int  parentID = aHit->getParentId();
	      float vxget = aHit->getVx();
	      float vyget = aHit->getVy();
	      float vzget = aHit->getVz();
	    */
	    //        if(theCAFI->entries() == 20){
	    //     std::cout << "2020202020202020202020202020202020202020 " << std::endl;
#ifdef mmydebug
	    
	    std::cout << "======================Hit Collection" << std::endl;
	    std::cout << "lastpo.x() = " << lastpo.x() << std::endl;
	    std::cout << "lastpo.y() = " << lastpo.y() << std::endl;
	    std::cout << "lastpo.z() = " << lastpo.z() << std::endl;
	    std::cout << "hitPoint = " << hitPoint << std::endl;
	    std::cout << "hitEntryLocalPoint = " << hitEntryLocalPoint << std::endl;
	    std::cout << "hitExitLocalPoint = " << hitExitLocalPoint << std::endl;
	    std::cout << "elmenergy = " << elmenergy << "hadrenergy = " << hadrenergy << std::endl;
	    std::cout << "incidentEnergyHit = " << incidentEnergyHit << "trackIDhit = " << trackIDhit << std::endl;
	    std::cout << "unitID=" << unitID <<std::endl;
	    std::cout << "timeslice = " << timeslice << "timesliceID = " << timesliceID << std::endl;
	    std::cout << "depenergy = " << depenergy << "pabs = " << pabs  << std::endl;
	    std::cout << "tof = " << tof << "losenergy = " << losenergy << std::endl;
	    std::cout << "particletype = " << particletype << "thetaEntry = " << thetaEntry << std::endl;
	    std::cout << "phiEntry = " << phiEntry << "xEntry = " << xEntry  << std::endl;
	    std::cout << "yEntry = " << yEntry << "zEntry = " << zEntry << std::endl;
	    std::cout << "parentID = " << parentID << "vxget = " << vxget << std::endl;
	    std::cout << "vyget = " << vyget << "vzget = " << vzget << std::endl;
#endif
	    //////////    break;
	    //    }
	    //double th_hit    = hitPoint.theta();
	    //double eta_hit = -log(tan(th_hit/2));
	    double phi_hit   = hitPoint.phi();
	    if (phi_hit < 0.) phi_hit += twopi;
	    //double phigrad_hit = phi_hit*180./pi;
	    //UserNtuples->fillg60(eta_hit,losenergy);
	    //UserNtuples->fillg61(eta_hit,1.);
	    //UserNtuples->fillg62(phigrad_hit,losenergy);
	    //UserNtuples->fillg63(phigrad_hit,1.);
	    
	    // double   xx    = hitPoint.x();
	    // double   yy    = hitPoint.y();
	    double   zz    = hitPoint.z();
	    if (verbosity > 2) {
	      std::cout << "RecFP420Test:zHits = " << zz << std::endl;
	    }
	    themap[unitID] += losenergy;
	    totallosenergy += losenergy;
	    
	    int det, zside, sector, zmodule;
	    FP420NumberingScheme::unpackFP420Index(unitID, det, zside, sector, zmodule);
	    
	    //////////////                                                             //////////////
	    //test of # hits per every iitest:
	    // iitest   is a continues numbering of FP420
	    unsigned int iitest = 2*(pn0-1)*(sector - 1)+2*(zmodule - 1)+zside;
	    ++AATest[iitest-1];
	    //////////////                                                             //////////////
	    
	    // zside=1,2 ; zmodule=1,10 ; sector=1,3
	    if(zside==0||sector==0||zmodule==0){
	      std::cout << "RecFP420Test:ERROR: zside = " << zside  << " sector = " << sector  << " zmodule = " << zmodule  << " det = " << det  << std::endl;
	    }
	    
	    double kplane = -(pn0-1)/2+(zmodule-1); 
	    
	    
	    double zdiststat = 0.;
	    if(sector==2) zdiststat = zD2;
	    if(sector==3) zdiststat = zD3;
	    double zcurrent = -150. +(ZSiStep-ZSiPlane)/2  + kplane*ZSiStep + zdiststat;  
	    
	    
	    if(zside==1){
	      zcurrent += (ZGapLDet+ZSiDetL/2);
	    }
	    if(zside==2){
	      //     zcurrent += (ZGapLDet+ZSiDetL+ZBoundDet+ZSiDetR/2);
	      zcurrent += (ZGapLDet+ZSiDetR/2)+ZSiPlane/2;
	    }     
	    
	    //=======================================
	    // SimHit position in Local reference frame - middle :
	    G4ThreeVector middle = (hitExitLocalPoint+hitEntryLocalPoint)/2.;
	    G4ThreeVector mid = (hitExitLocalPoint-hitEntryLocalPoint)/2.;
	    
#ifdef mmydebug
	    int sScale = 2*(pn0-1); // intindex is a continues numbering of FP420
	    int zScale = 2;   unsigned int intindex = sScale*(sector - 1)+zScale*(zmodule - 1)+zside; //intindex=1-30:X,Y,X,Y,X,Y...
	    
	    // int zScale = 10;   unsigned int intindex = sScale*(sector - 1)+zScale*(zside - 1)+zmodule; //intindex=1-30:XXXXXXXXXX,YYYYYYYYYY,...
	    std::cout << " ***********RecFP420Test::1    index = " << unitID  << " iu = " << intindex  << std::endl;
	    std::cout << " zside = " << zside  << " sector = " << sector  << " zmodule = " << zmodule  << std::endl;
	    std::cout << " hitPoint.z() = " << hitPoint.z() << std::endl;
	    std::cout << " hitExitLocalPoint   = " << hitExitLocalPoint  << std::endl;
	    std::cout << " hitEntryLocalPoint   = " << hitEntryLocalPoint  << std::endl;
	    std::cout << " hitPoint   = " << hitPoint  << std::endl;
	    std::cout << " middle.x() = " << middle.x() << " middle.y() = " << middle.y() << " middle.z() = " << middle.z() << std::endl;
	    std::cout << " mid.x() = " << mid.x() << " mid.y() = " << mid.y() << " mid.z() = " << mid.z() << std::endl;
	    std::cout << " themapz+ZSiDetL/2 = " << hitPoint.z()+ZSiDetL/2 << std::endl;
	    std::cout << " zcurrent = " << zcurrent << std::endl;
	    //std::cout << "  ZSiPlane= " << ZSiPlane << "  ZSiStep= " << ZSiStep << "  zUnit= " << zUnit << std::endl;
	    std::cout << "  ZSiPlane= " << ZSiPlane << "  ZSiStep= " << ZSiStep << "  zD2= " << zD2 << "  zD3= " << zD3 << std::endl;
	    std::cout << "  ZGapLDet= " << ZGapLDet << "  ZSiDetL= " << ZSiDetL << std::endl;
#endif
	    //
	    if (verbosity > 2) {
	      std::cout << "RecFP420Test:check " << std::endl;
	      std::cout << " zside = " << zside  << " sector = " << sector  << " zmodule = " << zmodule  << std::endl;
	    }
	    //=======================================
	    //=======================================!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	    //=======================================
#ifdef mmydebug
	    std::cout << " zside = " << zside  << " sector = " << sector  << " zmodule = " << zmodule  << std::endl;
	    std::cout << " hitPoint.z()+ mid.z() = " << hitPoint.z()+ mid.z() << std::endl;
	    std::cout << " zcurrent = " << zcurrent << " det = " << det << std::endl;
	    std::cout << " diff = " << hitPoint.z()+ mid.z()- zcurrent << std::endl;
#endif
	    //    themapz[unitID]  = hitPoint.z()+ mid.z(); // this line just for studies
	    themapz[unitID]  = zcurrent;// finally must be this line !!!
	    //=======================================
	    //=======================================!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	    //=======================================
	    
	    themapxystrip[unitID] = -1.;// charge in strip coord 
	    
	    float numStrips=-1.,pitch=1.;
	    //=======================================
	    // Y global
	    if(zside==1) {
	      //UserNtuples->fillg24(losenergy,1.);
	      if(losenergy > 0.00003) {
		themap1[unitID] += 1.;
	      }
	      numStrips = numStripsY;
	      pitch=pitchY;
	      
	      themapxystrip[unitID] = 0.5*(numStrips-1) + middle.x()/pitch ;// charge in strip coord 
	      
	      themapxy[unitID]  = (numStrips-1)*pitch/2. + middle.x();// hit coordinate in l.r.f starting at bot edge of plate
	    }
	    //X
	    if (verbosity > 2) {
	      std::cout << "RecFP420Test:check1111 " << std::endl;
	    }
	    if(zside==2){
	      //UserNtuples->fillg25(losenergy,1.);
	      if(losenergy > 0.00003) {
		themap1[unitID] += 1.;
	      }
	      numStrips = numStripsX;
	      
	      //themapxystrip[unitID]  = int(fabs(middle.y()/pitch + 0.5*numStrips + 1.));// nearest strip number
	      themapxystrip[unitID] = 0.5*(numStrips-1) + middle.y()/pitch ;// charge in strip coord 
	      if (verbosity > 2) {
		std::cout << "RecFP420Test:check2222 " << std::endl;
	      }
	      
	      themapxy[unitID]  = (numStrips-1)*pitch/2. + middle.y(); // hit coordinate in l.r.f starting at left edge of plate 
	    }
	    //	   }
	    //
	    
	  }  // for loop on all hits ENDED  ENDED  ENDED  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	  // =========

   //     !!!!!!!!!!!!!
   //     !!!!!!!!!!!!!
   //     !!!!!!!!!!!!!



//=================================================================================
  //
  //                                                                 DIGI:                                                   .
  //                                                                 DIGI:                                                   .
  //                                                                 DIGI:                                                   .
  //                                                                 DIGI:                                                   .
  //                                                                 DIGI:                                                   .
  //                                                                 DIGI:                                                   .
  //                                                                 DIGI:                                                   .
  //
//=================================================================================
                                                                                                 


          
//====================================================================================================== number of hits
  int tothits=0;
  for (int j=0; j<80; j++) {
    tothits += AATest[j];
    if(tothits!=0) break;
  }
                               if(tothits  > 0) {
// ===============  produce Digi start
//    produce();
#ifdef mydigidebug10
   std::cout <<" ===== RecFP420Test:: call produce of DigitizerFP420" << std::endl;
#endif

   //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
   theDigitizerFP420->produce(theCAFI,output);

    if (verbosity > 2) {
      std::cout << "RecFP420Test:  start of access to the collector" << std::endl;
    }
  for (int sector=1; sector<sn0; sector++) {
    for (int zmodule=1; zmodule<pn0; zmodule++) {
      for (int zside=1; zside<3; zside++) {
	//int det= 1;
	//int index = FP420NumberingScheme::packFP420Index(det, zside, sector, zmodule);

	// intindex is a continues numbering of FP420
	int sScale = 2*(pn0-1);
	int zScale=2;  unsigned int iu = sScale*(sector - 1)+zScale*(zmodule - 1)+zside;
	// int zScale=10;	unsigned int intindex = sScale*(sector - 1)+zScale*(zside - 1)+zmodule;

	std::vector<HDigiFP420> collector;
	collector.clear();
	DigiCollectionFP420::Range outputRange;
	outputRange = output.get(iu);
  // fill output in collector vector (for may be sorting? or other checks)
  DigiCollectionFP420::ContainerIterator sort_begin = outputRange.first;
  DigiCollectionFP420::ContainerIterator sort_end = outputRange.second;

  for ( ;sort_begin != sort_end; ++sort_begin ) {
    collector.push_back(*sort_begin);
  } // for

#ifdef mydigidebug10
std::cout << "DigiCollectionFP420:: collector.size() = " << collector.size() << std::endl;
#endif


   vector<HDigiFP420>::const_iterator simHitIter = collector.begin();
   vector<HDigiFP420>::const_iterator simHitIterEnd = collector.end();
   for (;simHitIter != simHitIterEnd; ++simHitIter) {
     //  const HDigiFP420 istrip = *simHitIter;
     // Y:
     if(zside==1){
     //  double moduleThickness = ZSiDetL; 
     //  float sigmanoise =  ENC*moduleThickness/Thick300/ElectronPerADC;
     //  TheHistManager->GetHisto("DigiYstrip")->Fill(istrip.strip());
     //  TheHistManager->GetHisto("DigiYstripAdc")->Fill(istrip.adc());
     //  TheHistManager->GetHisto("DigiYstripAdcSigma")->Fill(istrip.adc()/sigmanoise);
     }
     // X:
     else if(zside==2){
//	    double moduleThickness = ZSiDetR; 
//	    float sigmanoise =  ENC*moduleThickness/Thick300/ElectronPerADC;
//	 TheHistManager->GetHisto("DigiXstrip")->Fill(istrip.strip());
//	 TheHistManager->GetHisto("DigiXstripAdc")->Fill(istrip.adc());
//	 TheHistManager->GetHisto("DigiXstripAdcSigma")->Fill(istrip.adc()/sigmanoise);
     }
#ifdef mydigidebug10
//     if(zside == 1) {
 std::cout << " RecFP420Test::check: HDigiFP420::  " << std::endl;
 // std::cout << " strip number = " << (*simHitIter).strip() << "  adc = " << (*simHitIter).adc() << std::endl;
 // std::cout << " strip number = " << simHitIter->strip() << "  adc = " << simHitIter->adc() << std::endl;
 std::cout << " strip number = " << istrip.strip() << "  adc = " << istrip.adc() << std::endl;
 std::cout << " channel = " << istrip.channel() << std::endl;
 // std::cout << " strip number = " << astrip->strip() << "  adc = " << astrip->adc() << std::endl;
 //    }
#endif
   }//for (;simHitIter
   //==================================
      }   // for
    }   // for
  }   // for
#ifdef mydigidebug10
  std::cout << " RecFP420Test::end of access to the strip collection " << std::endl;
  std::cout << " RecFP420Test:: end DIGI " << std::endl;
#endif

  //     end of check of access to the strip collection
// =======================================================================================check of access to strip collection
// ==============================================================================================
// ==============================================================================================

// ==============================================================================================
//                                       CLUSTERS:                                                               =START
//                                       CLUSTERS:                                                               =START
//                                       CLUSTERS:                                                               =START
// ==============================================================================================

    theClusterizerFP420->produce(output,soutput);
   //==================================a

    for (int sector=1; sector < sn0; sector++) {
      for (int zmodule=1; zmodule<pn0; zmodule++) {
	for (int zside=1; zside<3; zside++) {

	  // index is a continues numbering of 3D detector of FP420
	  //int det= 1;
	  //int index = FP420NumberingScheme::packFP420Index(det, zside, sector, zmodule);
	  
	  int sScale = 2*(pn0-1);
	  int zScale=2;  unsigned int iu = sScale*(sector - 1)+zScale*(zmodule - 1)+zside;

	  //============================================================================================================
	  std::vector<ClusterFP420> collector;
	  collector.clear();
	  ClusterCollectionFP420::Range outputRange;
	  outputRange = soutput.get(iu);
	  // fill output in collector vector (for may be sorting? or other checks)
	  ClusterCollectionFP420::ContainerIterator sort_begin = outputRange.first;
	  ClusterCollectionFP420::ContainerIterator sort_end = outputRange.second;
	  for ( ;sort_begin != sort_end; ++sort_begin ) {
	    collector.push_back(*sort_begin);
	  } // for
	  //============================================================================================================
	  /*	  
	  vector<ClusterFP420>::const_iterator simHitIter = collector.begin();
	  vector<ClusterFP420>::const_iterator simHitIterEnd = collector.end();
	  int icl = 0;
	  float clampmax = 0.;
	  ClusterFP420 iclustermax;
	  // loop in #clusters
	  for (;simHitIter != simHitIterEnd; ++simHitIter) {
	    const ClusterFP420 icluster = *simHitIter;
	    icl++;
	    float deltastrip = themapxystrip[index] - (icluster.barycenter()  );
	    float clsize =   icluster.amplitudes().size();
	  } // for loop in #clusters
*/
	  //================================== end of for loops in continuius number iu:
	}   // for
      }   // for
    }   // for
    


   //==================================
//                     CLUSTERS:                                                               =END

// ==============================================================================================
// ==============================================================================================


//                                                                                                   TrackReconstruction:                                          .
//                                                                               TrackReconstruction:                                                    .
////                                                                                          access to Tracks

 //
   //==================================
    theTrackerizerFP420->produce(soutput,toutput);
   //==================================
 std::vector<TrackFP420> collector;
 collector.clear();
 TrackCollectionFP420::Range outputRange;
 int StID = 1111;
 outputRange = toutput.get(StID);
 //
 // fill output in collector vector (for may be sorting? or other checks)
 //
 TrackCollectionFP420::ContainerIterator sort_begin = outputRange.first;
 TrackCollectionFP420::ContainerIterator sort_end = outputRange.second;
 //
 for ( ;sort_begin != sort_end; ++sort_begin ) {
   collector.push_back(*sort_begin);
 } // for  sort_begin
 TheHistManager->GetHisto("ntrackscoll")->Fill(collector.size());



 //                                                                          .
 //                      loop in #tracks                      loop in #tracks                              loop in #tracks
 //
 vector<TrackFP420>::const_iterator simHitIter = collector.begin();
 vector<TrackFP420>::const_iterator simHitIterEnd = collector.end();
 //
 //
 //
 //           loop in #tracks            loop in #tracks          loop in #tracks
 //
 int ntracks = 0;
 for (;simHitIter != simHitIterEnd; ++simHitIter) {
   ++ntracks;
   
   //
#ifdef myTrackdebug10
   const TrackFP420 itrack = *simHitIter;
   std::cout << "RecFP420Test::check: nclusterx = " << itrack.nclusterx() << "  nclustery = " << itrack.nclustery() << std::endl;
   std::cout << "  ax = " << itrack.ax() << "  bx = " << itrack.bx() << std::endl;
   std::cout << "  ay = " << itrack.ay() << "  by = " << itrack.by() << std::endl;
   std::cout << " chi2x= " << itrack.chi2x() << " chi2y= " << itrack.chi2y() << std::endl;
   std::cout <<" ntracks=" << ntracks << std::endl;
   std::cout <<" ===" << std::endl;
   std::cout <<" =======================" << std::endl;
#endif
   //
 }//   for  simHitIter
 //           loop in #tracks      ENDED      loop in #tracks     ENDED     loop in #tracks      ENDED
 //
 //if(ntracks>1) {
 //}
 ////////////////////////////////////   Track finished
			       }   // if(tothits
			       else{
				 //#ifdef mydebug10
				 std::cout << "Else: tothits =0 " << std::endl;
				 //#endif
			       }
	  
	} // varia MIonly or noMIonly ENDED
	else{
	  //#ifdef mydebug10
	  std::cout << "Else: varia: MI or no MI or all " << std::endl;
	  //#endif
	}

    }                                                // primary end
    else{
      //#ifdef mydebug10
      std::cout << "Else: primary  " << std::endl;
      //#endif
    }
    //=========================== thePrim != 0  end   ===========================
    
}


