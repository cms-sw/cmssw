// -*- C++ -*-
//
// Package:    AlCaHOCalibProducer
// Class:      AlCaHOCalibProducer
// 
/**\class AlCaHOCalibProducer AlCaHOCalibProducer.cc Calibration/AlCaHOCalibProducer/src/AlCaHOCalibProducer.cc

change magnetic field inside 
../data/HOCosmicCalib_RecoLocalMuon.cff
../data/HOCosmicCalib_RecoLocalTracker.cff



 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
Missing towers : eta=5, phi=18-19
               : eta = -5, phi =11-14

HO tile sizes
Ring +-2 : width  Tray 6:404.6, 5&4:347.6, 3:352.6, 2:364.6, 1:315.6 
                  (phi ordering is opposite) 
           lenght Tile 1:420.1, 2:545.1, 3:583.3, 4:626.0, 5:335.5  

                   (five tiles, 1 is close to Ring 1 and 5 is towardslc endcap)
Ring +-1 : width  Tray 6:404.6, 5&4:347.6, 3:352.6, 2:364.6, 1:315.6  (same as Ring+-2)
           lenght Tile 1:391.5, 2:394.2, 3:411.0, 4:430.9, 5:454.0, 6:426.0
                  (1: near R0 and 6 near R2)

Ring 0 L1 : Width Tray (6:290.6, 5&4:345.6, 3:350.6, 2:362.6, 1:298.6  
            lenght 1:351.2, 2:353.8, 3:359.2, 4:189.1 (4 is towards Ring1)

Ring 0 L0 : Width Tray 6:266.6, 5&4:325.6, 3:330.6, 2:341.6, 1:272.6
            length 1:331.5, 2:334.0, 3:339.0, 4:248.8 (4 is towards Ring1)


*/
//
// Original Author:  Gobinda Majumder
//         Created:  Fri Jul  6 17:17:21 CEST 2007
// $Id: AlCaHOCalibProducer.cc,v 1.14 2008/07/17 12:10:09 kodolova Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
// #include "FWCore/Framework/interface/Handle.h"
#include "DataFormats/FWLite/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/DTGeometry/interface/DTSuperLayer.h"
#include "DataFormats/DTRecHit/interface/DTRecHitCollection.h"


#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"

#include "CalibCalorimetry/HcalAlgos/interface/HcalAlgoUtils.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrationWidths.h"

//08/07/07 #include "CondTools/Hcal/interface/HcalDbPool.h"
//#include "CondFormats/HcalObjects/interface/HcalPedestals.h"
//#include "CondFormats/HcalObjects/interface/HcalPedestalWidths.h"


// #include "TrackingTools/GeomPropagators/interface/HelixArbitraryPlaneCrossing.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
//#include "Geometry/Surface/interface/PlaneBuilder.h"
#include "DataFormats/GeometrySurface/interface/PlaneBuilder.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "DataFormats/HcalCalibObjects/interface/HOCalibVariables.h"
// #include "DataFormats/HOCalibHit/interface/HOCalibVariableCollection.h"
#include "DataFormats/Math/interface/Error.h"
#include "CLHEP/Vector/LorentzVector.h"

//#include "DataFormats/Candidate/interface/Candidate.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"

#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/Math/interface/Error.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"
//#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixStateInfo.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Framework/interface/TriggerNames.h"


#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TProfile.h"
#include "TTree.h"
/* C++ Headers */
#include <string>

#include <iostream>
#include <fstream>
//
// class decleration
//
using namespace std;
using namespace edm;
using namespace reco;

const int netabin= 16;  
const int nphimx = 72;
const int ndepthmx=4;
//const int ncapmx = 4;
const int netamx = 32;
const int nchnmx = 10;
const int ncidmx = 5;
const int nsigpk = 7;
const int nstrbn = 0; //0 <= nstrbn <= nchnmx-nsigpk
const int ntrgp = 11;


const int netahbmx = 60;
const int netahb3mx = 32;

static const unsigned int nL1trg = 200;

static const unsigned int nL1mx=140;
static const unsigned int nHLTmx=140;

class AlCaHOCalibProducer : public edm::EDProducer {
   public:
      explicit AlCaHOCalibProducer(const edm::ParameterSet&);
      ~AlCaHOCalibProducer();

    typedef Basic3DVector<float>   PositionType;
    typedef Basic3DVector<float>   DirectionType;
    typedef Basic3DVector<float>   RotationType;


   private:
      void findHOEtaPhi(int iphsect, int& ietaho, int& iphiho);
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // ----------member data ---------------------------

  float  xhor0; //x-position in ring 0
  float  yhor0; //y-position in ring 0
  float  xhor1; //x-position in ring 1
  float  yhor1; //y-position in ring 1   
  int iring;    //Ring number -2,-1,0,1,2

  float  localxhor0; //local x-distance from edege in ring 0
  float  localyhor0; //local y-distance from edege in ring 0
  float  localxhor1; //local x-distance from edege in ring 1
  float  localyhor1; //local y-distance from edege in ring 1

  float pedestal[netamx][nphimx][ncidmx]; 
  //  float pedestal_lib[netamx][nphimx][ncidmx]; 
  /*
  float pedestal[32][72][4][4];
  float pedestalHB[32][72][4][4];
  static const int netabin= 16;  
  static const int nphimx = 72;
  static const int ndepthmx=4;
  static const int ncapmx = 4;
  static const int netamx = 32;
  */
  float ho_time[nchnmx];

  float hb_time[nchnmx];
  std::string digiLabel;
  
  bool debug;
  

  std::string theRootFileName;
  //  std::string thePedestalFile;
  //  TFile* theFile;
#ifdef __DO_HISTOGRAMMING__
  TH1F* libhoped;
  TH1F* allhotime; // [netamx][nphimx];
  TH1F* hotime[ntrgp+1]; // [netamx][nphimx];
  TH1F* hopedtime; // [netamx][nphimx];

  TProfile* hopedpr;  
  TH1F* hopedrms;  
  TH1F* hst_hopedrms;    

  TProfile* hopeak[ntrgp+1];
  TProfile* horatio;

  TH1F* Nallhotime; // [netamx][nphimx];
  TH1F* Nhotime[ntrgp+1]; // [netamx][nphimx];
  TH1F* Nhopedtime; // [netamx][nphimx];

  TH1F* allhb1;
  TH1F* allhb2;
  TH1F* allhb3;

  TH1F* Nallhb1;
  TH1F* Nallhb2;
  TH1F* Nallhb3;

  TProfile* hb1pedpr;  
  TH1F* hb1pedrms;  
  TH1F* hst_hb1pedrms;    

  TH1F* ho_occupency[5];  
#endif 

  bool m_hotime;

  edm::InputTag muonTags_;   // cosmicMuons or standAloneMuons
  edm::InputTag hbheLabel_;
  edm::InputTag hoLabel_;
  edm::InputTag hltLabel_;
  edm::InputTag l1Label_;  
  edm::InputTag towerLabel_;    

  bool m_digiInput;            // digi (true) or rechit (false)
  bool m_hbinfo;
  int m_startTS;
  int m_endTS;    
  double m_magscale;
  double m_sigma;

  //  const HcalQIEShape* m_shape;
  //  const HcalQIECoder* m_coder;

  typedef math::Error<5>::type CovarianceMatrix;  
  int Nevents;
  int Npass;
  int irunold;
  int nRuns;
  SteppingHelixPropagator* stepProp;
  FreeTrajectoryState getFreeTrajectoryState( const reco::Track& tk, const MagneticField* field, int itag, bool dir);

  edm::ESHandle<HcalDbService> conditions_;
  const HcalQIEShape* m_shape;
  const HcalQIECoder* m_coder;

  HcalCalibrations calibped;
  HcalCalibrationWidths calibwidth;

  unsigned int Ntp; // # of HLT trigger paths (should be the same for all events!)
  std::map<std::string, bool> fired; 

  //  std::vector<unsigned int> l1Accepts_; // number of events accepted by each L1 algorithm
  //  std::vector<std::string> l1Names_;    // name of each L1 algorithm

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
AlCaHOCalibProducer::AlCaHOCalibProducer(const edm::ParameterSet& iConfig)
  :  muonTags_(iConfig.getUntrackedParameter<edm::InputTag>("muons"))

{
   //register your products
/* Examples
   produces<ExampleData2>();

   //if do put with a label
   produces<ExampleData2>("label");
*/

  theRootFileName = iConfig.getUntrackedParameter<string>("RootFileName","tmp.root");
  m_digiInput = iConfig.getUntrackedParameter<bool>("digiInput", true);
  m_hbinfo = iConfig.getUntrackedParameter<bool>("hbinfo", false);
  m_startTS = iConfig.getUntrackedParameter<int>("firstTS", 4);
  m_hotime = iConfig.getUntrackedParameter<bool>("hotime", false);
  if(m_startTS<0) m_startTS=0;
  m_endTS = iConfig.getUntrackedParameter<int>("lastTS", 7);
  if (m_endTS < m_startTS) m_endTS = m_startTS + 3;
  if (m_endTS >9) m_endTS=9;
  m_magscale = iConfig.getUntrackedParameter<double>("m_scale", 4.0);
  m_sigma = iConfig.getUntrackedParameter<double>("sigma", 1.0);

  hoLabel_ = iConfig.getParameter<edm::InputTag>("hoInput");
  hbheLabel_ = iConfig.getParameter<edm::InputTag>("hbheInput");
  hltLabel_ = iConfig.getParameter<edm::InputTag>("hltInput");
  l1Label_ = iConfig.getParameter<edm::InputTag>("l1Input");
  towerLabel_ = iConfig.getParameter<edm::InputTag>("towerInput");  

  produces<HOCalibVariableCollection>("HOCalibVariableCollection").setBranchAlias("HOCalibVariableCollection");
  //  produces<TrackCollection>("TrackCollection");

  
#ifdef __DO_HISTOGRAMMING_
_ 
  edm::Service<TFileService> fs;

  for (int i=0; i<nchnmx; i++) {ho_time[i] = hb_time[i] = 0.0;}

  char title[200];
  if (m_hotime && m_digiInput) {
    //    theFile = new TFile(theRootFileName.c_str(), "RECREATE");
    //    theFile->cd();

    libhoped = fs->make<TH1F>("libhoped", "libhoped", ncidmx*netamx*nphimx, -0.5, ncidmx*netamx*nphimx-0.5);
    
    allhotime = fs->make<TH1F>("allhotime", "allhotime", nchnmx*netamx*nphimx, -0.5, nchnmx*netamx*nphimx-0.5);
    
    for (int ij=0; ij<=ntrgp; ij++) {
       sprintf(title, "hotime_trgp_%i", ij+1);
       hotime[ij] = fs->make<TH1F>(title, title, nchnmx*netamx*nphimx, -0.5, nchnmx*netamx*nphimx-0.5);
       
       sprintf(title, "hopeak_trgp_%i", ij+1);
       hopeak[ij] = fs->make<TProfile>(title, title,netamx*nphimx, -0.5, netamx*nphimx-0.5);    
    }

    horatio = fs->make<TProfile>("horatio", "horatio",netamx*nphimx, -0.5, netamx*nphimx-0.5);    

    hopedtime = fs->make<TH1F>("hopedtime", "hopedtime", nchnmx*netamx*nphimx, -0.5, nchnmx*netamx*nphimx-0.5);
    

    Nallhotime = fs->make<TH1F>("Nallhotime", "Nallhotime", nchnmx*netamx*nphimx, -0.5, nchnmx*netamx*nphimx-0.5);
    
    hopedpr = fs->make<TProfile>("hopedpr", "hopedpr", nchnmx*netamx*nphimx, -0.5, nchnmx*netamx*nphimx-0.5);
    
    hopedrms = fs->make<TH1F>("hopedrms", "hopedrms", nchnmx*netamx*nphimx, -0.5, nchnmx*netamx*nphimx-0.5);
    
    hst_hopedrms = fs->make<TH1F>("hst_hopedrms", "hst_hopedrms", 100, 0.0, 0.1);


    for (int ij=0; ij<=ntrgp; ij++) {
      sprintf(title, "Nhotime_trgp_%i", ij+1);
      Nhotime[ij] = fs->make<TH1F>(title, title, nchnmx*netamx*nphimx, -0.5, nchnmx*netamx*nphimx-0.5);
    }
    
    Nhopedtime = fs->make<TH1F>("Nhopedtime", "Nhopedtime", nchnmx*netamx*nphimx, -0.5, nchnmx*netamx*nphimx-0.5);
    
    allhb1 = fs->make<TH1F>("allhb1", "allhb1", nchnmx*netahbmx*nphimx, -0.5, nchnmx*netahbmx*nphimx-0.5);
    
    allhb2 = fs->make<TH1F>("allhb2", "allhb2", nchnmx*netahb3mx*nphimx, -0.5, nchnmx*netahb3mx*nphimx-0.5); 
    
    allhb3 = fs->make<TH1F>("allhb3", "allhb3", nchnmx*netahb3mx*nphimx, -0.5, nchnmx*netahb3mx*nphimx-0.5); 
    
    Nallhb1 = fs->make<TH1F>("Nallhb1", "Nallhb1", nchnmx*netahbmx*nphimx, -0.5, nchnmx*netahbmx*nphimx-0.5);
    
    Nallhb2 = fs->make<TH1F>("Nallhb2", "Nallhb2", nchnmx*netahb3mx*nphimx, -0.5, nchnmx*netahb3mx*nphimx-0.5);
    
    Nallhb3 = fs->make<TH1F>("Nallhb3", "Nallhb3", nchnmx*netahb3mx*nphimx, -0.5, nchnmx*netahb3mx*nphimx-0.5);  
    hb1pedpr = fs->make<TProfile>("hb1pedpr", "hb1pedpr", nchnmx*netahbmx*nphimx, -0.5, nchnmx*netahbmx*nphimx-0.5);
    hb1pedrms = fs->make<TH1F>("hb1pedrms", "hb1pedrms", nchnmx*netahbmx*nphimx, -0.5, nchnmx*netahbmx*nphimx-0.5);
    hst_hb1pedrms = fs->make<TH1F>("hst_hb1pedrms", "hst_hb1pedrms", 100, 0., 0.1);
    
  }
    
  for (int i=0; i<5; i++) {
    sprintf(title, "ho_occupency (>%i #sigma)", i); 
    ho_occupency[i] = fs->make<TH1F>(title, title, netamx*nphimx, -0.5, netamx*nphimx-0.5); 
  }
  
#endif

}


AlCaHOCalibProducer::~AlCaHOCalibProducer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

  // Write the histos to file

#ifdef __DO_HISTOGRAMMING__

  if (m_hotime && m_digiInput) {
    //    theFile->cd();
    
    allhotime->Divide(Nallhotime);
    for (int ij=0; ij<=ntrgp; ij++) {
      hotime[ij]->Divide(Nhotime[ij]);
    }

    hopedtime->Divide(Nhopedtime);

    libhoped->Scale(1./max(1,nRuns));

    for (int i=0; i<nchnmx*netamx*nphimx; i++) {
      float xx = hopedpr->GetBinError(i+1);
      if (hopedpr->GetBinEntries(i+1) >0) {
	hopedrms->Fill(i, xx);
	hst_hopedrms->Fill(xx);
      }
    }
    
    allhb1->Divide(Nallhb1);
    allhb2->Divide(Nallhb2);
    allhb3->Divide(Nallhb3);
    
    for (int i=0; i<nchnmx*netahbmx*nphimx; i++) {
      float xx = hb1pedpr->GetBinError(i+1);
      if (hb1pedpr->GetBinEntries(i+1) >0) {
	hb1pedrms->Fill(i, xx);
	hst_hb1pedrms->Fill(xx);
      }
    }  
    
  }
  
  //  theFile->Write();
  //  theFile->Close();
#endif
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
AlCaHOCalibProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  using namespace edm;
  int irun = iEvent.id().run();
  //31/03/08 move it to ::beginjob()
  if (m_digiInput) {
    if (irunold !=irun)  { 
      iSetup.get<HcalDbRecord>().get(conditions_);
      m_shape = (*conditions_).getHcalShape();
    }
  }

  if (m_hotime && m_digiInput) {
    if (irunold !=irun) {
      nRuns++;
      for (int i =-netabin+1; i<=netabin-1; i++) {
	if (i==0) continue;
	int tmpeta1 =  (i>0) ? i -1 : -i +14; 
	if (tmpeta1 <0 || tmpeta1 >netamx) continue;
	for (int j=0; j<nphimx; j++) {
	  
	  HcalDetId id(HcalOuter, i, j+1, 4);
	  //	  conditions_->makeHcalCalibration(id,&calibped); //CMSSW_1_8_4
	  calibped = conditions_->getHcalCalibrations(id);

	  for (int k =0; k<ncidmx-1; k++) {
	    pedestal[tmpeta1][j][k] = calibped.pedestal(k); // pedm->getValue(k);
	    pedestal[tmpeta1][j][ncidmx-1] += (1./(ncidmx-1))*pedestal[tmpeta1][j][k];
	  }
#ifdef __DO_HISTOGRAMMING__
	  for (int k =0; k<ncidmx; k++) {
	    libhoped->Fill(nphimx*ncidmx*tmpeta1 + ncidmx*j + k, pedestal[tmpeta1][j][k]);
	    //	    if (k==0) {cout <<"ij "<<nphimx*tmpeta1+j<<" "<<tmpeta1<<" "<<i<<" "<<j+1<<endl;}

	  }
#endif
	}
      }
    }
  }

  Nevents++;
  irunold = irun;

  //  int irun = iEvent.id().run();
  //  int ievt = iEvent.id().event();
  if (Nevents%500==1) 
  cout <<"AlCaHOCalibProducer Processing event # "<<Nevents<<endl;
  std::auto_ptr<HOCalibVariableCollection> hostore (new HOCalibVariableCollection);

  edm::Handle<HODigiCollection> ho;   
  bool isHO = true;
  
  edm::Handle<HBHEDigiCollection> hbhe; 
  bool isHBHE = true;

  if (m_digiInput) {
    try {
      //GMA      iEvent.getByType(ho);
      iEvent.getByLabel(hoLabel_,ho);
    } catch ( cms::Exception &iEvent ) { isHO = false; } 
    
    try {
      //GMA      iEvent.getByType(hbhe); 
      iEvent.getByLabel(hbheLabel_,hbhe);
    } catch ( cms::Exception &iEvent ) { isHBHE = false; }  
  }

  if (m_hotime && m_digiInput) {
    if (isHO && (*ho).size()>0) {
      for (HODigiCollection::const_iterator j=(*ho).begin(); j!=(*ho).end(); j++){
	const HODataFrame digi = (const HODataFrame)(*j);
	HcalDetId id =digi.id();
	int tmpeta= id.ieta();
	int tmpphi= id.iphi();
	m_coder = (*conditions_).getHcalCoder(id);

	float tmpdata[nchnmx];
	int tmpeta1 = (tmpeta>0) ? tmpeta -1 : -tmpeta +14; 
	for (int i=0; i<digi.size() && i<nchnmx; i++) {
	  tmpdata[i] = m_coder->charge(*m_shape,digi.sample(i).adc(),digi.sample(i).capid());
#ifdef __DO_HISTOGRAMMING__
	  allhotime->Fill(nphimx*nchnmx*tmpeta1 + nchnmx*(tmpphi-1) + i, tmpdata[i]);
	  Nallhotime->Fill(nphimx*nchnmx*tmpeta1 + nchnmx*(tmpphi-1) + i, 1.);
#endif
	}
      }
    }
    
    if (isHBHE && (*hbhe).size()>0) {
      for (HBHEDigiCollection::const_iterator j=(*hbhe).begin(); j!=(*hbhe).end(); j++){
	const HBHEDataFrame digi = (const HBHEDataFrame)(*j);
	HcalDetId id =digi.id();
	int tmpeta= id.ieta();
	int tmpphi= id.iphi();
	int tmpdepth =id.depth();
	m_coder = (*conditions_).getHcalCoder(id);

	int tmpeta1 =  (tmpeta>0) ? tmpeta -15 : -tmpeta + 1; 
	if (tmpdepth==1) tmpeta1 =  (tmpeta>0) ? tmpeta -1 : -tmpeta +29;  
	
	for (int i=0; i<digi.size() && i<nchnmx; i++) {
	  float signal = m_coder->charge(*m_shape,digi.sample(i).adc(),digi.sample(i).capid());

#ifdef __DO_HISTOGRAMMING__
	  if (tmpdepth==1) { 
	    allhb1->Fill(nphimx*nchnmx*tmpeta1 + nchnmx*(tmpphi-1) + i, signal);
	    Nallhb1->Fill(nphimx*nchnmx*tmpeta1 + nchnmx*(tmpphi-1) + i, 1);
	    hb1pedpr->Fill(nphimx*nchnmx*tmpeta1 + nchnmx*(tmpphi-1) + i, signal);}

	  if (tmpdepth==2) { 
	    allhb2->Fill(nphimx*nchnmx*tmpeta1 + nchnmx*(tmpphi-1) + i, signal);
	    Nallhb2->Fill(nphimx*nchnmx*tmpeta1 + nchnmx*(tmpphi-1) + i, 1);}
	  
	  if (tmpdepth==3) { 
	    allhb3->Fill(nphimx*nchnmx*tmpeta1 + nchnmx*(tmpphi-1) + i, signal);
	    Nallhb3->Fill(nphimx*nchnmx*tmpeta1 + nchnmx*(tmpphi-1) + i, 1);}
#endif
	}
      }
    }
  }
  //  cout <<"3ng event # "<<Nevents<<endl;

  double pival = acos(-1.);

  Handle<reco::TrackCollection> cosmicmuon;
  bool isMuon = true;
  try {
    iEvent.getByLabel(muonTags_, cosmicmuon);
    //    iEvent.getByLabel("ALCARECOHcalCalZMuMu","StandAlone", cosmicmuon);

  } catch ( cms::Exception &iEvent ) { isMuon = false; } 
    //cout <<"ismuon "<<(int)isMuon<<" "<<cosmicmuon->size()<<endl;

  if (isMuon && cosmicmuon->size()>0) { 
    
   int l1trg = 0;
   int hlttr = 0;
   int ntrgpas[ntrgp]={0,0,0,0,0,0,0,0,0,0};
   
   //L1 trigger
   Handle<L1GlobalTriggerReadoutRecord> L1GTRR;
   bool isL1Trig=true;
   try{   
     iEvent.getByLabel(l1Label_,L1GTRR);  //gtDigis
   } catch (cms::Exception &iEvent) { isL1Trig=false;}
   
   if (isL1Trig && L1GTRR.isValid()) {
     const unsigned int n(L1GTRR->decisionWord().size());
     const bool accept(L1GTRR->decision());
     if (accept) {
       //       for (unsigned int i=0; i!=n && i<32; ++i) {
       for (unsigned int i=0; i!=n ; ++i) {
	 int il1trg = (L1GTRR->decisionWord()[i]) ? 1 : 0;
	 //	 cout <<"il1trg "<<i<<" "<<il1trg<<" "<<L1GTRR->decisionWord()[i]<<endl;
	 if (il1trg>0 && i<32) l1trg +=int(pow(2., double(i%32))*il1trg);
       }
     }
     //      tmpHOCalib.trig1 = l1trg;  
   }// else { return;}

    //HLT 

   Handle<edm::TriggerResults> trigRes;
   bool isTrig=true;

   try{
     //GMA     iEvent.getByType(trigRes);
     iEvent.getByLabel(hltLabel_, trigRes);; //"TriggerResults" //GMA	iEvent.getByType(trigRes);
   } catch (cms::Exception &iEvent) { isTrig=false;}
   if (isTrig) {
     unsigned int size = trigRes->size();
     edm::TriggerNames triggerNames(*trigRes);
     
     // loop over all paths, get trigger decision
     for(unsigned i = 0; i != size && i<32; ++i) {
       std::string name = triggerNames.triggerName(i);
       fired[name] = trigRes->accept(i);
       int ihlt =  trigRes->accept(i);
       if (ihlt >0 && i < (int)ntrgp) { ntrgpas[i] = 1;}
       
       //       cout <<"name "<<i<<" "<<name<<" "<<fired[name]<<" "<<ihlt<<" "<<size<<endl;
       if (i<32 && ihlt>0) hlttr += int(pow(2., double(i%32))*ihlt);
     }
     //    tmpHOCalib.trig2 = hlttr;
   }

   //   cout <<"muon->size() "<<Nevents<<" "<<(int)isMuon <<" "<<cosmicmuon->size()<<" "<<l1trg<<" "<<hlttr<<endl;
    
   for(reco::TrackCollection::const_iterator ncosm = cosmicmuon->begin();
      ncosm != cosmicmuon->end();  ++ncosm) {
    
    HOCalibVariables tmpHOCalib;

    tmpHOCalib.trig1 = l1trg;
    tmpHOCalib.trig2 = hlttr;    

    int charge = ncosm->charge();  
    
    double innerr = (*ncosm).innerPosition().Perp2();
    double outerr = (*ncosm).outerPosition().Perp2();
    int iiner = (innerr <outerr) ? 1 : 0;

    //---------------------------------------------------
    //             in_to_out  Dir         in_to_out  Dir
    //   StandAlone ^         ^     Cosmic    ^    |
    //              |         |               |    v
    //---------------------------------------------------Y=0
    //   StandAlone |         |     Cosmic    ^    |
    //              v         v               |    v
    //----------------------------------------------------

    double posx, posy, posz;
    double momx, momy, momz;

    if (iiner==1) {
      posx = (*ncosm).innerPosition().X();
      posy = (*ncosm).innerPosition().Y();
      posz = (*ncosm).innerPosition().Z();

      momx = (*ncosm).innerMomentum().X();
      momy = (*ncosm).innerMomentum().Y();
      momz = (*ncosm).innerMomentum().Z();

    } else {
      posx = (*ncosm).outerPosition().X();
      posy = (*ncosm).outerPosition().Y();
      posz = (*ncosm).outerPosition().Z();

      momx = (*ncosm).outerMomentum().X();
      momy = (*ncosm).outerMomentum().Y();
      momz = (*ncosm).outerMomentum().Z();
    }


    //    cout <<"momx "<< posx<<" "<<posy<<" "<<posz<<" "<<momx<<" "<<momy<<" "<<momz<<endl;
    PositionType trkpos(posx, posy, posz);

    Hep3Vector tmpmuon3v(posx, posy, posz);
    Hep3Vector tmpmuondir(momx, momy, momz);
    
    bool samedir = (tmpmuon3v.dot(tmpmuondir) >0) ? true : false;
    for (int i=0; i<3; i++) {tmpHOCalib.caloen[i] = 0.0;}
    int inearbymuon = 0;
    int inearbymuon2 = 0;
    for(reco::TrackCollection::const_iterator ncosmcor = cosmicmuon->begin();
	ncosmcor != cosmicmuon->end();  ++ncosmcor) {
      if (ncosmcor==ncosm) continue;
      
      Hep3Vector tmpmuon3vcor;
      Hep3Vector tmpmom3v;
      if (iiner==1) {
	tmpmuon3vcor = Hep3Vector((*ncosmcor).innerPosition().X(),(*ncosmcor).innerPosition().Y(),(*ncosmcor).innerPosition().Z());
	tmpmom3v = Hep3Vector((*ncosmcor).innerMomentum().X(),(*ncosmcor).innerMomentum().Y(),(*ncosmcor).innerMomentum().Z());
      } else {
	tmpmuon3vcor = Hep3Vector((*ncosmcor).outerPosition().X(),(*ncosmcor).outerPosition().Y(),(*ncosmcor).outerPosition().Z());
	tmpmom3v = Hep3Vector((*ncosmcor).outerMomentum().X(),(*ncosmcor).outerMomentum().Y(),(*ncosmcor).outerMomentum().Z());	
	
      }
      //      if (tmpmom3v.mag()<1.0) continue;
      if (tmpmom3v.mag()<0.2 || (*ncosmcor).ndof()<5) continue;

      double angle = tmpmuon3v.angle(tmpmuon3vcor);
      if (angle < 7.5*pival/180.) {inearbymuon=1;} //  break;}
      if (angle < 35.0*pival/180.) {inearbymuon2=1;}
      if (muonTags_.label() =="cosmicMuons") {
	if (angle <7.5*pival/180.) { tmpHOCalib.caloen[0] +=1.;}
	if (angle <15.0*pival/180.) { tmpHOCalib.caloen[1] +=1.;}
	if (angle <35.0*pival/180.) { tmpHOCalib.caloen[2] +=1.;}
      }
    }
    
    //    cout <<"caloen "<<tmpHOCalib.caloen[0]<<" "<<tmpHOCalib.caloen[1]<<" "<<tmpHOCalib.caloen[2]<<endl;

    localxhor0 = localyhor0 = 20000;  //GM for 22OCT07 data
    
    if (muonTags_.label() =="standAloneMuons") {

      Handle<CaloTowerCollection> calotower;
      bool isCaloTower = true;

      
      try {
	//GMA	iEvent.getByLabel("towerMaker",calotower);
	iEvent.getByLabel(towerLabel_, calotower);
      } catch ( cms::Exception &iEvent ) { isCaloTower = false; } 
      if (isCaloTower) {
	for (CaloTowerCollection::const_iterator calt = calotower->begin();
	     calt !=calotower->end(); calt++) {
	  //2_0_10	  const math::RhoEtaPhiVector towermom = (*calt).momentum();
	  const math::XYZVector towermom = (*calt).momentum();
	  double ith = towermom.theta();
	  double iph = towermom.phi();

	  Hep3Vector calo3v(sin(ith)*cos(iph), sin(ith)*sin(iph), cos(ith));
	    
	  double angle = tmpmuon3v.angle(calo3v);

	  if (angle < 7.5*pival/180.) {tmpHOCalib.caloen[0] += calt->emEnergy()+calt->hadEnergy();}
	  if (angle < 15*pival/180.) {tmpHOCalib.caloen[1] += calt->emEnergy()+calt->hadEnergy();}
	  if (angle < 35*pival/180.) {tmpHOCalib.caloen[2] += calt->emEnergy()+calt->hadEnergy();}
	}
      } 
      
    }
    if (tmpHOCalib.caloen[0] >10.0) continue;
    
    GlobalPoint glbpt(posx, posy, posz);

    double mom = sqrt(momx*momx + momy*momy +momz*momz);

    momx /= mom;
    momy /= mom;
    momz /= mom;
      
    DirectionType trkdir(momx, momy, momz);

    tmpHOCalib.trkdr = (*ncosm).d0();
    tmpHOCalib.trkdz = (*ncosm).dz();

    tmpHOCalib.nmuon = cosmicmuon->size();
    tmpHOCalib.trkvx = glbpt.x();
    tmpHOCalib.trkvy = glbpt.y();
    tmpHOCalib.trkvz = glbpt.z();
    tmpHOCalib.trkmm = mom*charge;
    tmpHOCalib.trkth = trkdir.theta();
    tmpHOCalib.trkph = trkdir.phi();

    tmpHOCalib.nrecht = (inearbymuon ==0) ? (int)(*ncosm).recHitsSize() : -(int)(*ncosm).recHitsSize(); //22OCT07
    tmpHOCalib.ndof  = (inearbymuon2 ==0) ? (int)(*ncosm).ndof() : -(int)(*ncosm).ndof();
    tmpHOCalib.chisq = (*ncosm).normalizedChi2(); // max(1.,tmpHOCalib.ndof);
    tmpHOCalib.therr = 0.;
    tmpHOCalib.pherr = 0.;

    if (iiner==1) {
      reco::TrackBase::CovarianceMatrix innercov = (*ncosm).innerStateCovariance();
      tmpHOCalib.therr = innercov(1,1); //thetaError();
      tmpHOCalib.pherr = innercov(2,2); //phi0Error();
    } else {
      reco::TrackBase::CovarianceMatrix outercov = (*ncosm).outerStateCovariance();
      tmpHOCalib.therr = outercov(1,1); //thetaError();
      tmpHOCalib.pherr = outercov(2,2); //phi0Error();
    }

    ESHandle<MagneticField> theMagField;
    iSetup.get<IdealMagneticFieldRecord>().get(theMagField );     
    GlobalVector magfld = theMagField->inInverseGeV(glbpt); // inTesla(glbpt); // inKGauss(glbpt); inInverseGeV(glbpt);

    //    double magoffield = (m_magscale/4.)*magfld.z(); // mag();
    //    float curve = charge*magoffield/max(0.001,trkpt); //FIXGM p or pt
    // For zero tesla data invert the curvature (a naive approximation)
    //    if (magfld.mag() <0.1 && magfld.x()==0 && magfld.y()==0) curve *=-1.;
    //    HelixArbitraryPlaneCrossing helixx(trkpos, trkdir, curve, (samedir) ? oppositeToMomentum : alongMomentum);
    
    double phiho = trkpos.phi();
    if (phiho<0) phiho +=2*pival;

    int iphisect=int(6*(phiho+pival/18.)/pival); //for u 18/12/06

    double phipos = iphisect*pival/6.;
    double phirot = phipos;
    //from hcalouteralgo.xml "shiftY" value="4.0575*m  "shiftX" value="-36.0674*cm"
    //Layer0 name="shiftY"       value="3.945*m"/>
    
    GlobalVector xLocal(-sin(phirot), cos(phirot), 0.);

    GlobalVector yLocal(0., 0., 1.);
    GlobalVector zLocal = xLocal.cross(yLocal).unit();
    //    GlobalVector zLocal(cos(phirot), sin(phirot), 0.0); 

    FreeTrajectoryState freetrajectorystate_ = getFreeTrajectoryState(*ncosm,&(*theMagField), iiner, samedir);

    Surface::RotationType rot(xLocal, yLocal, zLocal);

    
    //    cout <<"xlocal "<<xLocal<<" "<< yLocal<<" "<< zLocal<<endl;

    int ipath = 0;
    for (int ik=0; ik<2; ik++) { //propagate track in two HO layers

      double radial = 405.757; // sqrt(405.8**2+33.06**2) if rotation too, not
      if (ik==0) radial = 394.5;
      Surface::PositionType pos(radial*cos(phipos), radial*sin(phipos), 0.);
      PlaneBuilder::ReturnType aPlane = PlaneBuilder().plane(pos,rot);

      Surface* aPlane2 = new Plane(pos,rot);

      PositionType hotrkpos(0., 0., 0.);
      DirectionType hotrkdir(0., 0., 0.);
      
      
      //      std::pair<bool,double>  pathlength = helixx.pathLength(*aPlane);
      //      if (pathlength.first) {
      //	if (i==1) ipath = 1;  //Only look for tracks which as hits in layer 1

      //      	hotrkpos = helixx.position(pathlength.second);
      //      	hotrkdir = helixx.direction(pathlength.second);
      //      	LocalVector lclvt0 = (*aPlane).toLocal(GlobalVector(hotrkpos.x(),hotrkpos.y(),hotrkpos.z()));

      //      cout <<"xlocal3 "<<ik<<" "<<xLocal<<" "<< yLocal<<" "<< zLocal<<endl;
      SteppingHelixStateInfo steppingHelixstateinfo_ = stepProp->propagate(freetrajectorystate_, (*aPlane2));
      
      //      cout <<"xlocal4 "<<ik<<" "<<xLocal<<" "<< yLocal<<" "<< zLocal<<endl;
      if (steppingHelixstateinfo_.isValid()) {
	if (ik==1) ipath = 1;  //Only look for tracks which as hits in layer 1

	GlobalVector hotrkpos2(steppingHelixstateinfo_.position().x(), steppingHelixstateinfo_.position().y(), steppingHelixstateinfo_.position().z());
	Hep3Vector hotrkdir2(steppingHelixstateinfo_.momentum().x(), steppingHelixstateinfo_.momentum().y(),steppingHelixstateinfo_.momentum().z());
	
	LocalVector lclvt0 = (*aPlane).toLocal(hotrkpos2);

	switch (ik) 
	  {
	  case 0 : 
	    xhor0 = lclvt0.x();
	    yhor0 = lclvt0.y();
	    break;
	  case 1 :
	    xhor1 = lclvt0.x();
	    yhor1 = lclvt0.y();
	    //	    tmpHOCalib.hoang = zLocal.dot(GlobalVector(hotrkdir.x(),hotrkdir.y(),hotrkdir.z()).unit());
	    tmpHOCalib.hoang = Hep3Vector(zLocal.x(),zLocal.y(),zLocal.z()).dot(hotrkdir2.unit());
	    break;
	  default : break;
	  }
      }
    }

    //    cout <<"ipath "<<ipath<<" "<<xhor0<<" "<<yhor0<<" "<<xhor1<<" "<<yhor1<<endl;
    if (ipath) { //If muon crossed HO laeyrs

      int ietaho = 50;
      int iphiho = -1;

      for (int i=0; i<9; i++) {tmpHOCalib.hosig[i]=-100.0;}
      for (int i=0; i<18; i++) {tmpHOCalib.hocorsig[i]=-100.0;}
      for (int i=0; i<9; i++) {tmpHOCalib.hbhesig[i]=-100.0;}
      tmpHOCalib.hocro = -100;
      
      int isect = 0;
      

      findHOEtaPhi(iphisect, ietaho, iphiho);

      
      //      cout <<"etaho "<<iphisect<<" "<<ietaho<<" "<<iphiho<<" "<<iring<<endl;
      if (ietaho !=0 && iphiho !=0 && abs(iring)<=2) { //Muon passed through a tower
	isect = 100*abs(ietaho+30)+abs(iphiho);
	if (abs(ietaho) >=netabin || iphiho<0) isect *=-1; //Not extrapolated to any tower
	if (abs(ietaho) >=netabin) isect -=1000000;  //not matched with eta
	if (iphiho<0)        isect -=2000000; //not matched with phi
	tmpHOCalib.isect = isect;
	
	tmpHOCalib.hodx = localxhor1;
	tmpHOCalib.hody = localyhor1;      
	
	if (iring==0) {
	  tmpHOCalib.hocorsig[8] = localxhor0;
	  tmpHOCalib.hocorsig[9] = localyhor0;
	}
	
	int etamn=-4;
	int etamx=4;
	if (iring==1) {etamn=5; etamx = 10;}
	if (iring==2) {etamn=11; etamx = 16;}
	if (iring==-1){etamn=-10; etamx = -5;}
	if (iring==-2){etamn=-16; etamx = -11;}
	
	int phimn = 1;
	int phimx = 2;
	if (iring ==0) {
	  phimx =2*int((iphiho+1)/2.);
	  phimn = phimx - 1;
	} else {
	  phimn = 3*int((iphiho+1)/3.) - 1; 
	  phimx = phimn + 2;
	}

	if (phimn <1) phimn += nphimx;
	if (phimx >72) phimx -= nphimx;

	int sigstr = m_startTS; // 5;
	int sigend = m_endTS; // 8;
	
	if (iphiho <=nphimx/2) { //GMA310508
	  sigstr -=1; //GMA300608
	  sigend -=1;
	}

	if (m_hbinfo) {
	  for (int i=0; i<9; i++) {tmpHOCalib.hbhesig[i]=-100.0;}

	  if (m_digiInput) {
	    if (isHBHE && (*hbhe).size() >0) {
	      for (HBHEDigiCollection::const_iterator j=(*hbhe).begin(); j!=(*hbhe).end(); j++){
		const HBHEDataFrame digi = (const HBHEDataFrame)(*j);
		
		HcalDetId id =digi.id();
		int tmpeta= id.ieta();
		int tmpphi= id.iphi();
		m_coder = (*conditions_).getHcalCoder(id);
		calibped = conditions_->getHcalCalibrations(id);

		int deta = tmpeta-ietaho;
		if (tmpeta==-1 && ietaho== 1) deta = -1;
		if (tmpeta== 1 && ietaho==-1) deta =  1;
		int dphi = tmpphi-iphiho;
		if (phimn >phimx) {
		  if (dphi==71) dphi=-1;
		  if (dphi==-71) dphi=1;
		}
		
		int ipass2 = (abs(deta) <=1 && abs(dphi)<=1) ? 1 : 0; //NEED correction in full CMS detector
		
		if (ipass2 ==0 ) continue;
		
		float tmpdata[nchnmx];
		for (int i=0; i<digi.size() && i<nchnmx; i++) {
		  tmpdata[i] = m_coder->charge(*m_shape,digi.sample(i).adc(),digi.sample(i).capid());
		  if (deta==0 && dphi==0) hb_time[i] +=tmpdata[i];
		}
		
		float signal = 0;
		//		for (int i=m_startTS; i<digi.size() && i<=m_endTS; i++) { //GMA310508
		for (int i=1; i<digi.size() && i<=8; i++) {
		  signal += tmpdata[i] - calibped.pedestal(digi.sample(i).capid());; 
		}

		if (ipass2 == 1) {
		  if (3*(deta+1)+dphi+1<9)  tmpHOCalib.hbhesig[3*(deta+1)+dphi+1] = signal;
		}
	      }
	    }
	    
	  } else {
	    
	    edm::Handle<HBHERecHitCollection> hbheht;// iEvent.getByType(hbheht);
	    bool isHBHEht = true;
	    try {
	      //GMA	      iEvent.getByType(hbheht);   
	      iEvent.getByLabel(hbheLabel_,hbheht);
	    } catch ( cms::Exception &iEvent ) { isHBHEht = false; }  
	    
	    if (isHBHEht && (*hbheht).size()>0) {
	      if(!(*hbheht).size()) throw (int)(*hbheht).size();
	      
	      for (HBHERecHitCollection::const_iterator j=(*hbheht).begin(); j!=(*hbheht).end(); j++){
		const HBHERecHit hbhehtrec = (const HBHERecHit)(*j);
		
		HcalDetId id =hbhehtrec.id();
		int tmpeta= id.ieta();
		int tmpphi= id.iphi();

		int deta = tmpeta-ietaho;
		if (tmpeta==-1 && ietaho== 1) deta = -1;
		if (tmpeta== 1 && ietaho==-1) deta =  1;
		int dphi = tmpphi-iphiho;
		if (phimn >phimx) {
		  if (dphi==71) dphi=-1;
		  if (dphi==-71) dphi=1;
		}
		
		int ipass2 = (abs(deta) <=1 && abs(dphi)<=1) ? 1 : 0; //NEED correction in full CMS detector
		if ( ipass2 ==0 ) continue;
		
		float signal = hbhehtrec.energy();
		
		if (3*(deta+1)+dphi+1<9)  tmpHOCalib.hbhesig[3*(deta+1)+dphi+1] = signal;
	      }
	    }
	  
	  } //else m_digilevel

	} //m_hbinfo #endif


	if (m_digiInput) {
	  if (isHO && (*ho).size()>0) {

	    int isFilled[netamx*nphimx]; 
	    for (int j=0; j<netamx*nphimx; j++) {isFilled[j]=0;}

	  for (HODigiCollection::const_iterator j=(*ho).begin(); j!=(*ho).end(); j++){
	  const HODataFrame digi = (const HODataFrame)(*j);
	    
	    HcalDetId id =digi.id();

	    int tmpeta= id.ieta();
	    int tmpphi= id.iphi();
	    m_coder = (*conditions_).getHcalCoder(id);
	    
	    int ipass1 =0;
	    if (tmpeta >=etamn && tmpeta <=etamx) {
	      if (phimn < phimx) {
		ipass1 = (tmpphi >=phimn && tmpphi <=phimx ) ? 1 : 0;
	      } else {
		ipass1 = (tmpphi==71 || tmpphi ==72 || tmpphi==1) ? 1 : 0;
	      }
	    }
	    
	    int deta = tmpeta-ietaho;
	    if (tmpeta==-1 && ietaho== 1) deta = -1;
	    if (tmpeta== 1 && ietaho==-1) deta =  1;

	    int dphi = tmpphi -iphiho;
	    if (phimn>phimx) {
	      if (dphi==71) dphi=-1;
	      if (dphi==-71) dphi=1;
	    }

	    int ipass2 = (abs(deta) <=1 && abs(dphi)<=1) ? 1 : 0;
	    
	    int tmpeta1 = (tmpeta>0) ? tmpeta -1 : -tmpeta +14; 

	    float tmpdata[nchnmx]={0,0,0,0,0,0,0,0,0,0};
	    float sigvall[nsigpk]={0,0,0,0,0,0,0};
	    

	    for (int i=0; i<digi.size() && i<nchnmx; i++) {
	      
	      tmpdata[i] = m_coder->charge(*m_shape,digi.sample(i).adc(),digi.sample(i).capid());
	      if (deta==0 && dphi==0) { 
		ho_time[i] +=tmpdata[i];
		if (m_hotime) { 
		  //calculate signals in 4 time slices, 0-3,.. 6-9
		  if (i>=7-nsigpk) {
		    for (int ncap=0; ncap<nsigpk; ncap++) {
		      if (i-ncap >= nstrbn && i-ncap <= nstrbn+3) { 
			sigvall[ncap] +=tmpdata[i];
		      }
		    }
		  }
		  if (i==digi.size()-1) {
		    float mxled=-1;
		    int imxled = 0;
		    for (int ij=0; ij<nsigpk; ij++) {
		      if (sigvall[ij] > mxled) {mxled = sigvall[ij]; imxled=ij;}
		    }

		    double pedx = 0.0;
		    for (int ij=0; ij<4; ij++) {
		      pedx +=pedestal[tmpeta1][tmpphi-1][ij];
		    }
#ifdef __DO_HISTOGRAMMING__
		    if (mxled-pedx >2 && mxled-pedx <20 ) {
		      hopeak[ntrgp]->Fill(nphimx*tmpeta1 + tmpphi-1, imxled+nstrbn);
		      for (int jk=0; jk<ntrgp; jk++) {
			if (ntrgpas[jk]>0) {
			  hopeak[jk]->Fill(nphimx*tmpeta1 + tmpphi-1, imxled+nstrbn);
			}
		      }
		      if (tmpdata[5]+tmpdata[6] >1) {
			horatio->Fill(nphimx*tmpeta1 + tmpphi-1, (tmpdata[5]-tmpdata[6])/(tmpdata[5]+tmpdata[6]));
		      }
		      for (int ij=0; ij<digi.size() && ij<nchnmx; ij++) {
			
			hotime[ntrgp]->Fill(nphimx*nchnmx*tmpeta1 + nchnmx*(tmpphi-1) + ij, tmpdata[ij]);
			Nhotime[ntrgp]->Fill(nphimx*nchnmx*tmpeta1 + nchnmx*(tmpphi-1) + ij, 1.);
			
			for (int jk=0; jk<ntrgp; jk++) {
			  if (ntrgpas[jk]>0) {
			    hotime[jk]->Fill(nphimx*nchnmx*tmpeta1 + nchnmx*(tmpphi-1) + ij, tmpdata[ij]);
			    Nhotime[jk]->Fill(nphimx*nchnmx*tmpeta1 + nchnmx*(tmpphi-1) + ij, 1.);
			  }
			}
		      }
		    }
#endif
		  }
		}
	      }
	    }
	    
	    if (abs(tmpeta) <=15 && deta==0 && dphi ==0) { 
	      float signal = 0;
	      
	      int icnt = 0;
	      
	      for (int i =0; i<nchnmx && i< digi.size(); i++) {
		if (i >=sigstr && i<=sigend) continue;
		signal += tmpdata[i] - pedestal[tmpeta1][tmpphi-1][digi.sample(i).capid()];
		if (++icnt >=4) break;
	      }
	      
	      tmpHOCalib.hocro = signal;
	    }


	    if (m_hotime) { 
	      if (ipass1 ==0 && ipass2 ==0 && cosmicmuon->size()<=2) {
		if (abs(ietaho) <=netabin && iphiho >0) {
		  if ((iphiho >=1 && iphiho<=nphimx/2 && tmpphi >=1 && tmpphi <=nphimx/2) ||
		      (iphiho >nphimx/2 && iphiho<=nphimx && tmpphi >nphimx/2 && tmpphi <=nphimx)) {
#ifdef __DO_HISTOGRAMMING__
		    if (isFilled[nphimx*tmpeta1+tmpphi-1]==0) {
		      isFilled[nphimx*tmpeta1+tmpphi-1]=1;
		      for (int i=0; i<digi.size() && i<nchnmx; i++) {
			hopedtime->Fill(nphimx*nchnmx*tmpeta1 + nchnmx*(tmpphi-1) + i, tmpdata[i]);
			Nhopedtime->Fill(nphimx*nchnmx*tmpeta1 + nchnmx*(tmpphi-1) + i, 1.); 
			hopedpr->Fill(nphimx*nchnmx*tmpeta1 + nchnmx*(tmpphi-1) + i, tmpdata[i]);
		      }
		    } //isFilled
#endif
		  }
		}
	      }
	    }

	    //	    if (ipass1 ==0 && ipass2 ==0 ) continue;

	    float signal = 0;
	    for (int i=sigstr; i<digi.size() && i<=sigend; i++) {
	      signal += tmpdata[i] - pedestal[tmpeta1][tmpphi-1][digi.sample(i).capid()];
	    }
	    if (signal <-100 || signal >100000) signal = -100;

#ifdef __DO_HISTOGRAMMING__	    
	    if (signal >-100) {
	      for (int i=0; i<5; i++) {

		if (signal >i*m_sigma) {
		  ho_occupency[i]->Fill(nphimx*tmpeta1+tmpphi-1);
		}
	      }
	    }
#endif
	    
	    if (ipass1 ==0 && ipass2 ==0 ) continue;

	    if (ipass1 ==1) {
	      int tmpdph = tmpphi-phimn;
	      if (tmpdph<0) tmpdph = 2;  //only case of iphi==1, where phimn=71
	      
	      int ilog = 2*(tmpeta-etamn)+tmpdph;
	      if (iring !=0) { 
		if (iring >0) {
		  ilog = 3*(tmpeta-etamn)+tmpdph; //Again CMS correction
		} else {
		  ilog = 3*(etamx-tmpeta)+tmpdph; //Again CMS correction
		}
	      }
	      if (ilog>-1 && ilog<18) { 
		tmpHOCalib.hocorsig[ilog] = signal;
	      }
	    }	      
	    
	    if (ipass2 ==1) {
	      if (3*(deta+1)+dphi+1<9) tmpHOCalib.hosig[3*(deta+1)+dphi+1] = signal; //Again CMS azimuthal near phi 1&72
	    }

	    /*	    
		   // Possibility to store pedestal by shifting phi tower by 6
		   // But, due to missing tower at +-5, we do not have always proper
		   // statistics and also in pedestal subtracted data, we do not have
		   // signal in that tower
		   // 
	    if (deta==0 && dphi ==0) {
	      int crphi = tmpphi + 6;
	      if (crphi >72) crphi -=72;
	      
	      for (HODigiCollection::const_iterator jcr=(*ho).begin(); jcr!=(*ho).end(); jcr++){
		const HODataFrame digicr = (const HODataFrame)(*jcr);
		HcalDetId idcr =digicr.id();
		int etacr= idcr.ieta();
		int phicr= idcr.iphi();
		m_coder = (*conditions_).getHcalCoder(idcr);

		if (tmpeta==etacr && crphi ==phicr) {
		  
		  float tmpdatacr[nchnmx];
		  for (int i=0; i<digicr.size() && i<nchnmx; i++) {
		    tmpdatacr[i] = m_coder->charge(*m_shape,digicr.sample(i).adc(),digicr.sample(i).capid());
		  }
		}
	      }
	    }
	    */

	  }
	} 
      } else {
	
	  edm::Handle<HORecHitCollection> hoht; //   iEvent.getByType(hoht);
	bool isHOht = true;
	try {
	  //GMA	  iEvent.getByType(hoht);
	  iEvent.getByLabel(hoLabel_,hoht);
	} catch ( cms::Exception &iEvent ) { isHOht = false; } 

	if (isHOht && (*hoht).size()>0) {
	  for (HORecHitCollection::const_iterator j=(*hoht).begin(); j!=(*hoht).end(); j++){
	    const HORecHit hohtrec = (const HORecHit)(*j);
	    
	    HcalDetId id =hohtrec.id();
	    int tmpeta= id.ieta();
	    int tmpphi= id.iphi();
	    
	    int ipass1 =0;
	    if (tmpeta >=etamn && tmpeta <=etamx) {
	      if (phimn < phimx) {
		ipass1 = (tmpphi >=phimn && tmpphi <=phimx ) ? 1 : 0;
	      } else {
		ipass1 = (tmpphi==71 || tmpphi ==72 || tmpphi==1) ? 1 : 0;
	      }
	    }
	    
	    int deta = tmpeta-ietaho;
	    if (tmpeta==-1 && ietaho== 1) deta = -1;
	    if (tmpeta== 1 && ietaho==-1) deta =  1;
	    
	    int dphi = tmpphi -iphiho;
	    if (phimn>phimx) {
	      if (dphi==71) dphi=-1;
	      if (dphi==-71) dphi=1;
	    }

	    int ipass2 = (abs(deta) <=1 && abs(dphi)<=1) ? 1 : 0;
	    
	    if (ipass1 ==0 && ipass2 ==0 ) continue;

	    float signal = hohtrec.energy();

	    if (ipass1 ==1) {
	      int tmpdph = tmpphi-phimn;
	      if (tmpdph<0) tmpdph = 2;  //only case of iphi==1, where phimn=71
	      
	      int ilog = 2*(tmpeta-etamn)+tmpdph;
	      if (iring !=0) { 
		if (iring >0) {
		  ilog = 3*(tmpeta-etamn)+tmpdph; //Again CMS correction
		} else {
		  ilog = 3*(etamx-tmpeta)+tmpdph; //Again CMS correction
		}
	      }
	      if (ilog>-1 && ilog<18) {
		tmpHOCalib.hocorsig[ilog] = signal;
	      }
	    }	      
	    
	    if (ipass2 ==1) {
	      
	      if (3*(deta+1)+dphi+1<9) {
		tmpHOCalib.hosig[3*(deta+1)+dphi+1] = signal; //Again CMS azimuthal near phi 1&72
	      }
	    }
	    
	    if (deta==0 && dphi ==0) {
	      int crphi = tmpphi + 6;
	      if (crphi >72) crphi -=72;
	      
	      for (HORecHitCollection::const_iterator jcr=(*hoht).begin(); jcr!=(*hoht).end(); jcr++){
		const HORecHit reccr = (const HORecHit)(*jcr);
		HcalDetId idcr =reccr.id();
		int etacr= idcr.ieta();
		int phicr= idcr.iphi();
		if (tmpeta==etacr && crphi ==phicr) {
		  
		  tmpHOCalib.hocro = reccr.energy();

		}
	      }
	    }
	  }
	} 
	}
	
	Npass++;
	//	cout <<"event passed "<<Npass<<endl;
	hostore->push_back(tmpHOCalib);	
	
      } else {cout <<"ietaho "<<ietaho<<" "<<iphiho<<" "<<iring<<" "<<tmpHOCalib.trkvx<<" "<<tmpHOCalib.trkvy<<" "<<tmpHOCalib.trkvz<<" "<<tmpHOCalib.trkmm<<" "<<tmpHOCalib.trkth<<" "<<tmpHOCalib.trkph<<" "<<xhor0<<" "<<yhor0<<" "<<xhor1<<" "<<yhor1<<endl;}
    } else {cout <<"ipath == 0" <<" "<<tmpHOCalib.trkvy<<" "<<tmpHOCalib.trkvz<<" "<<tmpHOCalib.trkmm<<" "<<tmpHOCalib.trkth<<" "<<tmpHOCalib.trkph<<" "<<xhor0<<" "<<yhor0<<" "<<xhor1<<" "<<yhor1<<endl;} //if (ipath) { //If muon crossed HO laeyrs

  } //for(reco::TrackCollection::const_iterator ncosm = cosmicmuon->begin();
  } 
  
  //  cout <<"end of producer "<<hostore->size()<<endl;
  if (hostore->size()>0) iEvent.put(hostore, "HOCalibVariableCollection");

  
}

// ------------ method called once each job just before starting event loop  ------------
void 
AlCaHOCalibProducer::beginJob(const edm::EventSetup& iSetup)
{
  Nevents = 0;
  Npass = 0;
  irunold = -1;
  nRuns = 0;
  edm::ESHandle<MagneticField> bField;
  iSetup.get<IdealMagneticFieldRecord>().get(bField);
  stepProp  = new SteppingHelixPropagator(&*bField,anyDirection);
  stepProp->setMaterialMode(false);
  stepProp->applyRadX0Correction(true);
  
  //  edm::ESHandle<CaloGeometry>PG;
  //  EventSetup.get<IdealGeometryRecord>().get(pG);
  //  geo = pG.product();
  
  //  iSetup.get<HcalDbRecord>().get(conditions_);

  for (int i=0; i<netamx; i++) {
    for (int j=0; j<nphimx; j++) {
      for (int k=0; k<ncidmx; k++) {
	pedestal[i][j][k]=0.0;
      }
    }
  }



}

// ------------ method called once each job just after ending the event loop  ------------
void 
AlCaHOCalibProducer::endJob() {

  cout<<"Processed event # "<<Nevents<<" and selected evt # "<<Npass<<endl;
  if (m_hotime && m_digiInput) {
    cout <<"ho_time ";
    for (int i=0; i<nchnmx; i++) { cout <<ho_time[i]<<" ";} 
    cout<<endl;
    cout <<"hb_time ";
    for (int i=0; i<nchnmx; i++) { cout <<hb_time[i]<<" ";} 
    cout<<endl;
  }

}


void AlCaHOCalibProducer::findHOEtaPhi(int iphisect, int& ietaho, int& iphiho) {
  
  //18/12/06 : use only position, not angle phi

  //  double etalow[netabin]={0.02, 35.42, 70.85, 106.82, 143.33, 182.53, 222, 263.15, 306.29, 351.74, 408.94, 451, 505.56, 563.94, 626.64, 660.25}; //16th(last) entries is arbitray
  //  double etahgh[netabin]={35.37, 70.75, 106.72, 125.68, 182.43, 221.9, 263.05, 306.19, 351.64, 394.29, 450.9, 505.46, 563.84, 626.54, 659.94, 700.25};

  //  23/06/08 : Do not use 13mm shift in HO pannel
  //  double etalow[netabin]={   0.025,  35.195,  70.625, 106.595, 144.165, 183.365, 222.835, 263.985, 307.125, 352.575, 407.425, 449.485, 504.045, 562.425, 625.125, 660.25};
  //  double etahgh[netabin]={  35.145,  70.575, 106.545, 125.505, 183.315, 222.785, 263.935, 307.075, 352.525, 395.175, 449.435, 503.995, 562.375, 625.075, 658.475, 700.25};

double etalow[netabin]={   0.025,  35.195,  70.625, 106.595, 141.565, 180.765, 220.235, 261.385, 304.525, 349.975, 410.025, 452.085, 506.645, 565.025, 627.725, 660.25};
double etahgh[netabin]={  35.145,  70.575, 106.545, 125.505, 180.715, 220.185, 261.335, 304.475, 349.925, 392.575, 452.035, 506.595, 564.975, 627.675, 661.075, 700.25};

  double philow[6]={-76.27, -35.11, 0.35, 35.81, 71.77, 108.93};  //Ring+/-1 & 2
  double phihgh[6]={-35.81, -0.35, 35.11, 71.07, 108.23, 140.49};

  double philow00[6]={-60.27, -32.91, 0.35, 33.61, 67.37, 102.23}; //Ring0 L0
  double phihgh00[6]={-33.61, -0.35, 32.91, 66.67, 101.53, 129.49};

  double philow01[6]={-64.67, -34.91, 0.35, 35.61, 71.37, 108.33}; //Ring0 L1
  double phihgh01[6]={-35.61, -0.35, 34.91, 70.67, 107.63, 138.19};


  iring = -10;

  double tmpdy =  abs(yhor1);
  for (int i=0; i<netabin; i++) {
    if (tmpdy >etalow[i] && tmpdy <etahgh[i]) {
      ietaho = i+1; 
      float tmp1 = fabs(tmpdy-etalow[i]);
      float tmp2 = fabs(tmpdy-etahgh[i]);
 
      localyhor1 = (tmp1 < tmp2) ? -tmp1 : tmp2;

      if (i<4) iring =0;
      if (i>=4 && i<10) iring=1;
      if (i>=10 && i<netabin) iring=2;
      break;
    }
  }

  int tmpphi = 0;
  int tmpphi0 = 0;

  if (ietaho >4) { //Ring 1 and 2
    for (int i=0; i<6; i++) {
      if (xhor1 >philow[i] && xhor1 <phihgh[i]) { 
	tmpphi=i+1; 
	float tmp1 = fabs(xhor1-philow[i]);
	float tmp2 = fabs(xhor1-phihgh[i]);
	localxhor1 = (tmp1 < tmp2) ? -tmp1 : tmp2;
	break;
      }
    }
  } else {  //Ring 0
    for (int i=0; i<6; i++) {
      if (xhor1 >philow01[i] && xhor1 <phihgh01[i]) { 
	tmpphi=i+1; 
	float tmp1 = fabs(xhor1-philow01[i]);
	float tmp2 = fabs(xhor1-phihgh01[i]);
	localxhor1 = (tmp1 < tmp2) ? -tmp1 : tmp2;
	break;
      }
    }

    for (int i=0; i<6; i++) {
      if (xhor0 >philow00[i] && xhor0 <phihgh00[i]) { 
	tmpphi0=i+1; 
	float tmp1 = fabs(xhor0-philow00[i]);
	float tmp2 = fabs(xhor0-phihgh00[i]);
	localxhor0 = (tmp1 < tmp2) ? -tmp1 : tmp2;
	if (tmpphi !=tmpphi0) localxhor0 +=10000.;
	break;
      }
    }

    double tmpdy =  abs(yhor0);
    for (int i=0; i<4; i++) {
      if (tmpdy >etalow[i] && tmpdy <etahgh[i]) {
	float tmp1 = fabs(tmpdy-etalow[i]);
	float tmp2 = fabs(tmpdy-etahgh[i]);
	localyhor0 = (tmp1 < tmp2) ? -tmp1 : tmp2;
	if (i+1 != ietaho)  localyhor0 +=10000.;
	break;
      }
    }
  }

  if (tmpphi!=0) { 
    iphiho = 6*iphisect -2 + tmpphi;
    if (iphiho <=0) iphiho +=nphimx;
    if (iphiho >nphimx) iphiho -=nphimx;
  }

  //  isect2 = 15*iring+iphisect+1;

  if (yhor1 <0) { 
    if (abs(ietaho) >netabin) { //Initialised with 50
      ietaho +=1; 
    } else {
      ietaho *=-1; 
    }
    //    isect2 *=-1; 
    iring *=-1;
  } 
}

FreeTrajectoryState AlCaHOCalibProducer::getFreeTrajectoryState( const reco::Track& tk, const MagneticField* field, int iiner, bool dir)
{

  if (iiner ==0) {
    GlobalPoint gpos( tk.outerX(), tk.outerY(), tk.outerZ());
    GlobalVector gmom( tk.outerPx(), tk.outerPy(), tk.outerPz());
    if (dir) gmom *=-1.;
    GlobalTrajectoryParameters par( gpos, gmom, tk.charge(), field);
    CurvilinearTrajectoryError err( tk.extra()->outerStateCovariance());
    return FreeTrajectoryState( par, err);
  } else {
    GlobalPoint gpos( tk.innerPosition().X(), tk.innerPosition().Y(), tk.innerPosition().Z());
    GlobalVector gmom( tk.innerMomentum().X(), tk.innerMomentum().Y(), tk.innerMomentum().Z());
    if (dir) gmom *=-1.;
    GlobalTrajectoryParameters par( gpos, -gmom, tk.charge(), field);
    CurvilinearTrajectoryError err( tk.extra()->innerStateCovariance());
    return FreeTrajectoryState( par, err);
  }

}

#include "FWCore/Framework/interface/MakerMacros.h"

//define this as a plug-in
DEFINE_FWK_MODULE(AlCaHOCalibProducer);

