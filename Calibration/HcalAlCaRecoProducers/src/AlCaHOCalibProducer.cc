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
// $Id: AlCaHOCalibProducer.cc,v 1.3 2008/03/24 18:19:54 kodolova Exp $
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
//08/07/07 #include "CondTools/Hcal/interface/HcalDbPool.h"
#include "CondFormats/HcalObjects/interface/HcalPedestals.h"
#include "CondFormats/HcalObjects/interface/HcalPedestalWidths.h"


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

#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"
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

  float pedestal[32][72][4][4];
  float pedestalHB[32][72][4][4];

  static const int netabin= 16;  
  static const int nphimx = 72;
  static const int ndepthmx=4;
  static const int ncapmx = 4;
  static const int netamx = 32;

  float ho_time[10];
  float hb_time[10];
  std::string digiLabel;
  
  bool debug;
  std::string theRootFileName;
  std::string thePedestalFile;
  TFile* theFile;
  TH1F* hotime[netamx][nphimx];
  TH1F* hopedtime[netamx][nphimx];
  TH1F* crossg[netamx][nphimx];

  TH2F* ho_occupency[5];  

  bool m_hotime;

  edm::InputTag muonTags_;   // cosmicMuons or standAloneMuons
  bool m_digiInput;            // digi (true) or rechit (false)
  bool m_hbinfo;
  int m_startTS;
  int m_endTS;    
  double m_magscale;
  double m_sigma;

  //  const HcalQIEShape* m_shape;
  //  const HcalQIECoder* m_coder;

  //  edm::ESHandle<HcalDbService> conditions_;

  typedef math::Error<5>::type CovarianceMatrix;  
  int Nevents;
  SteppingHelixPropagator* stepProp;
  FreeTrajectoryState getFreeTrajectoryState( const reco::Track& tk, const MagneticField* field, int itag, bool dir);

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
  thePedestalFile = iConfig.getUntrackedParameter<string>("PedestalFile","pedetal25179_2.txt");
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

  produces<HOCalibVariableCollection>("HOCalibVariableCollection").setBranchAlias("HOCalibVariableCollection");
  //  produces<TrackCollection>("TrackCollection");

   //now do what ever other initialization is needed
  for (int i=0; i<netamx; i++) {
    for (int j=0; j<nphimx; j++) {
      for (int k=0; k<ndepthmx; k++) {
	for (int l=0; l <ncapmx; l++) {
	  pedestal[i][j][k][l] = 4.0; // 3.2; // 4.2;
	  pedestalHB[i][j][k][l] = 4.0; //3.2; // 4.2;
	}
      }
    }
  }

  /*
  int nphi, neta, ndepth;
  ifstream file_db;
  file_db.open(thePedestalFile.c_str());
  while(!(file_db.eof())){
    file_db >>neta>>nphi>>ndepth
	    >>pedestal[(neta<0) ? netabin-1-neta : neta-1][nphi-1][ndepth-1][0]
	    >>pedestal[(neta<0) ? netabin-1-neta : neta-1][nphi-1][ndepth-1][1]
	    >>pedestal[(neta<0) ? netabin-1-neta : neta-1][nphi-1][ndepth-1][2]
	    >>pedestal[(neta<0) ? netabin-1-neta : neta-1][nphi-1][ndepth-1][3];
  }
  */



  for (int i=0; i<10; i++) {ho_time[i] = hb_time[i] = 0.0;}

  char title[200];
  theFile = new TFile(theRootFileName.c_str(), "RECREATE");
  if (m_hotime) {
//    theFile = new TFile(theRootFileName.c_str(), "RECREATE");
    theFile->cd();
    for (int j=0; j<netamx; j++) {
      for (int i=0; i<nphimx; i++) {
	int ieta = (j<15) ? j+1 : 14-j;
	sprintf(title, "hotime_eta%i_phi%i", ieta, i+1);
	hotime[j][i] = new TH1F(title, title, 10, -0.5, 9.5);
	
	sprintf(title, "hopedtime_eta%i_phi%i", ieta, i+1);
	hopedtime[j][i] = new TH1F(title, title, 10, -0.5, 9.5);
	
	if (abs(ieta) <=15) { //novGlobal  && ((i >=16 && i<=21) || (i >=52 && i<=69))) {
	  sprintf(title, "crossg_eta%i_phi%i", ieta, i+1);

	  crossg[j][i] = new TH1F(title, title, 80, -5., 5.);
	}

      }
    }
  }

  for (int i=0; i<5; i++) {
    sprintf(title, "ho_occupency (>%i #sigma)", i); 
    ho_occupency[i] = new TH2F(title, title, netamx, -netamx/2, netamx/2, nphimx, 0.5, nphimx+0.5);
  }
  

}


AlCaHOCalibProducer::~AlCaHOCalibProducer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

  // Write the histos to file

  if (m_hotime) {
    theFile->cd();
    for (int i=0;i<nphimx; i++) {
      for (int j=0; j<netamx; j++) {
	hotime[j][i]->Scale(10./max(1.,hotime[j][i]->GetEntries()));
	hopedtime[j][i]->Scale(10./max(1.,hopedtime[j][i]->GetEntries()));
      }
    }
  }
  theFile->Write();
  theFile->Close();

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
AlCaHOCalibProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   

  using namespace edm;
  Nevents++;
  //  int irun = iEvent.id().run();
  //  int ievt = iEvent.id().event();
  //  cout << endl<<"--- AlCaHOCalibProducer: Event analysed #Run: " << irun << " #Event: " << ievt << endl;
  if (Nevents%500==1) cout <<"AlCaHOCalibProducer Processing event # "<<Nevents<<endl;
  std::auto_ptr<HOCalibVariableCollection> hostore (new HOCalibVariableCollection);


  /*
  edm::Handle<HODigiCollection> ho;     iEvent.getByType(ho);
  try{
    if(!(*ho).size()) throw (int)(*ho).size();
    for (HODigiCollection::const_iterator j=(*ho).begin(); j!=(*ho).end(); j++){
      const HODataFrame digi = (const HODataFrame)(*j);
      HcalDetId id =digi.id();
      int tmpeta= id.ieta();
      int tmpphi= id.iphi();
      int tmpdepth =id.depth();
//      cout <<"HO   "<<tmpeta<<" "<<tmpphi<<" "<<tmpdepth<<" ";
      float tmpdata[10];
      for (int i=0; i<digi.size() && i<10; i++) {
	tmpdata[i] = digi.sample(i).nominal_fC();
//	cout <<" "<<i<<" "<<tmpdata[i];
      }
//      cout<<endl;
    }
  }  catch ( cms::Exception &iEvent ) { } 

  edm::Handle<HBHEDigiCollection> hbhe;     iEvent.getByType(hbhe);
  try{
    if(!(*hbhe).size()) throw (int)(*hbhe).size();
    for (HBHEDigiCollection::const_iterator j=(*hbhe).begin(); j!=(*hbhe).end(); j++){
      const HBHEDataFrame digi = (const HBHEDataFrame)(*j);
      HcalDetId id =digi.id();
      int tmpeta= id.ieta();
      int tmpphi= id.iphi();
      int tmpdepth =id.depth();
//      cout <<"HBHE "<<tmpeta<<" "<<tmpphi<<" "<<tmpdepth<<" ";
      float tmpdata[10];
      for (int i=0; i<digi.size() && i<10; i++) {
	tmpdata[i] = digi.sample(i).nominal_fC();
//	cout <<" "<<i<<" "<<tmpdata[i];
      }
//      cout<<endl;
    }
  }  catch ( cms::Exception &iEvent ) { } 
  */
  
  double pival = acos(-1.);

  Handle<reco::TrackCollection> cosmicmuon;
  bool isMuon = true;
  try {
    //GM09/10/07 for CSA07 sample 
    // /store/CSA07/mc/2007/10/1/CSA07-DrellYan_mumu_40-ALCARECO-A1/0000/0059E054-8770-DC11-8432-000423D99A8E.root 
    iEvent.getByLabel(muonTags_, cosmicmuon);
    //    iEvent.getByLabel("ALCARECOHcalCalZMuMu","StandAlone", cosmicmuon);

  } catch ( cms::Exception &iEvent ) { isMuon = false; } 
  if (isMuon && cosmicmuon->size()>0) { 
  for(reco::TrackCollection::const_iterator ncosm = cosmicmuon->begin();
      ncosm != cosmicmuon->end();  ++ncosm) {
    
    HOCalibVariables tmpHOCalib;

    tmpHOCalib.trig1 = 0;
    tmpHOCalib.trig2 = 0;    
    

    int charge = ncosm->charge();  
    
    double innerr = (*ncosm).innerPosition().Perp2();
    double outerr = (*ncosm).outerPosition().Perp2();
    int iiner = (innerr <outerr) ? 1 : 0;
    //    cout <<"inner outer "<<innerr<<" "<<outerr<<endl;
    //    if (innery * outery <0) {
    //      cout <<"Not in same side "<< innery <<" "<<outery<<endl;
    //      continue;
    //    } 
    //             in_to_out  Dir         in_to_out  Dir
    //   StandAlone ^         ^     Cosmic    ^    |
    //              |         |               |    v
    //---------------------------------------------------Y=0
    //   StandAlone |         |     Cosmic    ^    |
    //              v         v               |    v
    //--------------------------------------

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

    PositionType trkpos(posx, posy, posz);

    Hep3Vector tmpmuon3v(posx, posy, posz);
    Hep3Vector tmpmuondir(momx, momy, momz);
    
    bool samedir = (tmpmuon3v.dot(tmpmuondir) >0) ? true : false;

    int inearbymuon = 0;
    for(reco::TrackCollection::const_iterator ncosmcor = cosmicmuon->begin();
	ncosmcor != cosmicmuon->end();  ++ncosmcor) {
      if (ncosmcor==ncosm) continue;
      Hep3Vector tmpmuon3vcor;
    if (iiner==1) {
	tmpmuon3vcor = Hep3Vector((*ncosmcor).innerPosition().X(),(*ncosmcor).innerPosition().Y(),(*ncosmcor).innerPosition().Z());
      } else {
	tmpmuon3vcor = Hep3Vector((*ncosmcor).outerPosition().X(),(*ncosmcor).outerPosition().Y(),(*ncosmcor).outerPosition().Z());
      }

      if (tmpmuon3v.angle(tmpmuon3vcor) < 35*pival/180.) {inearbymuon=1; break;}
    }

    localxhor0 = localyhor0 = 20000;  //GM for 22OCT07 data
    
    //22OCT07    if (inearbymuon==1) continue;  //Reject muons, where we have another track within 35 degree
    for (int i=0; i<3; i++) {tmpHOCalib.caloen[i] = 0.0;}
    if (muonTags_.label() =="standAloneMuons") {

      Handle<CaloTowerCollection> calotower;
      bool isCaloTower = true;

      try {
	iEvent.getByLabel("towerMaker",calotower);
      } catch ( cms::Exception &iEvent ) { isCaloTower = false; } 
      if (isCaloTower) {
	for (CaloTowerCollection::const_iterator calt = calotower->begin();
	     calt !=calotower->end(); calt++) {
	  const math::RhoEtaPhiVector towermom = (*calt).momentum();
	  double ith = towermom.theta();
	  double iph = towermom.phi();
	  Hep3Vector calo3v(sin(ith)*cos(iph), sin(ith)*sin(iph), cos(ith));
	    
	  double angle = tmpmuon3v.angle(calo3v);

	  if (angle < 15*pival/180.) {tmpHOCalib.caloen[0] += calt->emEnergy()+calt->hadEnergy();}
	  if (angle < 25*pival/180.) {tmpHOCalib.caloen[1] += calt->emEnergy()+calt->hadEnergy();}
	  if (angle < 35*pival/180.) {tmpHOCalib.caloen[2] += calt->emEnergy()+calt->hadEnergy();}
	}
      } else { tmpHOCalib.caloen[0] = tmpHOCalib.caloen[1] = tmpHOCalib.caloen[2] = -1000; }
    }


    
    if (tmpHOCalib.caloen[0] >6.0) continue;
    
    GlobalPoint glbpt(posx, posy, posz);

    //    double trkpt = sqrt(momx*momx + momy*momy);
    double mom = sqrt(momx*momx + momy*momy +momz*momz);

    momx /= mom;
    momy /= mom;
    momz /= mom;
      
    DirectionType trkdir(momx, momy, momz);

    tmpHOCalib.nmuon = cosmicmuon->size();
    tmpHOCalib.trkvx = glbpt.x();
    tmpHOCalib.trkvy = glbpt.y();
    tmpHOCalib.trkvz = glbpt.z();
    tmpHOCalib.trkmm = mom*charge;
    tmpHOCalib.trkth = trkdir.theta();
    tmpHOCalib.trkph = trkdir.phi();

    tmpHOCalib.nrecht = (inearbymuon ==0) ? (*ncosm).recHitsSize() : -(*ncosm).recHitsSize(); //22OCT07
    tmpHOCalib.ndof  = (int)(*ncosm).ndof();
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
    int ipath = 0;
    for (int i=0; i<2; i++) { //propagate track in two HO layers

      double radial = 405.757; // sqrt(405.8**2+33.06**2) if rotation too, not
      if (i==0) radial = 394.5;
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
      
      SteppingHelixStateInfo steppingHelixstateinfo_ = stepProp->propagate(freetrajectorystate_, (*aPlane2));
      if (steppingHelixstateinfo_.isValid()) {
	if (i==1) ipath = 1;  //Only look for tracks which as hits in layer 1

	GlobalVector hotrkpos2(steppingHelixstateinfo_.position().x(), steppingHelixstateinfo_.position().y(), steppingHelixstateinfo_.position().z());
	Hep3Vector hotrkdir2(steppingHelixstateinfo_.momentum().x(), steppingHelixstateinfo_.momentum().y(),steppingHelixstateinfo_.momentum().z());
	
	LocalVector lclvt0 = (*aPlane).toLocal(hotrkpos2);

	switch (i) 
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

    if (ipath) { //If muon crossed HO laeyrs

      int ietaho = 50;
      int iphiho = -1;

      for (int i=0; i<9; i++) {tmpHOCalib.hosig[i]=-100.0;}
      for (int i=0; i<18; i++) {tmpHOCalib.hocorsig[i]=-100.0;}
      for (int i=0; i<9; i++) {tmpHOCalib.hbhesig[i]=-100.0;}
      tmpHOCalib.hocro = -100;
      
      int isect = 0;
      findHOEtaPhi(iphisect, ietaho, iphiho);

      if (ietaho !=0 && iphiho !=0 && abs(iring)<=2) { //Muon passed through a tower
	isect = 100*abs(ietaho+30)+abs(iphiho);
	if (abs(ietaho) >=netabin || iphiho<0) isect *=-1; //Not extrapolated to any tower
	if (abs(ietaho) >=netabin) isect -=1000000;  //not matched with eta
	if (iphiho<0)        isect -=2000000; //not matched with phi
	tmpHOCalib.isect = isect;
	
	tmpHOCalib.hodx = localxhor1;
	tmpHOCalib.hody = localyhor1;      
	
	if (iring==0) {
	  //	  tmpHOCalib.hocorsig[16] = localxhor0;  //22OCT07 put those in 8 & 9
	  //	  tmpHOCalib.hocorsig[17] = localyhor0;
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

	if (m_digiInput) { 
	  //	  //	  edm::ESHandle<HcalDbService> conditions;  //Use database for pedestal etc
	  //	  //	  iSetup.get<HcalDbRecord>().get(conditions);

	  //	  for each id
	  //	  HcalCalibrations calibrations;
	  //	  //	  //	  conditions->makeHcalCalibration(cell, &calibrations);
	  //
	  //	  calibrations.pedestal(capid)
	  //      calibrations.gaqin(capid)
	  //	  // const HcalPedestalWidth* pedw = (*conditions_).getPedestalWidth(id);
	  //	  // const HcalPedestal* pedm = (*conditions_).getPedestal(id);

	}

	int sigstr = m_startTS; // 5;
	int sigend = m_endTS; // 8;

	if (m_hbinfo) {
	  for (int i=0; i<9; i++) {tmpHOCalib.hbhesig[i]=-100.0;}

	  if (m_digiInput) {
	    edm::Handle<HBHEDigiCollection> hbhe; iEvent.getByType(hbhe);
	    
	    try{
	      if(!(*hbhe).size()) throw (int)(*hbhe).size();
	      
	      for (HBHEDigiCollection::const_iterator j=(*hbhe).begin(); j!=(*hbhe).end(); j++){
		const HBHEDataFrame digi = (const HBHEDataFrame)(*j);
		
		HcalDetId id =digi.id();
		int tmpeta= id.ieta();
		int tmpphi= id.iphi();
		int tmpdepth= id.depth();
		//		const HcalPedestal* pedm = (*conditions_).getPedestal(id);

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
		
		float tmpdata[10];
		for (int i=0; i<digi.size() && i<10; i++) {
		  tmpdata[i] = digi.sample(i).nominal_fC();
		  if (deta==0 && dphi==0) hb_time[i] +=tmpdata[i];
		}
		
		float signal = 0;
		for (int i=m_startTS; i<digi.size() && i<=m_endTS; i++) {
		  signal += tmpdata[i]; 
		  if (tmpeta !=0 && abs(tmpeta) <=netabin && tmpphi<=nphimx && tmpphi>0 && tmpdepth>0 && tmpdepth<=ndepthmx) { 
		    //22OCT07		    signal -= 0.8*pedestalHB[(tmpeta<0) ? netabin-1-tmpeta:tmpeta-1][tmpphi-1][tmpdepth-1][digi.sample(i).capid()];
		    signal -= 0.8*pedestal[(tmpeta<0) ? netabin-1-tmpeta:tmpeta-1][tmpphi-1][tmpdepth-1][digi.sample(i).capid()];//pedm->getValue(digi.sample(i).capid());
		  } else {
		    signal =-100;
		  }
		}



		if (ipass2 == 1) {
		  if (3*(deta+1)+dphi+1<9)  tmpHOCalib.hbhesig[3*(deta+1)+dphi+1] = signal;
		}
	      }
	    }
	    
	    catch (int i ) {
	      //    m_logFile << "Event with " << i<<" HBHE Digis passed." << std::endl;
	    } 
	  } else {
	    
	    edm::Handle<HBHERecHitCollection> hbhe; iEvent.getByType(hbhe);
	    
	    try{
	      if(!(*hbhe).size()) throw (int)(*hbhe).size();
	      
	      for (HBHERecHitCollection::const_iterator j=(*hbhe).begin(); j!=(*hbhe).end(); j++){
		const HBHERecHit hbherec = (const HBHERecHit)(*j);
		
		HcalDetId id =hbherec.id();
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
		
		float signal = hbherec.energy();
		
		if (3*(deta+1)+dphi+1<9)  tmpHOCalib.hbhesig[3*(deta+1)+dphi+1] = signal;
	      }
	    }
	  
	    catch (int i ) {
	      //    m_logFile << "Event with " << i<<" HBHE Rechit passed." << std::endl;
	    }  

	  } //else m_digilevel

	} //m_hbinfo #endif


	if (m_digiInput) {
        edm::Handle<HODigiCollection> ho;     iEvent.getByType(ho);

	try{
	  if(!(*ho).size()) throw (int)(*ho).size();
	  
	  for (HODigiCollection::const_iterator j=(*ho).begin(); j!=(*ho).end(); j++){
	  const HODataFrame digi = (const HODataFrame)(*j);
	    
	    HcalDetId id =digi.id();

	    //	    //	    const HcalPedestalWidth* pedw = (*conditions_).getPedestalWidth(id);
	    //	    const HcalPedestal* pedm = (*conditions_).getPedestal(id);

	    int tmpeta= id.ieta();
	    int tmpphi= id.iphi();
	    int tmpdepth =id.depth();

	    //	    if (pedm && pedw) {
	    //	      for (int i=0; i<4; i++) {
	    //		cout <<"tmpeta "<< tmpeta<<" "<<tmpphi<<" "<<tmpdepth<<" "<<i<<" "<<pedw->getWidth(i)<<" "<<pedw->getSigma(i, (i==3) ? 0 : i+1)<<" "<<pedm->getValue(i)<<endl;
	    //	      }
	    //	    } else {
	    //	      cout <<"==========tmpeta "<< tmpeta<<" "<<tmpphi<<" "<<tmpdepth<<endl; 
	    //	    }

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
	    
	    //	    if (ipass1 ==0 && ipass2 ==0 ) continue;

	    //	    //	    m_coder = (*conditions).getHcalCoder(digi.id());
	    
	    int tmpeta1 = (tmpeta>0) ? tmpeta -1 : -tmpeta +14; 

	    float tmpdata[10]={0,0,0,0,0,0,0,0,0,0};
	    for (int i=0; i<digi.size() && i<10; i++) {
	      tmpdata[i] = digi.sample(i).nominal_fC();
	      if (deta==0 && dphi==0) { 
		ho_time[i] +=tmpdata[i];
		if (m_hotime) { hotime[tmpeta1][tmpphi-1]->Fill(i, tmpdata[i]);}
	      }
	    }
	    
	    
	    if (abs(tmpeta) <=15) {  //NovGlobal run
	      float signal = 0;
	      
	      //	      for (int i =0; i<4; i++) {
	      int icnt = 0;

	      for (int i =0; i<10 && i< digi.size(); i++) {
		if (i >=sigstr && i<=sigend) continue;
		signal +=tmpdata[i] -0.8*pedestal[(tmpeta<0)?netabin-1-tmpeta:tmpeta-1][tmpphi-1][tmpdepth-1][digi.sample(i).capid()];// -pedm->getValue(digi.sample(i).capid())
		if (++icnt >=4) break;
	      }

	      if (m_hotime) crossg[tmpeta1][tmpphi-1]->Fill(signal);
	      if (deta==0 && dphi ==0) tmpHOCalib.hocro = signal; //22OCT07
	    }


	    if (m_hotime) { 
	      if (ipass1 ==0 && ipass2 ==0 && cosmicmuon->size()<=2) {
		if (abs(ietaho) <=netabin && iphiho >0) {
		  if ((iphiho >=1 && iphiho<=nphimx/2 && tmpphi >=1 && tmpphi <=nphimx/2) ||
		      (iphiho >nphimx/2 && iphiho<=nphimx && tmpphi >nphimx/2 && tmpphi <=nphimx)) {
		    for (int i=0; i<digi.size() && i<10; i++) {
		      hopedtime[tmpeta1][tmpphi-1]->Fill(i, tmpdata[i]);
		    }
		  }
		}
	      }
	    }

	    //	    if (ipass1 ==0 && ipass2 ==0 ) continue;

	    float signal = 0;
	    for (int i=sigstr; i<digi.size() && i<=sigend; i++) {
	      signal += tmpdata[i];
	      if (tmpeta !=0 && abs(tmpeta) <=netabin && tmpphi<=nphimx && tmpphi>0 && tmpdepth>0 && tmpdepth<=ndepthmx) { 
		signal -= 0.8*pedestal[(tmpeta<0)?netabin-1-tmpeta:tmpeta-1][tmpphi-1][tmpdepth-1][digi.sample(i).capid()]; //22OCT07pedm->getValue(digi.sample(i).capid())
	      } else {
		signal =-100;
	      }
	    }
	    if (signal <-100 || signal >100000) signal = -100;
	    
	    if (signal >-100) {
	      for (int i=0; i<5; i++) {
		if (signal >i*m_sigma) {
		  ho_occupency[i]->Fill(tmpeta, tmpphi);
		}
	      }
	    }
	    
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
	      if (ilog>-1 && ilog<18) { // if (ilog>-1 && ((ilog<16) || (ilog<18 && iring !=0)) ) {  //22OCT07
		tmpHOCalib.hocorsig[ilog] = signal;
	      }
	    }	      
	    
	    if (ipass2 ==1) {
	      
	      if (3*(deta+1)+dphi+1<9) tmpHOCalib.hosig[3*(deta+1)+dphi+1] = signal; //Again CMS azimuthal near phi 1&72
	    }
	    
	    if (deta==0 && dphi ==0) {
	      int crphi = tmpphi + 6;
	      if (crphi >72) crphi -=72;
	      
	      for (HODigiCollection::const_iterator jcr=(*ho).begin(); jcr!=(*ho).end(); jcr++){
		const HODataFrame digicr = (const HODataFrame)(*jcr);
		HcalDetId idcr =digicr.id();
		int etacr= idcr.ieta();
		int phicr= idcr.iphi();
		if (tmpeta==etacr && crphi ==phicr) {
		  
		  float tmpdatacr[10];
		  for (int i=0; i<digicr.size() && i<10; i++) {
		    tmpdatacr[i] = digicr.sample(i).nominal_fC();
		  }
		  /* //22OCT07
		  tmpHOCalib.hocro = 0;
		  for (int i=sigstr; i<digicr.size() && i<=sigend; i++) {
		    tmpHOCalib.hocro += tmpdatacr[i];
		    if (etacr !=0 && abs(etacr) <=netabin && phicr<=nphimx && phicr>0 && tmpdepth>0 && tmpdepth<=ndepthmx) { 
		      tmpHOCalib.hocro -= 0.8*pedestal[(etacr<0)?netabin-1-etacr:etacr-1][phicr-1][tmpdepth-1][digicr.sample(i).capid()]; //22OCT07 pedm->getValue(digicr.sample(i).capid());
		    } else {
		      tmpHOCalib.hocro =-100;
		    }
		  }
		  */
		}
	      }
	    }
	  }
	} 
	catch (int i ) {
	  //    m_logFile << "Event with " << i<<" HO Digis passed." << std::endl;
	}

      } else {
	edm::Handle<HORecHitCollection> ho;    iEvent.getByType(ho);
	
	try{
	  if(!(*ho).size()) throw (int)(*ho).size();
	  
	  for (HORecHitCollection::const_iterator j=(*ho).begin(); j!=(*ho).end(); j++){
	    const HORecHit horec = (const HORecHit)(*j);
	    
	    HcalDetId id =horec.id();
	    int tmpeta= id.ieta();
	    int tmpphi= id.iphi();
	    //	    int tmpdepth =id.depth();
	    
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

	    float signal = horec.energy();

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
	      if (ilog>-1 && ilog<18) { // if (ilog>-1 && ((ilog<16) || (ilog<18 && iring !=0)) ) { //22OCT07  
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
	      
	      for (HORecHitCollection::const_iterator jcr=(*ho).begin(); jcr!=(*ho).end(); jcr++){
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
	catch (int i ) {
	  //    m_logFile << "Event with " << i<<" HO RecHit passed." << std::endl;
	}
	}
	
	hostore->push_back(tmpHOCalib);	
	
      } else {cout <<"ietaho "<<ietaho<<" "<<iphiho<<" "<<iring<<" "<<tmpHOCalib.trkvx<<" "<<tmpHOCalib.trkvy<<" "<<tmpHOCalib.trkvz<<" "<<tmpHOCalib.trkmm<<" "<<tmpHOCalib.trkth<<" "<<tmpHOCalib.trkph<<" "<<xhor0<<" "<<yhor0<<" "<<xhor1<<" "<<yhor1<<endl;}
    } else {cout <<"ipath == 0" <<" "<<tmpHOCalib.trkvy<<" "<<tmpHOCalib.trkvz<<" "<<tmpHOCalib.trkmm<<" "<<tmpHOCalib.trkth<<" "<<tmpHOCalib.trkph<<" "<<xhor0<<" "<<yhor0<<" "<<xhor1<<" "<<yhor1<<endl;} //if (ipath) { //If muon crossed HO laeyrs


    //    hostore->push_back(tmpHOCalib);
    
  } //for(reco::TrackCollection::const_iterator ncosm = cosmicmuon->begin();
  }// else { cout <<" No muon found " <<endl; }
  //  cout <<"hostore size "<<hostore->size()<<" "<<iEvent.id().run()<<endl;
  //  cout <<"AlCaHOCalibProducer event # "<<Nevents<<" Run # "<<iEvent.id().run()<<" Evt # "<<iEvent.id().event()<<" "<<hostore->size()<<endl;


  if (hostore->size()>0) iEvent.put(hostore, "HOCalibVariableCollection");


/* This is an event example
   //Read 'ExampleData' from the Event
   Handle<ExampleData> pIn;
   iEvent.getByLabel("example",pIn);

   //Use the ExampleData to create an ExampleData2 which 
   // is put into the Event
   std::auto_ptr<ExampleData2> pOut(new ExampleData2(*pIn));
   iEvent.put(pOut);
*/

/* this is an EventSetup example
   //Read SetupData from the SetupRecord in the EventSetup
   ESHandle<SetupData> pSetup;
   iSetup.get<SetupRecord>().get(pSetup);
*/
}

// ------------ method called once each job just before starting event loop  ------------
void 
AlCaHOCalibProducer::beginJob(const edm::EventSetup& iSetup)
{
  Nevents = 0;

  //  iSetup.get<HcalDbRecord>().get(conditions_);
    edm::ESHandle<MagneticField> bField;
    iSetup.get<IdealMagneticFieldRecord>().get(bField);
    stepProp  = new SteppingHelixPropagator(&*bField,anyDirection);
    stepProp->setMaterialMode(false);
    stepProp->applyRadX0Correction(true);

  //  edm::ESHandle<CaloGeometry>PG;
  //  EventSetup.get<IdealGeometryRecord>().get(pG);
  //  geo = pG.product();
}

// ------------ method called once each job just after ending the event loop  ------------
void AlCaHOCalibProducer::endJob() {

  cout <<"ho_time ";
  for (int i=0; i<10; i++) { cout <<ho_time[i]<<" ";} 
  cout<<endl;
  cout <<"hb_time ";
  for (int i=0; i<10; i++) { cout <<hb_time[i]<<" ";} 
  cout<<endl;
}


void AlCaHOCalibProducer::findHOEtaPhi(int iphisect, int& ietaho, int& iphiho) {
  
  //18/12/06 : use only position, not angle phi

  double etalow[netabin]={0.02, 35.42, 70.85, 106.82, 143.33, 182.53, 222, 263.15, 306.29, 351.74, 408.94, 451, 505.56, 563.94, 626.64, 660.25}; //16th(last) entries is arbitray
  
  double etahgh[netabin]={35.37, 70.75, 106.72, 125.68, 182.43, 221.9, 263.05, 306.19, 351.64, 394.29, 450.9, 505.46, 563.84, 626.54, 659.94, 700.25};

  double philow[6]={-76.27, -35.11, 0.35, 35.81, 71.77, 108.93};  //Ring+/-1 & 2
  double phihgh[6]={-35.81, -0.35, 35.11, 71.07, 108.23, 140.49};

  double philow00[6]={-60.27, -32.91, 0.35, 33.61, 67.37, 102.23,}; //Ring0 L0
  double phihgh00[6]={-33.61, -0.35, 32.91, 66.67, 101.53, 129.49,};

  double philow01[6]={-64.67, -34.91, 0.35, 35.61, 71.37, 108.33,}; //Ring0 L1
  double phihgh01[6]={-35.61, -0.35, 34.91, 70.67, 107.63, 138.19,};


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
