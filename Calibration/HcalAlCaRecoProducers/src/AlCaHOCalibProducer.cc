// -*- C++ -*-
//
// Package:    AlCaHOCalibProducer
// Class:      AlCaHOCalibProducer
// 
/**\class AlCaHOCalibProducer AlCaHOCalibProducer.cc Calibration/AlCaHOCalibProducer/src/AlCaHOCalibProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
Missing towers : eta=5, phi=18-19
               : eta = -5, phi =11-14

*/
//
// Original Author:  Gobinda Majumder
//         Created:  Fri Jul  6 17:17:21 CEST 2007
// $Id$
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

#include "TrackingTools/GeomPropagators/interface/HelixArbitraryPlaneCrossing.h"
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


  std::string digiLabel;
  
  bool debug;
    std::string thePedestalFile;
    edm::InputTag muonTags_;   // cosmicMuons or standAloneMuons
  bool m_digiInput;            // digi (true) or rechit (false)
  bool m_hbinfo;
  int m_startTS;
  int m_endTS;    
  double m_magscale;
  
  //  const HcalQIEShape* m_shape;
  //  const HcalQIECoder* m_coder;
  
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

  thePedestalFile = iConfig.getUntrackedParameter<string>("PedestalFile","pedetal.txt");
  m_digiInput = iConfig.getUntrackedParameter<bool>("digiInput", true);
  m_hbinfo = iConfig.getUntrackedParameter<bool>("hbinfo", false);
  m_startTS = iConfig.getUntrackedParameter<int>("firstTS", 4);

  if(m_startTS<0) m_startTS=0;
  m_endTS = iConfig.getUntrackedParameter<int>("lastTS", 7);
  if (m_endTS < m_startTS) m_endTS = m_startTS + 3;
  if (m_endTS >9) m_endTS=9;
  m_magscale = iConfig.getUntrackedParameter<double>("m_scale", 4.0);

  produces<HOCalibVariableCollection>("HOCalibVariableCollection").setBranchAlias("HOCalibVariableCollection");
  //  produces<TrackCollection>("TrackCollection");

   //now do what ever other initialization is needed
  for (int i=0; i<netamx; i++) {
    for (int j=0; j<nphimx; j++) {
      for (int k=0; k<ndepthmx; k++) {
	for (int l=0; l <ncapmx; l++) {
	  pedestal[i][j][k][l] = 3.2; // 4.2;
	  pedestalHB[i][j][k][l] = 3.2; // 4.2;
	}
      }
    }
  }


}


AlCaHOCalibProducer::~AlCaHOCalibProducer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

  // Write the histos to file

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
AlCaHOCalibProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   

   using namespace edm;

   //  int irun = iEvent.id().run();
   //  int ievt = iEvent.id().event();
  //  cout << endl<<"--- AlCaHOCalibProducer: Event analysed #Run: " << irun << " #Event: " << ievt << endl;

  std::auto_ptr<HOCalibVariableCollection> hostore (new HOCalibVariableCollection);
  //   std::auto_ptr<TrackCollection> hostore (new TrackCollection);
  double pival = acos(-1.);

  //  edm::Handle<TrajectorySeedCollection> seeds; 
  //  iEvent.getByLabel("CosmicMuonSeed",seeds);

  Handle<reco::TrackCollection> cosmicmuon;


  iEvent.getByLabel(muonTags_, cosmicmuon);
  cout <<"cosmicmuon ? "<<muonTags_<<endl;
  //  iEvent.getByLabel("cosmicMuons", cosmicmuon); //FIXGM replace "CosmicMuon" by input through configuration file

  //  iEvent.getByLabel("standAloneMuons", cosmicmuon);
  //  iEvent.getByLabel("standAloneMuons", "UpdatedAtVtx", cosmicmuon); 
  // Handle<reco::TrackCollection> standalonemuon;
  // iEvent.getByLabel("standAloneMuons",standalonemuon);
  // cout <<"stdmuon "<< standalonemuon->size()<<endl; 

  for(reco::TrackCollection::const_iterator ncosm = cosmicmuon->begin();
      ncosm != cosmicmuon->end();  ++ncosm) {
    
    HOCalibVariables tmpHOCalib;
    int charge = ncosm->charge();  
    
    double innerr = (*ncosm).innerPosition().Perp2();
    double outerr = (*ncosm).outerPosition().Perp2();

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

    if (innerr < outerr) {
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
    
    bool samedir = (tmpmuon3v.dot(tmpmuondir)) ? true : false;

    int inearbymuon = 0;
    for(reco::TrackCollection::const_iterator ncosmcor = cosmicmuon->begin();
	ncosmcor != cosmicmuon->end();  ++ncosmcor) {
      if (ncosmcor==ncosm) continue;
      Hep3Vector tmpmuon3vcor;
    if (innerr < outerr) {
	tmpmuon3vcor = Hep3Vector((*ncosmcor).innerPosition().X(),(*ncosmcor).innerPosition().Y(),(*ncosmcor).innerPosition().Z());
      } else {
	tmpmuon3vcor = Hep3Vector((*ncosmcor).outerPosition().X(),(*ncosmcor).outerPosition().Y(),(*ncosmcor).outerPosition().Z());
      }

      if (tmpmuon3v.angle(tmpmuon3vcor) < 35*pival/180.) {inearbymuon=1; break;}
    }
    if (inearbymuon==1) continue;  //Reject muons, where we have another track within 35 degree
    
    if (muonTags_.label() =="standAloneMuons") {
      Handle<CaloTowerCollection> calotower;
      bool isCaloTower = true;

      for (int i=0; i<3; i++) {tmpHOCalib.caloen[i] = 0.0;}
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
	  //          cout <<"angle "<< angle*180/pival<<" "<< ith<<" "<<iph<<endl;

	  if (angle < 15*pival/180.) {tmpHOCalib.caloen[0] += calt->emEnergy()+calt->hadEnergy();}
	  if (angle < 25*pival/180.) {tmpHOCalib.caloen[1] += calt->emEnergy()+calt->hadEnergy();}
	  if (angle < 35*pival/180.) {tmpHOCalib.caloen[2] += calt->emEnergy()+calt->hadEnergy();}
	}
      } else { tmpHOCalib.caloen[0] = tmpHOCalib.caloen[1] = tmpHOCalib.caloen[2] = -1000; }
    }

    if (tmpHOCalib.caloen[0] >6.0) continue;
    
    GlobalPoint glbpt(posx, posy, posz);

    double trkpt = sqrt(momx*momx + momy*momy);
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

    tmpHOCalib.nrecht = (*ncosm).recHitsSize();
    tmpHOCalib.ndof  = (int)(*ncosm).ndof();
    tmpHOCalib.chisq = (*ncosm).normalizedChi2(); // max(1.,tmpHOCalib.ndof);

    //    CovarianceMatrix innercov = (*ncosm).innerStateCovariance(); //FIXGM
    //    tmpHOCalib.therr = innercov(13); //thetaError();  FIXGM check this and put properly
    //    tmpHOCalib.pherr = innercov(15); //phi0Error();


    ESHandle<MagneticField> magfield;
    iSetup.get<IdealMagneticFieldRecord>().get(magfield );     
    GlobalVector magfld = magfield->inInverseGeV(glbpt); // inTesla(glbpt); // inKGauss(glbpt); inInverseGeV(glbpt);
    
    double magoffield = (m_magscale/4.)*magfld.z(); // mag();
    float curve = charge*magoffield/max(0.001,trkpt); //FIXGM p or pt

    //For zero tesla data invert the curvature (a naive approximation)
    if (magfld.mag() <0.1 && magfld.x()==0 && magfld.y()==0) curve *=-1.;
    HelixArbitraryPlaneCrossing helixx(trkpos, trkdir, curve, (samedir) ? oppositeToMomentum : alongMomentum);
    
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

    Surface::RotationType rot(xLocal, yLocal, zLocal);
    int ipath = 0;
    for (int i=0; i<2; i++) { //propagate track in two HO layers
      double radial = 405.757; // sqrt(405.8**2+33.06**2) if rotation too, not
      if (i==0) radial = 394.5;
      Surface::PositionType pos(radial*cos(phipos), radial*sin(phipos), 0.);
      PlaneBuilder::ReturnType aPlane = PlaneBuilder().plane(pos,rot);
      
      PositionType hotrkpos(0., 0., 0.);
      DirectionType hotrkdir(0., 0., 0.);
      
      std::pair<bool,double>  pathlength = helixx.pathLength(*aPlane);
      if (pathlength.first) {
	if (i==1) ipath = 1;  //Only look for tracks which as hits in layer 1
	hotrkpos = helixx.position(pathlength.second);
	hotrkdir = helixx.direction(pathlength.second);
	
	LocalVector lclvt0 = (*aPlane).toLocal(GlobalVector(hotrkpos.x(),hotrkpos.y(),hotrkpos.z()));

	LocalVector glbvt0 = (*aPlane).toLocal(GlobalVector(trkpos.x(),trkpos.y(),trkpos.z()));
	
	switch (i) 
	  {
	  case 0 : 
	    xhor0 = lclvt0.x();
	    yhor0 = lclvt0.y();
	    break;
	  case 1 :
	    xhor1 = lclvt0.x();
	    yhor1 = lclvt0.y();

	    tmpHOCalib.hoang = zLocal.dot(GlobalVector(hotrkdir.x(),hotrkdir.y(),hotrkdir.z()));
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
      tmpHOCalib.hocro = -100;
      
      int isect = 0;
      findHOEtaPhi(iphisect, ietaho, iphiho);
      if (ietaho !=0 && iphiho !=0) { //Muon passed through a tower
	isect = 100*abs(ietaho+30)+abs(iphiho);
	if (abs(ietaho) >=netabin || iphiho<0) isect *=-1; //Not extrapolated to any tower
	if (abs(ietaho) >=netabin) isect -=1000000;  //not matched with eta
	if (iphiho<0)        isect -=2000000; //not matched with phi
	tmpHOCalib.isect = isect;
	
	tmpHOCalib.hodx = localxhor1;
	tmpHOCalib.hody = localyhor1;      
	
	if (iring==0) {
	  tmpHOCalib.hocorsig[16] = localxhor0;
	  tmpHOCalib.hocorsig[17] = localyhor0;
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
	  //	  edm::ESHandle<HcalDbService> conditions;  //Use database for pedestal etc
	  //	  iSetup.get<HcalDbRecord>().get(conditions);
	}

	int sigstr = m_startTS; // 5; //FIXGM use input through .cfg
	int sigend = m_endTS; // 8; //FIXGM use input through .cfg 

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
		}
		
		float signal = 0;
		for (int i=m_startTS; i<digi.size() && i<=m_endTS; i++) {
		  signal += tmpdata[i]; 
		  if (tmpeta !=0 && abs(tmpeta) <=netabin && tmpphi<=nphimx && tmpphi>0 && tmpdepth>0 && tmpdepth<=ndepthmx) { 
		    signal -= pedestalHB[(tmpeta<0) ? netabin-1-tmpeta:tmpeta-1][tmpphi-1][tmpdepth-1][digi.sample(i).capid()];
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
	  //	  cout <<"digi "<<endl;
        edm::Handle<HODigiCollection> ho;     iEvent.getByType(ho);

	try{
	  if(!(*ho).size()) throw (int)(*ho).size();
	  
	  for (HODigiCollection::const_iterator j=(*ho).begin(); j!=(*ho).end(); j++){
	  const HODataFrame digi = (const HODataFrame)(*j);
	    
	    HcalDetId id =digi.id();
	    int tmpeta= id.ieta();
	    int tmpphi= id.iphi();
	    int tmpdepth =id.depth();
	    
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

	    //	    m_coder = (*conditions).getHcalCoder(digi.id());
	    
	    //	    int tmpeta1 = (tmpeta>0) ? tmpeta -1 : -tmpeta +14; 

	    float tmpdata[10]={0,0,0,0,0,0,0,0,0,0};
	    for (int i=0; i<digi.size() && i<10; i++) {
	      tmpdata[i] = digi.sample(i).nominal_fC();
	    }


	    if (ipass1 ==0 && ipass2 ==0 ) continue;

	    float signal = 0;
	    for (int i=sigstr; i<digi.size() && i<=sigend; i++) {
	      signal += tmpdata[i];
	      if (tmpeta !=0 && abs(tmpeta) <=netabin && tmpphi<=nphimx && tmpphi>0 && tmpdepth>0 && tmpdepth<=ndepthmx) { 
		signal -= pedestal[(tmpeta<0)?netabin-1-tmpeta:tmpeta-1][tmpphi-1][tmpdepth-1][digi.sample(i).capid()];
	      } else {
		signal =-100;
	      }
	    }

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
		  tmpHOCalib.hocro = 0;
		  for (int i=sigstr; i<digicr.size() && i<=sigend; i++) {
		    tmpHOCalib.hocro += tmpdatacr[i];
		    if (etacr !=0 && abs(etacr) <=netabin && phicr<=nphimx && phicr>0 && tmpdepth>0 && tmpdepth<=ndepthmx) { 
		      tmpHOCalib.hocro -= pedestal[(etacr<0)?netabin-1-etacr:etacr-1][phicr-1][tmpdepth-1][digicr.sample(i).capid()];
		    } else {
		      tmpHOCalib.hocro =-100;
		    }
		  }
		}
	      }
	    }
	  }
	} 
	catch (int i ) {
	  //    m_logFile << "Event with " << i<<" HO Digis passed." << std::endl;
	}

      } else {
	  //	  cout <<"rechit "<<endl;
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
	      if (ilog>-1 && ilog<18) { 
		tmpHOCalib.hocorsig[ilog] = signal;
	      }
	    }	      
	    
	    if (ipass2 ==1) {
	      
	      if (3*(deta+1)+dphi+1<9) tmpHOCalib.hosig[3*(deta+1)+dphi+1] = signal; //Again CMS azimuthal near phi 1&72
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
      }
    }

    hostore->push_back(tmpHOCalib);
    
  }
  
  //  cout <<"hostore size "<<hostore->size()<<endl;
  iEvent.put(hostore, "HOCalibVariableCollection");


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
AlCaHOCalibProducer::beginJob(const edm::EventSetup&)
{
  //  edm::ESHandle<CaloGeometry>PG;
  //  EventSetup.get<IdealGeometryRecord>().get(pG);
  //  geo = pG.product();
}

// ------------ method called once each job just after ending the event loop  ------------
void 
AlCaHOCalibProducer::endJob() {
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


  int iring = -10;
  //  cout <<"local ho "<<xhor1<<" "<<yhor1<<" "<<xhor0<<" "<<yhor0<<endl;
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


#include "FWCore/Framework/interface/MakerMacros.h"

//define this as a plug-in
DEFINE_FWK_MODULE(AlCaHOCalibProducer);


