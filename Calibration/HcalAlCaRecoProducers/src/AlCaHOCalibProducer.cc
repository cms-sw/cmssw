// -*- C++ -*-
// Dec 2015 : Added bool m_cosmic to choose cosmic or collision run through python file

//integrate the code with some 8_0_X or 7_6_X recent IB and run the 
// following tests: 4.22, 8.0, 25.0, 140.53. You can always activate them using
//runTheMatrix.py -l 4.22

// 7th Nov 2015 :  tmpHOCalib.ecal03 = iso05.sumPt; // iso03.emEt+muonenr.em;
//                   tmpHOCalib.inslumi=lumiScale->begin()->pileup();
//
// April 2015 : Remove all digi part
//  Also look for HO geometry in CMSSW in parallel with stanalone one.
// Official one has problem in reco geometry, particularly tiles at the edge of wheel
// Remove all histogrammes except occupancy one 
// Remove Trigger bits
// But addition of these variables, ilumi (analyser), inslumi (analyser), nprim

// Feb09 2009
// Move the initialisation of SteppingHelixPropagator from ::beginJob() to ::produce()
//
// Oct3 2008
// Difference in tag V00-02-45 with previous code

// 1. One new object on data format, which was realised in  
//     CRUZET data analysis.
//2.  Remove all histogram and cout in the code
//3. An upgrade in code, which increases the acceptance of 
//    muon near the edge (this also realised in CRUZET data).
// Difference in wrt V00-02-45
// 1. initialisation tmpHOCalib.htime = -1000;
// 2. By mistake HLT was commented out

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
#include "DataFormats/FWLite/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "DataFormats/GeometrySurface/interface/PlaneBuilder.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"

#include "DataFormats/HcalCalibObjects/interface/HOCalibVariables.h"
#include "DataFormats/Math/interface/Error.h"
#include "CLHEP/Vector/LorentzVector.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"

#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/Math/interface/Error.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/Luminosity/interface/LumiDetails.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/Scalers/interface/LumiScalers.h"

// Necessary includes for identify severity of flagged problems in HO rechits
//#include "RecoLocalCalo/HcalRecAlgos/interface/HcalCaloFlagLabels.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSeverityLevelComputer.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSeverityLevelComputerRcd.h"
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"
#include "CondFormats/DataRecord/interface/HcalChannelQualityRcd.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

#include "TH2F.h"

/* C++ Headers */
#include <string>

#include <iostream>
#include <fstream>
//
// class decleration
//


class AlCaHOCalibProducer : public edm::EDProducer {
public:
  explicit AlCaHOCalibProducer(const edm::ParameterSet&);
  ~AlCaHOCalibProducer() override;

  typedef Basic3DVector<float>   PositionType;
  typedef Basic3DVector<float>   DirectionType;
  typedef Basic3DVector<float>   RotationType;


private:
  void produce(edm::Event&, const edm::EventSetup&) override;
  void beginJob() override ;
  void endJob() override ;
  void beginRun(edm::Run const &, edm::EventSetup const &) override;
  void fillHOStore(const reco::TrackRef& ncosm,
		   HOCalibVariables& tmpHOCalib,
		   std::unique_ptr<HOCalibVariableCollection> &hostore,
		   int Noccu_old, int indx,
		   edm::Handle<reco::TrackCollection> cosmicmuon,
		   edm::View<reco::Muon>::const_iterator muon1,
		   const edm::Event& iEvent, const edm::EventSetup& iSetup);
  void findHOEtaPhi(int iphsect, int& ietaho, int& iphiho);
  //  virtual void endRun(edm::Run const &, edm::EventSetup const &) override;
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

  TH2F* ho_occupency[5];  
  bool m_occupancy;
  bool m_cosmic;

  const int netabin= 16;  
  const int nphimx = 72;
  const int netamx = 32;
  const int ncidmx = 5;
  const double rHOL0 = 382.0;
  const double rHOL1 = 407.0;

  edm::InputTag muonTags_;   // cosmicMuons (for cosmic run) or muons (for collision run)

  edm::EDGetTokenT<reco::TrackCollection> tok_muonsCosmic_;
  edm::EDGetTokenT<edm::View<reco::Muon> > tok_muons_;
  edm::EDGetTokenT<reco::VertexCollection> tok_vertex_;
  //  edm::EDGetTokenT<LumiDetails> tok_lumi_;
  edm::EDGetTokenT<LumiScalersCollection> tok_lumi_;

  edm::EDGetTokenT<HBHERecHitCollection> tok_hbhe_;
  edm::EDGetTokenT<HORecHitCollection> tok_ho_;
  edm::EDGetTokenT<CaloTowerCollection> tok_tower_;

  bool m_hbinfo;
  int m_startTS;
  int m_endTS;    
  double m_sigma;

  typedef math::Error<5>::type CovarianceMatrix;  
  int Noccu;
  int nRuns;

  //  SteppingHelixPropagator* stepProp;
  FreeTrajectoryState getFreeTrajectoryState( const reco::Track& tk, const MagneticField* field, int itag, bool dir);

  unsigned int Ntp; // # of HLT trigger paths (should be the same for all events!)
  std::map<std::string, bool> fired; 

  //hcal severity ES
  const HcalChannelQuality* theHcalChStatus;
  //  edm::ESHandle<HcalChannelQuality> theHcalChStatus;
  edm::ESHandle<HcalSeverityLevelComputer> hcalSevLvlComputerHndl;
  int Nevents;
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
AlCaHOCalibProducer::AlCaHOCalibProducer(const edm::ParameterSet& iConfig) {
  //register your products

  m_hbinfo = iConfig.getUntrackedParameter<bool>("hbinfo", false);
  m_sigma = iConfig.getUntrackedParameter<double>("sigma", 0.05);
  m_occupancy = iConfig.getUntrackedParameter<bool>("plotOccupancy", false);
  m_cosmic    =  iConfig.getUntrackedParameter<bool>("CosmicData", false);

  // keep InputTag muonTags_ since it is used below. - cowden
  muonTags_ =   iConfig.getUntrackedParameter<edm::InputTag>("muons");
  tok_muonsCosmic_ = consumes<reco::TrackCollection>(muonTags_);
  tok_muons_  = consumes<edm::View<reco::Muon> >(muonTags_);
  tok_vertex_ = consumes<reco::VertexCollection >(iConfig.getParameter<edm::InputTag>("vertexTags"));
  //  tok_lumi_ = consumes<LumiDetails ,edm::InLumi>(iConfig.getParameter<edm::InputTag>("lumiTags"));
  tok_lumi_ = consumes<LumiScalersCollection>(iConfig.getParameter<edm::InputTag>("lumiTags"));
  tok_ho_ = consumes<HORecHitCollection>(iConfig.getParameter<edm::InputTag>("hoInput"));
  tok_hbhe_ = consumes<HBHERecHitCollection>(iConfig.getParameter<edm::InputTag>("hbheInput"));
  tok_tower_ = consumes<CaloTowerCollection>(iConfig.getParameter<edm::InputTag>("towerInput"));  

  produces<HOCalibVariableCollection>("HOCalibVariableCollection").setBranchAlias("HOCalibVariableCollection");
  
  if (m_occupancy) {
    edm::Service<TFileService> fs;
    
    char title[200];

    for (int ij=0; ij<5; ij++) {
      sprintf(title, "ho_occupency (>%i #sigma)", ij+2); 
      ho_occupency[ij] = fs->make<TH2F>(title, title, netamx+1, -netamx-0.5, netamx/2+0.5, nphimx, 0.5, nphimx+0.5); 
    }
  }  
}


AlCaHOCalibProducer::~AlCaHOCalibProducer()
{
 
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
AlCaHOCalibProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  int irun = iEvent.id().run();
  //  int ilumi = iEvent.luminosityBlock();

  Nevents++;

  if (Nevents%5000==1)  edm::LogInfo("HOCalib") <<"AlCaHOCalibProducer Processing event # "<<Nevents<<" "<<Noccu<<" "<<irun<<" "<<iEvent.id().event();

  auto hostore = std::make_unique<HOCalibVariableCollection>();

  edm::Handle<reco::TrackCollection> cosmicmuon;
  edm::Handle<edm::View<reco::Muon> > collisionmuon;

  bool muonOK(true);
  HOCalibVariables tmpHOCalib;
  tmpHOCalib.nprim = -1;
  tmpHOCalib.inslumi=-1.;

  if (m_cosmic) {
    iEvent.getByToken(tok_muonsCosmic_, cosmicmuon);
    muonOK = (cosmicmuon.isValid() && !cosmicmuon->empty());
  } else {
    iEvent.getByToken(tok_muons_,collisionmuon);
    muonOK = (collisionmuon.isValid() && !collisionmuon->empty());

    if (iEvent.isRealData()) {
      edm::Handle<reco::VertexCollection> primaryVertices;
      iEvent.getByToken(tok_vertex_, primaryVertices);
      if (primaryVertices.isValid()) { tmpHOCalib.nprim = primaryVertices->size();}

      tmpHOCalib.inslumi=0.;
	
      edm::Handle<LumiScalersCollection> lumiScale;
      iEvent.getByToken(tok_lumi_, lumiScale);

      if (lumiScale.isValid()) {
        if ( lumiScale->empty() ) {
          edm::LogError("HOCalib") << "lumiScale collection is empty";
        } else {
          tmpHOCalib.inslumi=lumiScale->begin()->pileup();
        }
      }
    }
  }

  if (muonOK) { 
    
    int Noccu_old = Noccu;
    edm::View<reco::Muon>::const_iterator muon1;
    if (m_cosmic) {
      int indx(0);
      for(reco::TrackCollection::const_iterator ncosm = cosmicmuon->begin();
	  ncosm != cosmicmuon->end();  ++ncosm,++indx) {
	if ((*ncosm).ndof() < 15) continue;
	if ((*ncosm).normalizedChi2() >30.0) continue;
	reco::TrackRef tRef = reco::TrackRef(cosmicmuon,indx);
	fillHOStore(tRef,tmpHOCalib,hostore,Noccu_old,indx,cosmicmuon,muon1,
		    iEvent, iSetup);
      }
    } else {
      for( muon1 = collisionmuon->begin(); muon1 < collisionmuon->end(); muon1++ ) {
	if ((!muon1->isGlobalMuon()) || (!muon1->isTrackerMuon())) continue;
	reco::TrackRef ncosm =  muon1->innerTrack();
	fillHOStore(ncosm,tmpHOCalib,hostore,Noccu_old,0,cosmicmuon,muon1,
		    iEvent, iSetup);
      }
    }
  }

  iEvent.put(std::move(hostore), "HOCalibVariableCollection");
  
}

// ------------ method called once each job just before starting event loop  ------------
void 
AlCaHOCalibProducer::beginJob()
{
  Nevents = 0;
  nRuns = 0;
  Noccu = 0;
}

// ------------ method called once each job just after ending the event loop  ------------
void AlCaHOCalibProducer::endJob() {
  if (m_occupancy) { 
    for (int ij=0; ij<5; ij++) {
      ho_occupency[ij]->Scale(1./std::max(1,Noccu));
    }
  }
  edm::LogInfo("HOCalib") <<" AlCaHOCalibProducer processed event "<< Nevents;
}

// ------------ method called once each job just after ending the event loop  ------------
//void 
//  AlCaHOCalibProducer::endRun(edm::Run const & run, edm::EventSetup & es) {}
// ------------ method called once each job just before starting event loop  ------------
void 
AlCaHOCalibProducer::beginRun(edm::Run const & run,
			      const edm::EventSetup & es) {
  
  // HCAL channel status map ****************************************
  edm::ESHandle<HcalChannelQuality> hcalChStatus;    
  es.get<HcalChannelQualityRcd>().get("withTopo", hcalChStatus );
  theHcalChStatus = hcalChStatus.product();
}


void AlCaHOCalibProducer::fillHOStore(const reco::TrackRef& ncosm,
				      HOCalibVariables& tmpHOCalib,
				      std::unique_ptr<HOCalibVariableCollection> &hostore,
				      int Noccu_old, int indx,
				      edm::Handle<reco::TrackCollection> cosmicmuon,
				      edm::View<reco::Muon>::const_iterator muon1,
				      const edm::Event& iEvent,
				      const edm::EventSetup& iSetup) {

  edm::ESHandle<CaloGeometry> pG;
  iSetup.get<CaloGeometryRecord>().get(pG);
  const CaloGeometry* geo = pG.product();
  const CaloSubdetectorGeometry* gHO = 
    geo->getSubdetectorGeometry(DetId::Hcal,HcalOuter);
 
  // Get Hcal Severity Level Computer, so that the severity of each rechit flag/status may be determined
  //  edm::ESHandle<HcalSeverityLevelComputer> hcalSevLvlComputerHndl;
  iSetup.get<HcalSeverityLevelComputerRcd>().get(hcalSevLvlComputerHndl);
  const HcalSeverityLevelComputer* hcalSevLvlComputer = hcalSevLvlComputerHndl.product();

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
      
      
  PositionType trkpos(posx, posy, posz);
      
  CLHEP::Hep3Vector tmpmuon3v(posx, posy, posz);
  CLHEP::Hep3Vector tmpmuondir(momx, momy, momz);
      
  bool samedir = (tmpmuon3v.dot(tmpmuondir) >0) ? true : false;
  for (int ij=0; ij<3; ij++) {tmpHOCalib.caloen[ij] = 0.0;}
  int inearbymuon = 0;
  localxhor0 = localyhor0 = 20000;  //GM for 22OCT07 data

  if (m_cosmic) {
    int ind(0);
    for(reco::TrackCollection::const_iterator ncosmcor=cosmicmuon->begin();
	ncosmcor != cosmicmuon->end();  ++ncosmcor,++ind) {
      if (indx==ind) continue;
      CLHEP::Hep3Vector tmpmuon3vcor;
      CLHEP::Hep3Vector tmpmom3v;
      if (iiner==1) {
	tmpmuon3vcor = CLHEP::Hep3Vector((*ncosmcor).innerPosition().X(),(*ncosmcor).innerPosition().Y(),(*ncosmcor).innerPosition().Z());
	tmpmom3v = CLHEP::Hep3Vector((*ncosmcor).innerMomentum().X(),(*ncosmcor).innerMomentum().Y(),(*ncosmcor).innerMomentum().Z());
      } else {
	tmpmuon3vcor = CLHEP::Hep3Vector((*ncosmcor).outerPosition().X(),(*ncosmcor).outerPosition().Y(),(*ncosmcor).outerPosition().Z());
	tmpmom3v = CLHEP::Hep3Vector((*ncosmcor).outerMomentum().X(),(*ncosmcor).outerMomentum().Y(),(*ncosmcor).outerMomentum().Z());	
	    
      }

      if (tmpmom3v.mag()<0.2 || (*ncosmcor).ndof()<5) continue;
	  
      double angle = tmpmuon3v.angle(tmpmuon3vcor);
      if (angle < 7.5*CLHEP::deg) {inearbymuon=1;} //  break;}
	
      //	if (muonTagsi_.label() =="cosmicMuons") {
      if (angle <7.5*CLHEP::deg) { tmpHOCalib.caloen[0] +=1.;}
      if (angle <15.0*CLHEP::deg) { tmpHOCalib.caloen[1] +=1.;}
      if (angle <35.0*CLHEP::deg) { tmpHOCalib.caloen[2] +=1.;}
    }
  } else {
    //            if (muonTags_.label() =="muons") {
    edm::Handle<CaloTowerCollection> calotower;
    iEvent.getByToken(tok_tower_, calotower);
    
    for (CaloTowerCollection::const_iterator calt = calotower->begin();
	 calt !=calotower->end(); calt++) {
      //CMSSW_2_1_x	const math::XYZVector towermom = (*calt).momentum();
      double ith = (*calt).momentum().theta();
      double iph = (*calt).momentum().phi();
      
      CLHEP::Hep3Vector calo3v(sin(ith)*cos(iph), sin(ith)*sin(iph), cos(ith));
	
      double angle = tmpmuon3v.angle(calo3v);
	
      if (angle < 7.5*CLHEP::deg) {tmpHOCalib.caloen[0] += calt->emEnergy()+calt->hadEnergy();}
      if (angle < 15*CLHEP::deg) {tmpHOCalib.caloen[1] += calt->emEnergy()+calt->hadEnergy();}
      if (angle < 35*CLHEP::deg) {tmpHOCalib.caloen[2] += calt->emEnergy()+calt->hadEnergy();}
    }
  }
  if ((m_cosmic) || (tmpHOCalib.caloen[0] <=10.0)) {

    GlobalPoint glbpt(posx, posy, posz);
      
    double mom = sqrt(momx*momx + momy*momy +momz*momz);
      
    momx /= mom;
    momy /= mom;
    momz /= mom;
      
    DirectionType trkdir(momx, momy, momz);
      
    tmpHOCalib.trkdr = (*ncosm).d0();
    tmpHOCalib.trkdz = (*ncosm).dz();
    tmpHOCalib.nmuon = (m_cosmic) ? cosmicmuon->size() : 1;
    tmpHOCalib.trkvx = glbpt.x();
    tmpHOCalib.trkvy = glbpt.y();
    tmpHOCalib.trkvz = glbpt.z();
    tmpHOCalib.trkmm = mom*charge;
    tmpHOCalib.trkth = trkdir.theta();
    tmpHOCalib.trkph = trkdir.phi();
    tmpHOCalib.isect2 = -2;
    tmpHOCalib.isect = -2;
    tmpHOCalib.hodx = -100;
    tmpHOCalib.hody = -100;
    tmpHOCalib.hoang = -2.0;
    tmpHOCalib.momatho = -2;
    tmpHOCalib.ndof  = (inearbymuon ==0) ? (int)(*ncosm).ndof() : -(int)(*ncosm).ndof();
    tmpHOCalib.chisq = (*ncosm).normalizedChi2(); // max(1.,tmpHOCalib.ndof);
    if (!m_cosmic) {      
      reco::MuonEnergy muonenr = muon1->calEnergy();
      reco::MuonIsolation iso03 = muon1->isolationR03();    
      reco::MuonIsolation iso05 = muon1->isolationR05(); 

      tmpHOCalib.tkpt03 = iso03.sumPt;
      tmpHOCalib.ecal03 = iso05.sumPt; // iso03.emEt+muonenr.em;
      tmpHOCalib.hcal03 = iso03.hadEt+muonenr.had;
    }
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
    edm::ESHandle<MagneticField> theMagField;
    iSetup.get<IdealMagneticFieldRecord>().get(theMagField );     

    SteppingHelixPropagator myHelix(&*theMagField,anyDirection);
    myHelix.setMaterialMode(false);
    myHelix.applyRadX0Correction(true);
    double phiho = trkpos.phi();
    if (phiho<0) phiho +=CLHEP::twopi;
      
    int iphisect_dt=int(6*(phiho+10.0*CLHEP::deg)/CLHEP::pi); //for u 18/12/06
    if (iphisect_dt>=12) iphisect_dt=0;

    int iphisect = -1;
    bool ipath = false;
    for (int kl = 0; kl<=2; kl++) {

      int iphisecttmp = (kl<2) ? iphisect_dt + kl : iphisect_dt - 1;
      if (iphisecttmp <0) iphisecttmp = 11;
      if (iphisecttmp >=12) iphisecttmp = 0;
	
      double phipos = iphisecttmp*CLHEP::pi/6.;
      double phirot = phipos;
	
      GlobalVector xLocal(-sin(phirot), cos(phirot), 0.);
      GlobalVector yLocal(0., 0., 1.);
      GlobalVector zLocal = xLocal.cross(yLocal).unit();
      //    GlobalVector zLocal(cos(phirot), sin(phirot), 0.0); 
	

      FreeTrajectoryState freetrajectorystate_ = getFreeTrajectoryState(*ncosm,&(*theMagField), iiner, samedir);
	
      Surface::RotationType rot(xLocal, yLocal, zLocal);
	
      for (int ik=1; ik>=0; ik--) { //propagate track in two HO layers

	double     radial = rHOL1;
	if (ik==0) radial = rHOL0;

	Surface::PositionType pos(radial*cos(phipos), radial*sin(phipos), 0.);
	PlaneBuilder::ReturnType aPlane = PlaneBuilder().plane(pos,rot);

	auto aPlane2 = new Plane(pos,rot);

	SteppingHelixStateInfo steppingHelixstateinfo_;
	myHelix.propagate(SteppingHelixStateInfo(freetrajectorystate_), (*aPlane2), steppingHelixstateinfo_);

	if (steppingHelixstateinfo_.isValid()) {
	    
	  GlobalPoint hotrkpos2xx(steppingHelixstateinfo_.position().x(), steppingHelixstateinfo_.position().y(), steppingHelixstateinfo_.position().z());
	    
	  if (ik==1) {
	    HcalDetId ClosestCell = (HcalDetId) gHO->getClosestCell(hotrkpos2xx);
	    int ixeta = ClosestCell.ieta();
	    int ixphi = ClosestCell.iphi();
	    tmpHOCalib.isect2 = 100*std::abs(ixeta+50)+std::abs(ixphi);
	  }
	    

	  GlobalVector hotrkpos2(steppingHelixstateinfo_.position().x(), steppingHelixstateinfo_.position().y(), steppingHelixstateinfo_.position().z());
	  CLHEP::Hep3Vector hotrkdir2(steppingHelixstateinfo_.momentum().x(), steppingHelixstateinfo_.momentum().y(),steppingHelixstateinfo_.momentum().z());

	  LocalVector lclvt0 = (*aPlane).toLocal(hotrkpos2);
	    
	  double xx = lclvt0.x();
	  double yy = lclvt0.y();
	    
	  if (ik ==1) {
	    if ((std::abs(yy) < 130 && xx >-64.7 && xx <138.2) //Ring-0
		||(std::abs(yy) > 130 && std::abs(yy) <700 && xx >-76.3 && xx <140.5)) { //Ring +-1,2
	      ipath = true;  //Only look for tracks which as hits in layer 1
	      iphisect = iphisecttmp;
	    }
	  }
	    
	  if (iphisect != iphisecttmp) continue; //Look for ring-0 only when ring1 is accepted for that sector
	    
	  switch (ik) 
	    {
	    case 0 : 
	      xhor0 = xx; //lclvt0.x();
	      yhor0 = yy; //lclvt0.y();
	      break;
	    case 1 :
	      xhor1 = xx; //lclvt0.x();
	      yhor1 = yy; //lclvt0.y();
	      tmpHOCalib.momatho = hotrkdir2.mag();
	      tmpHOCalib.hoang = CLHEP::Hep3Vector(zLocal.x(),zLocal.y(),zLocal.z()).dot(hotrkdir2.unit());
	      break;
	    default : break;
	    }
	} else {
	  break;
	}
      }
      if (ipath) break;
    }
    if (ipath) { //If muon crossed HO laeyrs
	
      int ietaho = 50;
      int iphiho = -1;
	
      for (int ij=0; ij<9; ij++) {tmpHOCalib.hosig[ij]=-100.0;}
      for (int ij=0; ij<18; ij++) {tmpHOCalib.hocorsig[ij]=-100.0;}
      for (int ij=0; ij<9; ij++) {tmpHOCalib.hbhesig[ij]=-100.0;}
      tmpHOCalib.hocro = -100;
      tmpHOCalib.htime = -1000;
	
      int isect = 0;

      findHOEtaPhi(iphisect, ietaho, iphiho);
	
      if (ietaho !=0 && iphiho !=0 && std::abs(iring)<=2) { //Muon passed through a tower
	isect = 100*std::abs(ietaho+50)+std::abs(iphiho);
	if (std::abs(ietaho) >=netabin || iphiho<0) isect *=-1; //Not extrapolated to any tower
	if (std::abs(ietaho) >=netabin) isect -=1000000;  //not matched with eta
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
	  
	if (m_hbinfo) {
	  for (int ij=0; ij<9; ij++) {tmpHOCalib.hbhesig[ij]=-100.0;}
	    
	  edm::Handle<HBHERecHitCollection> hbheht;// iEvent.getByType(hbheht);
	  iEvent.getByToken(tok_hbhe_,hbheht);
	    
	  if (!(*hbheht).empty()) {
	    if((*hbheht).empty()) throw (int)(*hbheht).size();
	      
	    for (HBHERecHitCollection::const_iterator jk=(*hbheht).begin(); jk!=(*hbheht).end(); jk++){
	      HcalDetId id =(*jk).id();
	      int tmpeta= id.ieta();
	      int tmpphi= id.iphi();
		
	      int deta = tmpeta-ietaho;
	      if (tmpeta<0 && ietaho>0) deta += 1;
	      if (tmpeta>0 && ietaho<0) deta -= 1;

	      //		if (tmpeta==-1 && ietaho== 1) deta = -1;
	      //		if (tmpeta== 1 && ietaho==-1) deta =  1;

	      int dphi = tmpphi-iphiho;
	      if (dphi>nphimx/2) { dphi -=nphimx;}
	      if (dphi<-nphimx/2) { dphi +=nphimx;}

	      //		if (phimn >phimx) {
	      //		  if (dphi==71) dphi=-1;
	      //		  if (dphi==-71) dphi=1;
	      //		}

	      if (m_occupancy) {
		float signal = (*jk).energy();
		//		int tmpeta1 = (tmpeta>0) ? tmpeta -1 : -tmpeta +14; 
		if (signal >-100 && Noccu == Noccu_old) {
		  for (int ij=0; ij<5; ij++) {
		    if (signal >(ij+2)*m_sigma) {
		      ho_occupency[ij]->Fill(tmpeta, tmpphi);
		    }
		  }
		}
	      }
		
	      int ipass2 = (std::abs(deta) <=1 && std::abs(dphi)<=1) ? 1 : 0; //NEED correction in full CMS detector
	      if ( ipass2 ==0 ) continue;
		
	      float signal = (*jk).energy();
	      
	      if (3*(deta+1)+dphi+1<9)  tmpHOCalib.hbhesig[3*(deta+1)+dphi+1] = signal;
	    }
	  }
	} //m_hbinfo #endif
	  
	edm::Handle<HORecHitCollection> hoht;
	iEvent.getByToken(tok_ho_,hoht);
	  
	if (!(*hoht).empty()) {
	  for (HORecHitCollection::const_iterator jk=(*hoht).begin(); jk!=(*hoht).end(); jk++){
	    HcalDetId id =(*jk).id();
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
	    int dphi = tmpphi -iphiho;

	    if (tmpeta<0 && ietaho>0) deta += 1;
	    if (tmpeta>0 && ietaho<0) deta -= 1;
	    //	      if (tmpeta==-1 && ietaho== 1) deta = -1;
	    //	      if (tmpeta== 1 && ietaho==-1) deta =  1;
	      
	    if (dphi>nphimx/2) { dphi -=nphimx;}
	    if (dphi<-nphimx/2) { dphi +=nphimx;}
	    //	      if (phimn>phimx) {
	    //		if (dphi==71) dphi=-1;
	    //		if (dphi==-71) dphi=1;
	    //	      }

	    float signal = (*jk).energy();
	      
	    int ipass2 = (std::abs(deta) <=1 && std::abs(dphi)<=1) ? 1 : 0;
	      
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
		
	      if (3*(deta+1)+dphi+1<9) {
		tmpHOCalib.hosig[3*(deta+1)+dphi+1] = signal; //Again CMS azimuthal near phi 1&72
	      }
	    }
	      
	    if (deta==0 && dphi ==0) {
	      tmpHOCalib.htime = (*jk).time();
	      tmpHOCalib.hoflag = (*jk).flags();

	      // Get Channel Quality information for the given detID
	      unsigned theStatusValue = theHcalChStatus->getValues(id)->getValue();
	      // Now get severity of problems for the given detID, based on the rechit flag word and the channel quality status value
	      int hitSeverity=hcalSevLvlComputer->getSeverityLevel(id, (*jk).flags(),theStatusValue);
	      tmpHOCalib.hoflag = hitSeverity;
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
	
      //GMA	  Npass++;
      if (Noccu == Noccu_old) Noccu++;
      hostore->push_back(tmpHOCalib);	
    } // if (ipath)
  } // Cut on calo energy
}

void AlCaHOCalibProducer::findHOEtaPhi(int iphisect, int& ietaho, int& iphiho) {
  
  //18/12/06 : use only position, not angle phi

 const double etalow[16]={   0.025,  35.195,  70.625, 106.595, 141.565, 180.765, 220.235, 261.385, 304.525, 349.975, 410.025, 452.085, 506.645, 565.025, 627.725, 660.25};
 const double etahgh[16]={  35.145,  70.575, 106.545, 125.505, 180.715, 220.185, 261.335, 304.475, 349.925, 392.575, 452.035, 506.595, 564.975, 627.675, 661.075, 700.25};

  const double philow[6]={-76.27, -35.11, 0.35, 35.81, 71.77, 108.93};  //Ring+/-1 & 2
  const double phihgh[6]={-35.81, -0.35, 35.11, 71.07, 108.23, 140.49};

  const double philow00[6]={-60.27, -32.91, 0.35, 33.61, 67.37, 102.23}; //Ring0 L0
  const double phihgh00[6]={-33.61, -0.35, 32.91, 66.67, 101.53, 129.49};

  const double philow01[6]={-64.67, -34.91, 0.35, 35.61, 71.37, 108.33}; //Ring0 L1
  const double phihgh01[6]={-35.61, -0.35, 34.91, 70.67, 107.63, 138.19};

  iring = -10;

  double tmpdy =  std::abs(yhor1);
  for (int ij=0; ij<netabin; ij++) {
    if (tmpdy >etalow[ij] && tmpdy <etahgh[ij]) {
      ietaho = ij+1; 
      float tmp1 = fabs(tmpdy-etalow[ij]);
      float tmp2 = fabs(tmpdy-etahgh[ij]);
 
      localyhor1 = (tmp1 < tmp2) ? -tmp1 : tmp2;
      if (yhor1 <0) localyhor1 *=-1.;

      if (ij<4) iring =0;
      if (ij>=4 && ij<10) iring=1;
      if (ij>=10 && ij<netabin) iring=2;
      break;
    }
  }

  int tmpphi = 0;
  int tmpphi0 = 0;

  if (ietaho >4) { //Ring 1 and 2
    for (int ij=0; ij<6; ij++) {
      if (xhor1 >philow[ij] && xhor1 <phihgh[ij]) { 
	tmpphi=ij+1; 
	float tmp1 = fabs(xhor1-philow[ij]);
	float tmp2 = fabs(xhor1-phihgh[ij]);
	localxhor1 = (tmp1 < tmp2) ? -tmp1 : tmp2;
	break;
      }
    }
  } else {  //Ring 0
    for (int ij=0; ij<6; ij++) {
      if (xhor1 >philow01[ij] && xhor1 <phihgh01[ij]) { 
	tmpphi=ij+1; 
	float tmp1 = fabs(xhor1-philow01[ij]);
	float tmp2 = fabs(xhor1-phihgh01[ij]);
	localxhor1 = (tmp1 < tmp2) ? -tmp1 : tmp2;
	break;
      }
    }

    for (int ij=0; ij<6; ij++) {
      if (xhor0 >philow00[ij] && xhor0 <phihgh00[ij]) { 
	tmpphi0=ij+1; 
	float tmp1 = fabs(xhor0-philow00[ij]);
	float tmp2 = fabs(xhor0-phihgh00[ij]);
	localxhor0 = (tmp1 < tmp2) ? -tmp1 : tmp2;
	if (tmpphi !=tmpphi0) localxhor0 +=10000.;
	break;
      }
    }

    double tmpdy =  std::abs(yhor0);
    for (int ij=0; ij<4; ij++) {
      if (tmpdy >etalow[ij] && tmpdy <etahgh[ij]) {
	float tmp1 = fabs(tmpdy-etalow[ij]);
	float tmp2 = fabs(tmpdy-etahgh[ij]);
	localyhor0 = (tmp1 < tmp2) ? -tmp1 : tmp2;
	if (yhor0 <0) localyhor0 *=-1.;
	if (ij+1 != ietaho)  localyhor0 +=10000.;
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
    if (std::abs(ietaho) >netabin) { //Initialised with 50
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


