// -*- C++ -*-
//
// Package:    BSCTrigger
// Class:      BSCTrigger
// 
/**\class BSCTrigger BSCTrigger.cc L1TriggerOffline/BSCTriggerSimulation/src/BSCTrigger.cc

Description: <one line class summary>

Implementation:
<Notes on implementation>
*/
//
// Original Author:  Muriel VANDER DONCKT *:0
//         Created:  Wed Jul 16 16:11:05 CEST 2008
// $Id: BSCTrigger.cc,v 1.10 2010/10/07 22:02:55 yjlee Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GtTechnicalTrigger.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GtTechnicalTriggerRecord.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"


//
// class declaration
//

class BSCTrigger : public edm::EDProducer {
public:
  explicit BSCTrigger(const edm::ParameterSet&);
  ~BSCTrigger();

private:
  virtual void beginJob() ;
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  int   getBSCNum(int id, float z);
  bool  isInner(int id);
  bool  isZplus(int id);
    // ----------member data ---------------------------
  std::vector<unsigned> ttBits_;
  std::vector<std::string> names_;
  unsigned nEvt_;
  float theCoincidence_;
  float theResolution_;
  int theNinner_;
  int theNouter_;      
  int nevt_;
  edm::InputTag TheHits_tag_;
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
BSCTrigger::BSCTrigger(const edm::ParameterSet& iConfig)
{
  ttBits_=iConfig.getParameter< std::vector<unsigned> >("bitNumbers");
  names_= iConfig.getParameter< std::vector<std::string> >("bitNames");
  theCoincidence_= iConfig.getParameter<double>("coincidence");
  theResolution_= iConfig.getParameter<double>("resolution");
  TheHits_tag_= iConfig.getParameter<edm::InputTag>("theHits");
  produces<L1GtTechnicalTriggerRecord>();  
  nevt_=0;
}


BSCTrigger::~BSCTrigger()
{
 
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void BSCTrigger::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  std::vector<L1GtTechnicalTrigger> ttVec(ttBits_.size());
  std::vector<float>EnergyBX(32);
  std::vector<float>EnergyBXMinusDt(32);
  int ZMinnerBX=0, ZMouterBX=0;
  int ZPinnerBX=0, ZPouterBX=0;
  int ZMinnerBXMinusDt=0, ZMouterBXMinusDt=0;
  int ZPinnerBXMinusDt=0, ZPouterBXMinusDt=0;
  ++nevt_;
  std::auto_ptr<L1GtTechnicalTriggerRecord> BscRecord;
  float MipFraction=0.5;
  float MipEnergy=0.0027;
  float theThreshold=MipFraction*MipEnergy;

  edm::Handle<CrossingFrame<PSimHit> > cf;
  iEvent.getByLabel(TheHits_tag_, cf);
   
  if (!cf.failedToGet()) {
    for ( int c=0;c<32;++c){
      EnergyBX[c]=0;
      EnergyBXMinusDt[c]=0;
    }
    std::auto_ptr<MixCollection<PSimHit> > theBSCHitContainer( new MixCollection <PSimHit>(cf.product()));
    MixCollection<PSimHit>::MixItr itHit;
    float dt1,dt2;
    dt1=theCoincidence_/2 + theResolution_;
    dt2=theCoincidence_/2 - theResolution_;
    if ( edm::isDebugEnabled() ) LogDebug("BSCTrig")<<" ----------------new event ---with "<<theBSCHitContainer->size()<<" hits in the BSC";
    
    // collect total deposited energy in BSC segments -> EnergyBX[segment id], in GeV units ---------------------------------------------------------------
    for (itHit = theBSCHitContainer->begin(); itHit != theBSCHitContainer->end(); ++itHit) {
      float zh=itHit->entryPoint().z()/10;    
      int id=getBSCNum(itHit->detUnitId(),zh);
      if ( id > 31 ) continue;   // the small 2 paddles further from the IP
      float t=itHit->timeOfFlight();

      if ( edm::isDebugEnabled() ) {
	float rh=sqrt(itHit->entryPoint().x()*itHit->entryPoint().x()+itHit->entryPoint().y()*itHit->entryPoint().y())/10;    
	LogTrace("BSCTrig")<<" BSC Num "<<id<<" z="<<zh<<" isZplus="<<isZplus(id)<<" Hit DetId="<<getBSCNum(id,zh)<<" r="<<rh<<" isInner="<<isInner(id);
	LogTrace("BSCTrig")<<" Hit time="<<t<<" accepted range=["<<dt2<<","<<dt1<<"] from a "<<abs(itHit->particleType())<<" with energy " <<itHit->energyLoss();
      }
      if (fabs(t)> dt1 || fabs(t) <dt2 ) continue;
      if (t>0) EnergyBX[id]+=itHit->energyLoss();
      else EnergyBXMinusDt[id]+=itHit->energyLoss();    
    }

    // count number of segments hit in inner/outer and +z, -z ---------------------------------------------------------------------------------------------
    for ( unsigned int ipad = 0 ; ipad<32; ipad++) {
      if ( edm::isDebugEnabled() ) LogTrace("BSCTrig")<<" EnergyBX["<<ipad<<"]="<<EnergyBX[ipad];
      // hits after the bunch crossing
      if ( EnergyBX[ipad] > theThreshold ) {
	if ( isZplus(ipad)) {
	  if ( isInner(ipad) ) ZPinnerBX++;
	  else ZPouterBX++;
	} else {
	  if ( isInner(ipad) ) ZMinnerBX++;
	  else ZMouterBX++;	
	}
      } 
      // hits before the bunch crossing
      if ( EnergyBXMinusDt[ipad] > theThreshold ) {
	if ( isZplus(ipad)) {
	  if ( isInner(ipad) ) ZPinnerBXMinusDt++;
	  else ZPouterBXMinusDt++;
	} else {
	  if ( isInner(ipad) ) ZMinnerBXMinusDt++;
	  else ZMouterBXMinusDt++;	
	}
      } 
    }

    if ( edm::isDebugEnabled() ) LogTrace("BSCTrig")<<" Zplus I="<<ZPinnerBX<<" Zminus I="<<ZMinnerBX<<" Zplus O="<<ZPouterBX<<"  Zminus O="<<ZMouterBX;

    // minimum bias technical triggers that are also connected to 'external condition' triggers -----------------
    bool bit32=false;    // min bias inner >=1
    if(ZPinnerBX >= 1 && ZMinnerBX >= 1) bit32 = true;

    bool bit33=false;    // min bias inner >=2
    if(ZPinnerBX >= 2 && ZMinnerBX >= 2) bit33 = true;

    bool bit34=false;    // min bias OR
    if(ZPinnerBX + ZMinnerBX + ZPouterBX + ZMouterBX >= 1) bit34 = true;

    bool bit35=false;    // high multiplicity
    if(ZPinnerBX == 8 && ZMinnerBX == 8) bit35 = true;

    // beam halo technical triggers ----------------------------------------------------------------
    bool bit36=false;    // beam 2 halo inner
    if(ZPinnerBX >= 1 && ZMinnerBXMinusDt >= 1) bit36 = true;

    bool bit37=false;    // beam 2 halo outer
    if(ZPouterBX >= 1 && ZMouterBXMinusDt >= 1) bit37 = true;

    bool bit38=false;    // beam 1 halo inner
    if(ZMinnerBX >= 1 && ZPinnerBXMinusDt >= 1) bit38 = true;

    bool bit39=false;    // beam 1 halo outer
    if(ZMouterBX >= 1 && ZPouterBXMinusDt >= 1) bit39 = true;

    // minimum bias technical triggers ---------------------------------------------------------
    bool bit40=false;    // min bias all >=1
    if(ZPinnerBX + ZPouterBX >= 1 && ZMinnerBX + ZMouterBX >= 1) bit40 = true;

    bool bit41=false;    // min bias all >=2
    if(ZPinnerBX + ZPouterBX >= 2 && ZMinnerBX + ZMouterBX >= 2) bit41 = true;

    bool bit42=false;    // beam 1 splash (-z)
    if(ZMinnerBX >= 2) bit42 = true;

    bool bit43=false;    // beam 2 splash (+z)
    if(ZPinnerBX >= 2) bit43 = true;

    for ( unsigned i=0; i< ttBits_.size();++i ){
      bool bit = false;
      if ( names_.at(i) == names_[0] ) bit = bit32;
      if ( names_.at(i) == names_[1] ) bit = bit33;
      if ( names_.at(i) == names_[2] ) bit = bit34;
      if ( names_.at(i) == names_[3] ) bit = bit35;
      if ( names_.at(i) == names_[4] ) bit = bit36;
      if ( names_.at(i) == names_[5] ) bit = bit37;
      if ( names_.at(i) == names_[6] ) bit = bit38;
      if ( names_.at(i) == names_[7] ) bit = bit39;
      if ( names_.at(i) == names_[8] ) bit = bit40;
      if ( names_.at(i) == names_[9] ) bit = bit41;
      if ( names_.at(i) == names_[10]) bit = bit42;
      if ( names_.at(i) == names_[11]) bit = bit43;
      ttVec.at(i)=L1GtTechnicalTrigger(names_.at(i), ttBits_.at(i), 0, bit) ;
      if ( edm::isDebugEnabled() ) LogTrace("AnaBsc") << "bit: "<<ttBits_[i] << " VALUE:"<<bit ;
    }
  } else ttVec.clear();
  std::auto_ptr<L1GtTechnicalTriggerRecord> output(new L1GtTechnicalTriggerRecord());
  output->setGtTechnicalTrigger(ttVec);    
  iEvent.put(output);
}
// ------------ method called once each job just before starting event loop  ------------
void BSCTrigger::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void BSCTrigger::endJob() {
}

int BSCTrigger::getBSCNum( int id, float z ) {
  int zside = 0;
  if ( z > 0 ) zside = 1;
  if ( edm::isDebugEnabled() ) {
    int det    = (id&24)>>3;
    int station = id&7;
    LogTrace("BSCTrig")<<"id="<<id<<" zside="<<zside<<" det="<<det<<" station="<<station;
  }
  int BSCNum;
  if (id&16) BSCNum=32+(id&1)+(zside<<1) ;  // small paddles further from IP
  else BSCNum= (id&15)+(zside<<4);          // the BSC on the HF
  return BSCNum;
}

bool BSCTrigger::isInner( int id ){ 
  return ( (id&8)>>3 ) ;
}

bool BSCTrigger::isZplus( int id ){ 
  return ( (id&16)>>4 ) ;
}

//define this as a plug-in
DEFINE_FWK_MODULE(BSCTrigger);
