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
// $Id: BSCTrigger.cc,v 1.4 2009/02/01 15:26:48 boudoul Exp $
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
#include "FWCore/ParameterSet/interface/InputTag.h"
//
// class declaration
//

class BSCTrigger : public edm::EDProducer {
public:
  explicit BSCTrigger(const edm::ParameterSet&);
  ~BSCTrigger();

private:
  virtual void beginJob(const edm::EventSetup&) ;
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
  theNinner_=iConfig.getParameter<int>("minbiasInnerMin");
  theNouter_=iConfig.getParameter<int>("minbiasOuterMin");
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
void
BSCTrigger::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  std::vector<L1GtTechnicalTrigger> ttVec(ttBits_.size());
  std::vector<float>EnergyBX(32);
  std::vector<float>EnergyBXMinusDt(32);
  int ZMinnerBX=0, ZMouterBX=0;
  int ZPinnerBX=0, ZPouterBX=0;
  ++nevt_;
  std::auto_ptr<L1GtTechnicalTriggerRecord> BscRecord;
  float MipFraction=0.7;
  float MipEnergy=0.0027;
  float theThreshold=MipFraction*MipEnergy;
  edm::Handle<edm::PSimHitContainer> theBSCHitContainer;
  iEvent.getByLabel(TheHits_tag_,theBSCHitContainer);
  
  if (!theBSCHitContainer.failedToGet()) {
    for ( int c=0;c<32;++c){
      EnergyBX[c]=0;
      EnergyBXMinusDt[c]=0;
    }
    edm::PSimHitContainer::const_iterator itHit, jtHit;
    float dt1,dt2;
    dt1=theCoincidence_/2 + theResolution_;
    dt2=theCoincidence_/2 - theResolution_;
    if ( edm::isDebugEnabled() ) LogDebug("BSCTrig")<<" ----------------new event ---with "<<theBSCHitContainer->size()<<" hits in the BSC";
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
    for ( unsigned int ipad = 0 ; ipad<32; ipad++) {

      if ( edm::isDebugEnabled() ) LogTrace("BSCTrig")<<" EnergyBX["<<ipad<<"]="<<EnergyBX[ipad];

      if ( EnergyBX[ipad] > theThreshold ) {
	if ( isZplus(ipad)) {
	  if ( isInner(ipad) ) ZPinnerBX++;
	  else ZPouterBX++;
	} else {
	  if ( isInner(ipad) ) ZMinnerBX++;
	  else ZMouterBX++;	
	}
      } 
    }
    if ( edm::isDebugEnabled() ) LogTrace("BSCTrig")<<" Zplus I="<<ZPinnerBX<<" Zminus I="<<ZMinnerBX<<" Zplus O="<<ZPouterBX<<"  Zminus O="<<ZMouterBX;

    //halo 
    for ( unsigned i=0; i< ttBits_.size();++i ){
      bool bit=false;      
      if ( names_.at(i) == names_[0] ) {
	if ( EnergyBX[8] > theThreshold && EnergyBXMinusDt[27] > theThreshold ) bit=true;  
	if ( EnergyBX[9] > theThreshold && EnergyBXMinusDt[26] > theThreshold ) bit=true;  
	if ( EnergyBX[10] > theThreshold && EnergyBXMinusDt[25] > theThreshold ) bit=true;  
	if ( EnergyBX[11] > theThreshold && EnergyBXMinusDt[24] > theThreshold ) bit=true;  
	if ( EnergyBX[12] > theThreshold && EnergyBXMinusDt[31] > theThreshold ) bit=true;  
	if ( EnergyBX[13] > theThreshold && EnergyBXMinusDt[30] > theThreshold ) bit=true;  
	if ( EnergyBX[14] > theThreshold && EnergyBXMinusDt[29] > theThreshold ) bit=true;  
	if ( EnergyBX[15] > theThreshold && EnergyBXMinusDt[28] > theThreshold ) bit=true;  
	ttVec.at(i)=L1GtTechnicalTrigger(names_.at(i), ttBits_.at(i), 0, bit) ;
      }
      else if (  names_.at(i) == names_[1]) { 
	if ( EnergyBX[0] > theThreshold && EnergyBXMinusDt[18] > theThreshold ) bit=true;  
	if ( EnergyBX[1] > theThreshold && EnergyBXMinusDt[19] > theThreshold ) bit=true;  
	if ( EnergyBX[2] > theThreshold && EnergyBXMinusDt[16] > theThreshold ) bit=true;  
	if ( EnergyBX[3] > theThreshold && EnergyBXMinusDt[17] > theThreshold ) bit=true;  
	if ( EnergyBX[4] > theThreshold && EnergyBXMinusDt[22] > theThreshold ) bit=true;  
	if ( EnergyBX[5] > theThreshold && EnergyBXMinusDt[23] > theThreshold ) bit=true;  
	if ( EnergyBX[6] > theThreshold && EnergyBXMinusDt[20] > theThreshold ) bit=true;  
	if ( EnergyBX[7] > theThreshold && EnergyBXMinusDt[21] > theThreshold ) bit=true;  
	ttVec.at(i)=L1GtTechnicalTrigger(names_.at(i), ttBits_.at(i), 0, bit) ;
      }
      else if (  names_.at(i) == names_[2]) { 	
	if ( EnergyBXMinusDt[8] > theThreshold && EnergyBX[27] > theThreshold ) bit=true;  
	if ( EnergyBXMinusDt[9] > theThreshold && EnergyBX[26] > theThreshold ) bit=true;  
	if ( EnergyBXMinusDt[10] > theThreshold && EnergyBX[25] > theThreshold ) bit=true;  
	if ( EnergyBXMinusDt[11] > theThreshold && EnergyBX[24] > theThreshold ) bit=true;  
	if ( EnergyBXMinusDt[12] > theThreshold && EnergyBX[31] > theThreshold ) bit=true;  
	if ( EnergyBXMinusDt[13] > theThreshold && EnergyBX[30] > theThreshold ) bit=true;  
	if ( EnergyBXMinusDt[14] > theThreshold && EnergyBX[29] > theThreshold ) bit=true;  
	if ( EnergyBXMinusDt[15] > theThreshold && EnergyBX[28] > theThreshold ) bit=true;  
	ttVec.at(i)=L1GtTechnicalTrigger(names_.at(i), ttBits_.at(i), 0, bit) ;
      }
      else if (  names_.at(i) == names_[3] ) { 	
	if ( EnergyBXMinusDt[0] > theThreshold && EnergyBX[18] > theThreshold ) bit=true;  
	if ( EnergyBXMinusDt[1] > theThreshold && EnergyBX[19] > theThreshold ) bit=true;  
	if ( EnergyBXMinusDt[2] > theThreshold && EnergyBX[16] > theThreshold ) bit=true;  
	if ( EnergyBXMinusDt[3] > theThreshold && EnergyBX[17] > theThreshold ) bit=true;  
	if ( EnergyBXMinusDt[4] > theThreshold && EnergyBX[22] > theThreshold ) bit=true;  
	if ( EnergyBXMinusDt[5] > theThreshold && EnergyBX[23] > theThreshold ) bit=true;  
	if ( EnergyBXMinusDt[6] > theThreshold && EnergyBX[20] > theThreshold ) bit=true;  
	if ( EnergyBXMinusDt[7] > theThreshold && EnergyBX[21] > theThreshold ) bit=true; 
 	ttVec.at(i)=L1GtTechnicalTrigger(names_.at(i), ttBits_.at(i), 0, bit) ;
      }

    // the minbias trigger
      else if (  names_.at(i) == names_[4] ){
	if (ZPinnerBX > theNinner_ && ZMinnerBX > theNinner_ ) bit=true;	
        ttVec.at(i)=L1GtTechnicalTrigger(names_.at(i), ttBits_.at(i), 0, bit);
      }
      else if (  names_.at(i) == names_[5] ) {
	if ( ZPouterBX > theNouter_ && ZMouterBX > theNouter_ )  bit=true;	
	ttVec.at(i)=L1GtTechnicalTrigger(names_.at(i), ttBits_.at(i), 0, bit); 
      }

      if ( edm::isDebugEnabled() ) LogTrace("AnaBsc") << "bit: "<<ttBits_[i] << " VALUE:"<<bit ;

    }
  } else ttVec.clear();
  std::auto_ptr<L1GtTechnicalTriggerRecord> output(new L1GtTechnicalTriggerRecord());
  output->setGtTechnicalTrigger(ttVec);    
  iEvent.put(output);
}
// ------------ method called once each job just before starting event loop  ------------
void 
BSCTrigger::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
BSCTrigger::endJob() {
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
