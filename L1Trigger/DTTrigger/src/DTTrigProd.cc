#ifndef L1Trigger_DTTrigger_DTTrigProd_cc
#define L1Trigger_DTTrigger_DTTrigProd_cc

#include <iostream>
#include <memory>

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "L1Trigger/DTTrigger/interface/DTTrig.h"
#include "L1Trigger/DTTriggerServerPhi/interface/DTChambPhSegm.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhDigi.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThDigi.h"

using namespace std;
using namespace edm;

//typedef unsigned char myint8;
typedef vector<DTChambPhSegm>  InternalPhiSegm;
typedef InternalPhiSegm::const_iterator InternalPhiSegm_iterator;
typedef vector<DTChambThSegm>  InternalThSegm;
typedef InternalThSegm::const_iterator InternalThSegm_iterator;
typedef vector<L1MuDTChambPhDigi>  Phi_Container;
typedef vector<L1MuDTChambThDigi>  Theta_Container;

class DTTrigProd: public EDProducer{
public:
  /// Constructor
  DTTrigProd(const ParameterSet& pset){
    produces<L1MuDTChambPhContainer>();
    produces<L1MuDTChambThContainer>();
    MyTrig = new DTTrig();
    //MyTrig->config()->setParam("Debugging level","fullTRACO");
  }

  /// Destructor
  ~DTTrigProd(){
    delete MyTrig;
  }

  /// Create Trigger Units before starting event processing
  void beginJob(const EventSetup & iEventSetup){
    MyTrig->createTUs(iEventSetup);
    cout << "****TU's Created" << endl;
  }
  
  /// Main produce 
  void produce(Event & iEvent, const EventSetup& iEventSetup){
    MyTrig->triggerReco(iEvent,iEventSetup);
    cout << "***Trigger algorithm run for " <<iEvent.id() << endl;
    // Convert Phi Segments
    InternalPhiSegm myPhiSegm;
    myPhiSegm=MyTrig->TSPhTrigs();
    Phi_Container outPhi;

    for (InternalPhiSegm_iterator it=myPhiSegm.begin(); it!=myPhiSegm.end();it++)
      outPhi.push_back(L1MuDTChambPhDigi((*it).step(),
				     (*it).ChamberId().wheel(),
				     (*it).ChamberId().sector(),
				     (*it).ChamberId().station(),
				     (*it).phi(),
				     (*it).phiB(),
				     (*it).code(),
				     !(*it).isFirst(),
				     0
				     ));
    // Convert Theta Segments
    InternalThSegm myThetaSegm;
    myThetaSegm=MyTrig->TSThTrigs();
    Theta_Container outTheta;
    
    for (InternalThSegm_iterator it=myThetaSegm.begin(); it!=myThetaSegm.end();it++){
      int pos[7], qual[7];
      for (int i=0; i<7; i++){
	pos[i] =(*it).position(i);
	qual[i]=(*it).quality(i);
      }
      outTheta.push_back(L1MuDTChambThDigi((*it).step(),
					   (*it).ChamberId().wheel(),
					   (*it).ChamberId().sector(),
					   (*it).ChamberId().station(),
					   pos,
					   qual
					   ));
    }
    // Write everything into the event
    std::auto_ptr<L1MuDTChambPhContainer> resultPhi (new L1MuDTChambPhContainer);
    resultPhi->setContainer(outPhi);
    iEvent.put(resultPhi);
    std::auto_ptr<L1MuDTChambThContainer> resultTheta (new L1MuDTChambThContainer);
    resultTheta->setContainer(outTheta);
    iEvent.put(resultTheta);
  }
  
private:

  DTTrig* MyTrig;

};
 
#endif

