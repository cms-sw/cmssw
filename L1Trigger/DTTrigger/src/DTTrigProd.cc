//-------------------------------------------------
//
/**  \class DTTrigProd
 *     Main EDProducer for the DTTPG
 *
 *
 *   $Date: 2006/09/12 $
 *   $Revision: 1.1 $
 *
 *   \author C. Battilana
 *
 */
//
//--------------------------------------------------

// This class's header
#include "L1Trigger/DTTrigger/interface/DTTrigProd.h"

// Collaborating classes
#include <iostream>

const double DTTrigProd::myTtoTDC = 32./25.;

DTTrigProd::DTTrigProd(const ParameterSet& pset){
  produces<L1MuDTChambPhContainer>();
  produces<L1MuDTChambThContainer>();
  bool globaldelay = pset.getUntrackedParameter<bool>("globalSync");
  double syncdelay = pset.getUntrackedParameter<double>("syncDelay");
  stringstream myos;
  myos << syncdelay;
  if (globaldelay) {
    cout << "Using same synchronization for all the stations" << endl;
    MyTrig = new DTTrig();
    MyTrig->config()->setParam("Programmable Dealy",myos.str());
    double ftdelay = pset.getUntrackedParameter<double>("globalSyncValue");
    //MyTrig->config()->setParamValue("BTI Fine sync delay","psetdelay",ftdelay*10);      
    MyTrig->config()->setParamValue("BTI setup time","psetdelay",ftdelay*myTtoTDC);
    cout << "****Delay set to " << ftdelay  << " ns (as set in parameterset)" << endl; 
  }
  else {
    cout << "Using chamber by chamber sync configuration" << endl;
    MyTrig = new DTTrig(pset.getUntrackedParameter<ParameterSet>("L1DTFineSync"),myos.str());
  }
  //MyTrig->config()->setParam("Debugging level","fullTRACO");
}

DTTrigProd::~DTTrigProd(){
  delete MyTrig;
}

void DTTrigProd::beginJob(const EventSetup & iEventSetup){
  MyTrig->createTUs(iEventSetup);
  cout << "****TU's Created" << endl;
}

void DTTrigProd::produce(Event & iEvent, const EventSetup& iEventSetup){

  MyTrig->triggerReco(iEvent,iEventSetup);
  cout << "***Trigger algorithm run for " <<iEvent.id() << endl;
  // Convert Phi Segments
  InternalPhiSegm myPhiSegm;
  myPhiSegm=MyTrig->SCTrigs();
  Phi_Container outPhi;
  for (InternalPhiSegm_iterator it=myPhiSegm.begin(); it!=myPhiSegm.end();it++){
    int ch_sector = (*it).ChamberId().sector();
    int sc_sector = 0;
    switch (ch_sector){
    case 13:
      sc_sector = 4;
      break;
    case 14:
      sc_sector = 10;
      break;
    default:
      sc_sector = ch_sector; // Is it correct or DTTF needs ch_sector -1?
      break;
    }
    outPhi.push_back(L1MuDTChambPhDigi((*it).step(),
				       (*it).ChamberId().wheel(),
				       sc_sector, //(*it).ChamberId().sector(),
				       (*it).ChamberId().station(),
				       (*it).phi(),
				       (*it).phiB(),
				       (*it).code(),
				       !(*it).isFirst(),
				       0
				       ));
  }

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

