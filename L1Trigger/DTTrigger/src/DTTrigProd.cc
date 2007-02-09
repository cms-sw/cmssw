//-------------------------------------------------
//
/**  \class DTTrigProd
 *     Main EDProducer for the DTTPG
 *
 *
 *   $Date: 2006/10/13 10:55:31 $
 *   $Revision: 1.3 $
 *
 *   \author C. Battilana
 *
 */
//
//--------------------------------------------------

// This class's header
#include "L1Trigger/DTTrigger/interface/DTTrigProd.h"

// Framework related classes
#include "FWCore/Framework/interface/ESHandle.h"


// Data Formats classes
#include "L1Trigger/DTSectorCollector/interface/DTSectCollPhSegm.h"
#include "L1Trigger/DTSectorCollector/interface/DTSectCollThSegm.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhDigi.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThDigi.h"

// DataFormats interface
typedef vector<DTSectCollPhSegm>  SectCollPhiColl;
typedef SectCollPhiColl::const_iterator SectCollPhiColl_iterator;
typedef vector<DTSectCollThSegm>  SectCollThetaColl;
typedef SectCollThetaColl::const_iterator SectCollThetaColl_iterator;

// Collaborating classes
#include <iostream>

DTTrigProd::DTTrigProd(const ParameterSet& pset){
  
  produces<L1MuDTChambPhContainer>();
  produces<L1MuDTChambThContainer>();
  my_debug = pset.getUntrackedParameter<bool>("debug");
  my_DTTFnum = pset.getParameter<bool>("DTTFSectorNumbering");
  my_BXoffset = pset.getParameter<int>("BXOffset");
  if (my_DTTFnum) 
    cout << "[DTTrigProd] Using DTTF Sector Numbering" << endl;
  my_trig = new DTTrig(pset.getParameter<ParameterSet>("DTTPGParameters"));

}

DTTrigProd::~DTTrigProd(){

  delete my_trig;

}

void DTTrigProd::beginJob(const EventSetup & iEventSetup){

  my_trig->createTUs(iEventSetup);
  if (my_debug)
    cout << "[DTTrigProd] TU's Created" << endl;

}

void DTTrigProd::produce(Event & iEvent, const EventSetup& iEventSetup){

  my_trig->triggerReco(iEvent,iEventSetup);
  if (my_debug)
    cout << "[DTTrigProd] Trigger algorithm run for " <<iEvent.id() << endl;
  
  // Convert Phi Segments
  SectCollPhiColl myPhiSegments;
  myPhiSegments = my_trig->SCPhTrigs();
  vector<L1MuDTChambPhDigi> outPhi;

  SectCollPhiColl_iterator SCPCend = myPhiSegments.end();
  for (SectCollPhiColl_iterator it=myPhiSegments.begin();it!=SCPCend;++it){
    int step = (*it).step() - my_BXoffset; // This moves correct BX to 0 (useful for DTTF)
    int sc_sector = (*it).SCId().sector();
    if (my_DTTFnum == true) sc_sector--; // Modified for DTTF numbering [0-11]
    outPhi.push_back(L1MuDTChambPhDigi(step,
				       (*it).ChamberId().wheel(),
				       sc_sector,
				       (*it).ChamberId().station(),
				       (*it).phi(),
				       (*it).phiB(),
				       (*it).code(),
				       !(*it).isFirst(),
				       0
				       ));
  }

  // Convert Theta Segments
  SectCollThetaColl myThetaSegments;
  myThetaSegments = my_trig->SCThTrigs();
  vector<L1MuDTChambThDigi> outTheta;
  
  SectCollThetaColl_iterator SCTCend = myThetaSegments.end();
  for (SectCollThetaColl_iterator it=myThetaSegments.begin();it!=SCTCend;++it){
    int pos[7], qual[7];
    for (int i=0; i<7; i++){
      pos[i] =(*it).position(i);
      qual[i]=(*it).quality(i);
    }
    int step =(*it).step() - my_BXoffset; // This moves correct BX to 0 (useful for DTTF)
    int sc_sector =  (*it).SCId().sector();
    if (my_DTTFnum == true) sc_sector--; // Modified for DTTF numbering [0-11]
    outTheta.push_back(L1MuDTChambThDigi( step,
					 (*it).ChamberId().wheel(),
					 sc_sector,
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

