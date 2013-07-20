/** \file
 *
 *  $Date: 2012/09/13 14:53:56 $
 *  $Revision: 1.6 $
 *  \author S. Bolognesi - INFN TO
 */

#include "FilterByLTC.h"


#include "FWCore/Framework/interface/Event.h"
// #include "DataFormats/Common/interface/Handle.h"
// #include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/LTCDigi/interface/LTCDigi.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include <iostream>

using namespace std;
using namespace edm;

FilterByLTC::FilterByLTC(const ParameterSet& pset) : 
  nEventsProcessed(0),
  nEventsSelected(0),
  ltcTag_(pset.getParameter<edm::InputTag>("ltcTag"))
{
  theTriggerSource = pset.getParameter<int>("triggerSource");
}

FilterByLTC::~FilterByLTC(){
}

bool FilterByLTC::filter(Event & event, const EventSetup& eventSetup){

  bool selectThisEvent = false;
  
  nEventsProcessed++;

  edm::Handle<LTCDigiCollection> ltcdigis;
  event.getByLabel(ltcTag_, ltcdigis);

  bool DTtrig=false, CSCtrig=false, RPCtrig=false;
  for (std::vector<LTCDigi>::const_iterator ltc_it = ltcdigis->begin(); ltc_it != ltcdigis->end(); ltc_it++){
    if ((*ltc_it).HasTriggered(0))
      DTtrig=true;
    if ((*ltc_it).HasTriggered(1))
      CSCtrig=true;
    if ((*ltc_it).HasTriggered(2))
      RPCtrig=true;
    if ((*ltc_it).HasTriggered(3))
      RPCtrig=true;
    if ((*ltc_it).HasTriggered(4))
      RPCtrig=true;
  }
 
  switch ( theTriggerSource ) {

  case 1 : 
    //only DT trigger
    if(DTtrig && !CSCtrig && !RPCtrig)
      selectThisEvent = true;
    break;

  case 2 : 
    //only CSC trigger
    if(!DTtrig && CSCtrig && !RPCtrig)
      selectThisEvent = true;
    break;

  case 3 : 
    //only RPC trigger
    if(!DTtrig && !CSCtrig && RPCtrig)
      selectThisEvent = true;
    break;

  case 4 : 
    //both DT && CSC trigger
    if(DTtrig && CSCtrig && !RPCtrig)
      selectThisEvent = true;
    break;

  case 5 : 
    // both DT && RPC trigger
    if(DTtrig && !CSCtrig && RPCtrig)
      selectThisEvent = true;
    break;

  case 6 :
    // both CSC && RPC trigger
    if(!DTtrig && CSCtrig && RPCtrig)
      selectThisEvent = true;
    break;

  case 7 :
    // all CSC && RPC && DT trigger
    if(DTtrig && CSCtrig && RPCtrig)
      selectThisEvent = true;
    break;

  case 8 :
    // No DT
    if(!DTtrig)
      selectThisEvent = true;
    break;

  case 9 :
    // No CSC
    if(!CSCtrig)
      selectThisEvent = true;
    break;

  case 10 :
    // No RPC
    if(!RPCtrig)
      selectThisEvent = true;
    break;

  case 11:
    //DT at least
    if(DTtrig)
      selectThisEvent = true;
    break;
  
  case 12:
    //CSC at least
    if(CSCtrig)
      selectThisEvent = true;
    break;
   
  case 13:
    //RPC at least
    if(RPCtrig)
      selectThisEvent = true;
    break;
   
  default : 
    cout<<"[FilterByLTC] Wrong trigger source selected"<<endl;

  }

  if(selectThisEvent)
    nEventsSelected++;

  //cout<<"DT "<<DTtrig<<"  CSC "<<CSCtrig<<"  RPC "<<RPCtrig<<endl;
  //cout<<"events selected "<<selectThisEvent<<endl;
  //cout<<"total "<<nEventsSelected<<" / "<<nEventsProcessed<<endl;

  return selectThisEvent;
}

DEFINE_FWK_MODULE(FilterByLTC);


  
