//---------------------------------------------
//
//   \class L1MuGMTHWFileReader
//
//   Description: Puts the GMT input information from 
//                a GMT ascii HW testfile into the Event   
//
//
//   $Date: 2007/05/02 10:15:06 $
//   $Revision: 1.2 $
//
//   Author :
//   Tobias Noebauer                 HEPHY Vienna
//   Ivan Mikulec                    HEPHY Vienna
//
//--------------------------------------------------
#ifndef L1TriggerGlobalMuonTrigger_L1MuGMTHWFileReader_h
#define L1TriggerGlobalMuonTrigger_L1MuGMTHWFileReader_h

//---------------
// C++ Headers --
//---------------
#include <fstream>

//----------------------
// Base Class Headers --
//----------------------
#include "FWCore/Sources/interface/ProducerSourceFromFiles.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTInputEvent.h"

//---------------------
//-- Class Interface --
//---------------------

class L1MuGMTHWFileReader : public edm::ProducerSourceFromFiles {

 public:
  explicit L1MuGMTHWFileReader(edm::ParameterSet const&,
                               edm::InputSourceDescription const&);

  ~L1MuGMTHWFileReader();

  //read an event from the input stream
  //returns an event with run and event number zero when no more events
  void readNextEvent();


 private:
   virtual bool setRunAndEventInfo(edm::EventID& id, edm::TimeValue_t& time);
   virtual void produce(edm::Event&);

   std::ifstream m_in;
   L1MuGMTInputEvent m_evt;
};

#endif // L1TriggerGlobalMuonTrigger_L1MuGMTHWFileReader_h
