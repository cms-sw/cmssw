//-------------------------------------------------
//
//   \class L1MuTriggerScalesProducer
//
//   Description:  A class to produce the L1 mu emulator scales record in the event setup
//
//   $Date: $
//   $Revision: $
//
//   Author :
//   I. Mikulec
//
//--------------------------------------------------
#include "L1TriggerConfig/L1ScalesProducers/interface/L1MuTriggerScalesProducer.h"

L1MuTriggerScalesProducer::L1MuTriggerScalesProducer(const edm::ParameterSet& ps)
{
 
  setWhatProduced(this, &L1MuTriggerScalesProducer::produceL1MuTriggerScales);
  
}


L1MuTriggerScalesProducer::~L1MuTriggerScalesProducer() {}


//
// member functions
//

// ------------ method called to produce the data  ------------
std::auto_ptr<L1MuTriggerScales> 
L1MuTriggerScalesProducer::produceL1MuTriggerScales(const L1MuTriggerScalesRcd& iRecord)
{
   using namespace edm::es;

   std::auto_ptr<L1MuTriggerScales> l1muscale = std::auto_ptr<L1MuTriggerScales>( new L1MuTriggerScales() );

   return l1muscale ;
}

