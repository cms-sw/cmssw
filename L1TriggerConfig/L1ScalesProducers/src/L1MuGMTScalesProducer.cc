//-------------------------------------------------
//
//   \class L1MuGMTScalesProducer
//
//   Description:  A class to produce the L1 GMT emulator scales record in the event setup
//
//   $Date: $
//   $Revision: $
//
//   Author :
//   I. Mikulec
//
//--------------------------------------------------
#include "L1TriggerConfig/L1ScalesProducers/interface/L1MuGMTScalesProducer.h"

L1MuGMTScalesProducer::L1MuGMTScalesProducer(const edm::ParameterSet& ps)
{
 
  setWhatProduced(this, &L1MuGMTScalesProducer::produceL1MuGMTScales);
  
}


L1MuGMTScalesProducer::~L1MuGMTScalesProducer() {}


//
// member functions
//

// ------------ method called to produce the data  ------------
std::auto_ptr<L1MuGMTScales> 
L1MuGMTScalesProducer::produceL1MuGMTScales(const L1MuGMTScalesRcd& iRecord)
{
   using namespace edm::es;

   std::auto_ptr<L1MuGMTScales> l1muscale = std::auto_ptr<L1MuGMTScales>( new L1MuGMTScales() );

   return l1muscale ;
}

