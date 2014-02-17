//-------------------------------------------------
//
//   \class L1MuTriggerPtScaleProducer
//
//   Description:  A class to produce the L1 mu emulator scales record in the event setup
//
//   $Date: 2008/04/17 23:33:09 $
//   $Revision: 1.1 $
//
//   Author :
//   W. Sun (copied from L1MuTriggerScalesProducer)
//
//--------------------------------------------------
#include "L1TriggerConfig/L1ScalesProducers/interface/L1MuTriggerPtScaleProducer.h"

L1MuTriggerPtScaleProducer::L1MuTriggerPtScaleProducer(const edm::ParameterSet& ps)
  : m_scales( ps.getParameter<int>("nbitPackingPt"),
	      ps.getParameter<bool>("signedPackingPt"),
	      ps.getParameter<int>("nbinsPt"),
	      ps.getParameter<std::vector<double> >("scalePt") )
{
  setWhatProduced(this, &L1MuTriggerPtScaleProducer::produceL1MuTriggerPtScale);
}

L1MuTriggerPtScaleProducer::~L1MuTriggerPtScaleProducer() {}


//
// member functions
//

// ------------ method called to produce the data  ------------
std::auto_ptr<L1MuTriggerPtScale> 
L1MuTriggerPtScaleProducer::produceL1MuTriggerPtScale(const L1MuTriggerPtScaleRcd& iRecord)
{
   using namespace edm::es;

   std::auto_ptr<L1MuTriggerPtScale> l1muscale =
     std::auto_ptr<L1MuTriggerPtScale>( new L1MuTriggerPtScale( m_scales ) );

   return l1muscale ;
}

