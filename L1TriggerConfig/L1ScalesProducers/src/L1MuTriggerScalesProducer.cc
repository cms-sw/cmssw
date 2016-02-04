//-------------------------------------------------
//
//   \class L1MuTriggerScalesProducer
//
//   Description:  A class to produce the L1 mu emulator scales record in the event setup
//
//   $Date: 2008/04/17 23:33:41 $
//   $Revision: 1.2 $
//
//   Author :
//   I. Mikulec
//
//--------------------------------------------------
#include "L1TriggerConfig/L1ScalesProducers/interface/L1MuTriggerScalesProducer.h"

L1MuTriggerScalesProducer::L1MuTriggerScalesProducer(const edm::ParameterSet& ps)
  : m_scales( ps.getParameter<int>("nbitPackingDTEta"),
	      ps.getParameter<bool>("signedPackingDTEta"),
	      ps.getParameter<int>("nbinsDTEta"),
	      ps.getParameter<double>("minDTEta"),
	      ps.getParameter<double>("maxDTEta"),
	      ps.getParameter<int>("offsetDTEta"),

	      ps.getParameter<int>("nbitPackingCSCEta"),
	      ps.getParameter<int>("nbinsCSCEta"),
	      ps.getParameter<double>("minCSCEta"),
	      ps.getParameter<double>("maxCSCEta"),

	      ps.getParameter<std::vector<double> >("scaleRPCEta"),
	      ps.getParameter<int>("nbitPackingBrlRPCEta"),
	      ps.getParameter<bool>("signedPackingBrlRPCEta"),
	      ps.getParameter<int>("nbinsBrlRPCEta"),
	      ps.getParameter<int>("offsetBrlRPCEta"),
	      ps.getParameter<int>("nbitPackingFwdRPCEta"),
	      ps.getParameter<bool>("signedPackingFwdRPCEta"),
	      ps.getParameter<int>("nbinsFwdRPCEta"),
	      ps.getParameter<int>("offsetFwdRPCEta"),

	      ps.getParameter<int>("nbitPackingGMTEta"),
	      ps.getParameter<int>("nbinsGMTEta"),
	      ps.getParameter<std::vector<double> >("scaleGMTEta"),

	      ps.getParameter<int>("nbitPackingPhi"),
	      ps.getParameter<bool>("signedPackingPhi"),
	      ps.getParameter<int>("nbinsPhi"),
	      ps.getParameter<double>("minPhi"),
	      ps.getParameter<double>("maxPhi") )
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

   std::auto_ptr<L1MuTriggerScales> l1muscale =
     std::auto_ptr<L1MuTriggerScales>( new L1MuTriggerScales( m_scales ) );

   return l1muscale ;
}

