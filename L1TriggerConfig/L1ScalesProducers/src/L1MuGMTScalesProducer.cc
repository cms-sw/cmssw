//-------------------------------------------------
//
//   \class L1MuGMTScalesProducer
//
//   Description:  A class to produce the L1 GMT emulator scales record in the event setup
//
//   $Date: 2008/04/17 23:33:41 $
//   $Revision: 1.2 $
//
//   Author :
//   I. Mikulec
//
//--------------------------------------------------
#include "L1TriggerConfig/L1ScalesProducers/interface/L1MuGMTScalesProducer.h"

L1MuGMTScalesProducer::L1MuGMTScalesProducer(const edm::ParameterSet& ps)
  : m_scales( ps.getParameter<int>("nbitPackingReducedEta"),
	      ps.getParameter<int>("nbinsReducedEta"),
	      ps.getParameter<std::vector<double> >("scaleReducedEtaDT"),
	      ps.getParameter<std::vector<double> >("scaleReducedEtaBrlRPC"),
	      ps.getParameter<std::vector<double> >("scaleReducedEtaCSC"),
	      ps.getParameter<std::vector<double> >("scaleReducedEtaFwdRPC"),

	      ps.getParameter<int>("nbitPackingDeltaEta"),
	      ps.getParameter<bool>("signedPackingDeltaEta"),
	      ps.getParameter<int>("nbinsDeltaEta"),
	      ps.getParameter<double>("minDeltaEta"),
	      ps.getParameter<double>("maxDeltaEta"),
	      ps.getParameter<int>("offsetDeltaEta"),

	      ps.getParameter<int>("nbitPackingDeltaPhi"),
	      ps.getParameter<bool>("signedPackingDeltaPhi"),
	      ps.getParameter<int>("nbinsDeltaPhi"),
	      ps.getParameter<double>("minDeltaPhi"),
	      ps.getParameter<double>("maxDeltaPhi"),
	      ps.getParameter<int>("offsetDeltaPhi"),

	      ps.getParameter<int>("nbitPackingOvlEtaDT"),
	      ps.getParameter<int>("nbinsOvlEtaDT"),
	      ps.getParameter<double>("minOvlEtaDT"),
	      ps.getParameter<double>("maxOvlEtaDT"),

	      ps.getParameter<int>("nbitPackingOvlEtaCSC"),
	      ps.getParameter<int>("nbinsOvlEtaCSC"),
	      ps.getParameter<double>("minOvlEtaCSC"),
	      ps.getParameter<double>("maxOvlEtaCSC"),

	      ps.getParameter<std::vector<double> >("scaleOvlEtaRPC"),
	      ps.getParameter<int>("nbitPackingOvlEtaBrlRPC"),
	      ps.getParameter<int>("nbinsOvlEtaBrlRPC"),
	      ps.getParameter<int>("nbitPackingOvlEtaFwdRPC"),
	      ps.getParameter<int>("nbinsOvlEtaFwdRPC") )
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

   std::auto_ptr<L1MuGMTScales> l1muscale = std::auto_ptr<L1MuGMTScales>( new L1MuGMTScales( m_scales ) );

   return l1muscale ;
}

