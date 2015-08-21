// -*- C++ -*-
//
// Package:    HcalTopologyIdealEP
// Class:      HcalTopologyIdealEP
// 
/**\class HcalTopologyIdealEP HcalTopologyIdealEP.h tmp/HcalTopologyIdealEP/interface/HcalTopologyIdealEP.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jeremiah Mans
//         Created:  Mon Oct  3 11:35:27 CDT 2005
//
//

#include "Geometry/HcalEventSetup/interface/HcalTopologyIdealEP.h"
#include "Geometry/CaloTopology/interface/HcalTopologyRestrictionParser.h"
#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

//#define DebugLog

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
HcalTopologyIdealEP::HcalTopologyIdealEP(const edm::ParameterSet& conf)
  : m_restrictions(conf.getUntrackedParameter<std::string>("Exclude")),
    m_pSet(conf) {
#ifdef DebugLog
  std::cout << "HcalTopologyIdealEP::HcalTopologyIdealEP" << std::endl;
  edm::LogInfo("HCAL") << "HcalTopologyIdealEP::HcalTopologyIdealEP";
#endif
  setWhatProduced(this,
                  &HcalTopologyIdealEP::produce,
                  dependsOn( &HcalTopologyIdealEP::hcalRecordCallBack ));
}


HcalTopologyIdealEP::~HcalTopologyIdealEP() { }

void HcalTopologyIdealEP::fillDescriptions( edm::ConfigurationDescriptions & descriptions ) {
  edm::ParameterSetDescription hcalTopologyConstants;
  hcalTopologyConstants.add<std::string>( "mode", "HcalTopologyMode::LHC" );
  hcalTopologyConstants.add<int>( "maxDepthHB", 2 );
  hcalTopologyConstants.add<int>( "maxDepthHE", 3 );  

  edm::ParameterSetDescription hcalSLHCTopologyConstants;
  hcalSLHCTopologyConstants.add<std::string>( "mode", "HcalTopologyMode::SLHC" );
  hcalSLHCTopologyConstants.add<int>( "maxDepthHB", 7 );
  hcalSLHCTopologyConstants.add<int>( "maxDepthHE", 7 );

  edm::ParameterSetDescription desc;
  desc.addUntracked<std::string>( "Exclude", "" );
  desc.addOptional<edm::ParameterSetDescription>( "hcalTopologyConstants", hcalTopologyConstants );
  descriptions.add( "hcalTopologyIdeal", desc );

  edm::ParameterSetDescription descSLHC;
  descSLHC.addUntracked<std::string>( "Exclude", "" );
  descSLHC.addOptional<edm::ParameterSetDescription>( "hcalTopologyConstants", hcalSLHCTopologyConstants );
  descriptions.add( "hcalTopologyIdealSLHC", descSLHC );  
}

//
// member functions
//

// ------------ method called to produce the data  ------------
HcalTopologyIdealEP::ReturnType
HcalTopologyIdealEP::produce(const HcalRecNumberingRecord& iRecord) {
#ifdef DebugLog
  std::cout << "HcalTopologyIdealEP::produce(const IdealGeometryRecord& iRecord)" << std::endl;
  edm::LogInfo("HCAL") << "HcalTopologyIdealEP::produce(const HcalGeometryRecord& iRecord)";
#endif  
  edm::ESHandle<HcalDDDRecConstants> pHRNDC;
  iRecord.get( pHRNDC );
  const HcalDDDRecConstants* hdc = &(*pHRNDC);

#ifdef DebugLog
  std::cout << "mode = " << hdc->getTopoMode() << ", maxDepthHB = " 
	    << hdc->getMaxDepth(0) << ", maxDepthHE = " << hdc->getMaxDepth(1) 
	    << ", maxDepthHF = " << hdc->getMaxDepth(2) << std::endl;
  edm::LogInfo("HCAL") << "mode = " << hdc->getTopoMode() << ", maxDepthHB = " 
		       << hdc->getMaxDepth(0) << ", maxDepthHE = " 
		       << hdc->getMaxDepth(1) << ", maxDepthHF = " 
		       << hdc->getMaxDepth(2);
#endif
  ReturnType myTopo(new HcalTopology(hdc));

  HcalTopologyRestrictionParser parser(*myTopo);
  if (!m_restrictions.empty()) {
    std::string error=parser.parse(m_restrictions);
    if (!error.empty()) {
      throw cms::Exception("Parse Error","Parse error on Exclude "+error);
    }
  }
  return myTopo ;
}


