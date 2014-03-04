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
    m_pSet( conf ) {
  std::cout << "HcalTopologyIdealEP::HcalTopologyIdealEP" << std::endl;
  edm::LogInfo("HCAL") << "HcalTopologyIdealEP::HcalTopologyIdealEP";
  setWhatProduced(this,
		  &HcalTopologyIdealEP::produce,
		  dependsOn( &HcalTopologyIdealEP::hcalRecordCallBack ));
}


HcalTopologyIdealEP::~HcalTopologyIdealEP() { 
}

void HcalTopologyIdealEP::fillDescriptions( edm::ConfigurationDescriptions & descriptions ) {

  //  edm::ParameterSetDescription hcalTopologyConstants;
  edm::ParameterSetDescription desc;
  desc.addUntracked<std::string>( "Exclude", "" );
  //  desc.addOptional<edm::ParameterSetDescription>( "hcalTopologyConstants", hcalTopologyConstants );
  descriptions.add( "hcalTopologyIdeal", desc );
}

//
// member functions
//

// ------------ method called to produce the data  ------------
HcalTopologyIdealEP::ReturnType
HcalTopologyIdealEP::produce(const HcalRecNumberingRecord& iRecord) {
  std::cout << "HcalTopologyIdealEP::produce(const IdealGeometryRecord& iRecord)" << std::endl;
  edm::LogInfo("HCAL") << "HcalTopologyIdealEP::produce(const HcalGeometryRecord& iRecord)";
  
  edm::ESHandle<HcalDDDRecConstants> pHRNDC;
  iRecord.get( pHRNDC );
  const HcalDDDRecConstants* hdc = &(*pHRNDC);

  StringToEnumParser<HcalTopologyMode::Mode> eparser;
  HcalTopologyMode::Mode mode = eparser.parseString(hdc->getTopoMode());
  int maxDepthHB = hdc->getMaxDepth(0);
  int maxDepthHE = hdc->getMaxDepth(1);
  std::cout << "mode = " << mode << ", maxDepthHB = " << maxDepthHB << ", maxDepthHE = " << maxDepthHE << std::endl;
  edm::LogInfo("HCAL") << "mode = " << mode << ", maxDepthHB = " << maxDepthHB << ", maxDepthHE = " << maxDepthHE;

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


