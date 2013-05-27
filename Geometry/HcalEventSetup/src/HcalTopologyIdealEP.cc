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
    m_pSet( conf )
{
  //std::cout << "HcalTopologyIdealEP::HcalTopologyIdealEP" << std::endl;
  edm::LogInfo("HCAL") << "HcalTopologyIdealEP::HcalTopologyIdealEP";

  // copied from HcalHitRelabeller, input like {1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,4}
  m_segmentation.resize(29);
  std::vector<int> segmentation;
  for (int iring=1; iring<=29; ++iring) {
    char name[10];
    snprintf(name,10,"Eta%d",iring);
    if(conf.existsAs<std::vector<int> >(name, false)) {
      RingSegmentation entry;
      entry.ring = iring;
      entry.segmentation = conf.getUntrackedParameter<std::vector<int> >(name);
      m_segmentation.push_back(entry);
    }
  }
  setWhatProduced(this);
}


HcalTopologyIdealEP::~HcalTopologyIdealEP()
{ 
}

void
HcalTopologyIdealEP::fillDescriptions( edm::ConfigurationDescriptions & descriptions ) 
{
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
HcalTopologyIdealEP::produce(const IdealGeometryRecord& iRecord)
{
  //   std::cout << "HcalTopologyIdealEP::produce(const IdealGeometryRecord& iRecord)" << std::endl;
  edm::LogInfo("HCAL") <<  "HcalTopologyIdealEP::produce(const IdealGeometryRecord& iRecord)";
    
  using namespace edm::es;

  HcalTopologyMode::Mode mode = HcalTopologyMode::LHC;
  int maxDepthHB = 2;
  int maxDepthHE = 3;
  if( m_pSet.exists( "hcalTopologyConstants" ))
  {
    const edm::ParameterSet hcalTopoConsts( m_pSet.getParameter<edm::ParameterSet>( "hcalTopologyConstants" ));
    StringToEnumParser<HcalTopologyMode::Mode> eparser;
    mode = (HcalTopologyMode::Mode) eparser.parseString(hcalTopoConsts.getParameter<std::string>("mode"));
    maxDepthHB = hcalTopoConsts.getParameter<int>("maxDepthHB");
    maxDepthHE = hcalTopoConsts.getParameter<int>("maxDepthHE");
  }
  //  std::cout << "mode = " << mode << ", maxDepthHB = " << maxDepthHB << ", maxDepthHE = " << maxDepthHE << std::endl;
  edm::LogInfo("HCAL") << "mode = " << mode << ", maxDepthHB = " << maxDepthHB << ", maxDepthHE = " << maxDepthHE;

  ReturnType myTopo(new HcalTopology( mode, maxDepthHB, maxDepthHE ));

  HcalTopologyRestrictionParser parser(*myTopo);
  if (!m_restrictions.empty()) {
    std::string error=parser.parse(m_restrictions);
    if (!error.empty()) {
      throw cms::Exception("Parse Error","Parse error on Exclude "+error);
    }
  }

  // see if any depth segmentation needs to be added
  for(std::vector<RingSegmentation>::const_iterator ringSegItr = m_segmentation.begin();
      ringSegItr != m_segmentation.end(); ++ringSegItr) {
    myTopo->setDepthSegmentation(ringSegItr->ring, ringSegItr->segmentation);
  } 
  return myTopo ;
}


