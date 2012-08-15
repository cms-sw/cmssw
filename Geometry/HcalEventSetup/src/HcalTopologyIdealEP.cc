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
#include "CommonTools/Utils/interface/StringToEnumValue.h"
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
HcalTopologyIdealEP::HcalTopologyIdealEP(const edm::ParameterSet& conf) :
  m_restrictions(conf.getUntrackedParameter<std::string>("Exclude",""))
{
  const edm::ParameterSet hcalTopoConsts( conf.getParameter<edm::ParameterSet>( "hcalTopologyConstants" ));
  m_mode = (HcalTopology::Mode) StringToEnumValue<HcalTopology::Mode>(hcalTopoConsts.getParameter<std::string>("mode"));
  m_maxDepthHB = hcalTopoConsts.getParameter<int>("maxDepthHB");
  m_maxDepthHE = hcalTopoConsts.getParameter<int>("maxDepthHE");
    
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


//
// member functions
//

// ------------ method called to produce the data  ------------
HcalTopologyIdealEP::ReturnType
HcalTopologyIdealEP::produce(const IdealGeometryRecord& iRecord)
{
  using namespace edm::es;

  ReturnType myTopo(new HcalTopology(m_mode, m_maxDepthHB, m_maxDepthHE));

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


