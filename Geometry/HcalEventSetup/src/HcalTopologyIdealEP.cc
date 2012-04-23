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
// $Id: HcalTopologyIdealEP.cc,v 1.4.2.2 2011/03/14 19:03:34 rpw Exp $
//
//

#include "Geometry/HcalEventSetup/interface/HcalTopologyIdealEP.h"
#include "Geometry/CaloTopology/interface/HcalTopologyRestrictionParser.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
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
  m_restrictions(conf.getUntrackedParameter<std::string>("Exclude","")),
  m_h2mode(conf.getUntrackedParameter<bool>("H2Mode",false)),
  m_SLHCmode(conf.getUntrackedParameter<bool>("SLHCMode",false))
{
  // copied from HcalHitRelabeller, input like {1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,4}
  m_segmentation.resize(29);
  std::vector<int> segmentation;
  for (int iring=1; iring<=29; ++iring) {
    char name[10];
    snprintf(name,10,"Eta%d",iring);
    if(conf.existsAs<std::vector<int> >(name, false))
    {
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
	HcalTopology::Mode mode=HcalTopology::md_LHC;
	
	if (m_h2mode){
		edm::LogInfo("HCAL") << "Using H2 Topology";
		mode=HcalTopology::md_H2;	
	}
	if (m_SLHCmode) {
		edm::LogInfo("HCAL") << "Using SLHC Topology";
		mode=HcalTopology::md_SLHC;
	}
	
   ReturnType myTopo(new HcalTopology(mode));

   HcalTopologyRestrictionParser parser(*myTopo);
   if (!m_restrictions.empty()) {
     std::string error=parser.parse(m_restrictions);
     if (!error.empty()) {
       throw cms::Exception("Parse Error","Parse error on Exclude "+error);
     }
   }

   // see if any depth segmentation needs to be added
   for(std::vector<RingSegmentation>::const_iterator ringSegItr = m_segmentation.begin();
       ringSegItr != m_segmentation.end(); ++ringSegItr)
   {
     myTopo->setDepthSegmentation(ringSegItr->ring, ringSegItr->segmentation);
   } 
   return myTopo ;
}


