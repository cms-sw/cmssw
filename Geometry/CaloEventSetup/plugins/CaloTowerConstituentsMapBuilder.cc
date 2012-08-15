// -*- C++ -*-
//
// Package:    CaloTowerConstituentsMapBuilder
// Class:      CaloTowerConstituentsMapBuilder
// 
/**\class CaloTowerConstituentsMapBuilder CaloTowerConstituentsMapBuilder.h tmp/CaloTowerConstituentsMapBuilder/interface/CaloTowerConstituentsMapBuilder.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jeremiah Mans
//         Created:  Mon Oct  3 11:35:27 CDT 2005
// $Id: CaloTowerConstituentsMapBuilder.cc,v 1.4 2010/03/26 19:35:00 sunanda Exp $
//
//


// user include files
#include "Geometry/CaloEventSetup/plugins/CaloTowerConstituentsMapBuilder.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "CommonTools/Utils/interface/StringToEnumValue.h"
#include <zlib.h>
#include <cstdio>
#include <strings.h>

//
// constructors and destructor
//
CaloTowerConstituentsMapBuilder::CaloTowerConstituentsMapBuilder(const edm::ParameterSet& iConfig) :
    mapFile_(iConfig.getUntrackedParameter<std::string>("MapFile","")),
    m_hcalTopoConsts( iConfig.getParameter<edm::ParameterSet>( "hcalTopologyConstants" ))
  /*
  doStandardHBHE_(iConfig.getParameter<bool>("standardHBHE","true")),
  doStandardHF_(iConfig.getParameter<bool>("standardHF","true")),
  doStandardEB_(iConfig.getParameter<bool>("standardEB","true"))  
  */
{
   //the following line is needed to tell the framework what
   // data is being produced
   setWhatProduced(this);

   //now do what ever other initialization is needed
}


CaloTowerConstituentsMapBuilder::~CaloTowerConstituentsMapBuilder()
{ 
}


//
// member functions
//

// ------------ method called to produce the data  ------------
CaloTowerConstituentsMapBuilder::ReturnType
CaloTowerConstituentsMapBuilder::produce(const IdealGeometryRecord& iRecord)
{
   std::auto_ptr<CaloTowerConstituentsMap> prod(new CaloTowerConstituentsMap( new HcalTopology((HcalTopology::Mode) StringToEnumValue<HcalTopology::Mode>(m_hcalTopoConsts.getParameter<std::string>("mode")),
											       m_hcalTopoConsts.getParameter<int>("maxDepthHB"),
											       m_hcalTopoConsts.getParameter<int>("maxDepthHE"))));
   prod->useStandardHB(true);
   prod->useStandardHE(true);
   prod->useStandardHF(true);
   prod->useStandardHO(true);
   prod->useStandardEB(true);

   if (!mapFile_.empty()) {
     parseTextMap(mapFile_,*prod);
   }
   prod->sort();
   
   return prod;
}

void CaloTowerConstituentsMapBuilder::parseTextMap(const std::string& filename, CaloTowerConstituentsMap& theMap) {
  edm::FileInPath eff(filename);

  gzFile gzed=gzopen(eff.fullPath().c_str(),"rb");
  
  while (!gzeof(gzed)) {
    char line[1024];
    int ieta, iphi, rawid;
    gzgets(gzed,line,1023);
    if (index(line,'#')!=0)  *(index(line,'#'))=0;
    int ct=sscanf(line,"%i %d %d",&rawid,&ieta,&iphi);
    if (ct==3) {
      DetId detid(rawid);
      CaloTowerDetId tid(ieta,iphi);
      theMap.assign(detid,tid);
    }    
  }
  gzclose(gzed);

}
