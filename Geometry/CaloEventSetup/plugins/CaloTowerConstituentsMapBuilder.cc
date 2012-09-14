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
// $Id: CaloTowerConstituentsMapBuilder.cc,v 1.6 2012/08/27 15:21:57 yana Exp $
//
//


// user include files
#include "Geometry/CaloEventSetup/plugins/CaloTowerConstituentsMapBuilder.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include <zlib.h>
#include <cstdio>
#include <strings.h>

//
// constructors and destructor
//
CaloTowerConstituentsMapBuilder::CaloTowerConstituentsMapBuilder(const edm::ParameterSet& iConfig) :
    mapFile_(iConfig.getUntrackedParameter<std::string>("MapFile","")),
    m_pSet( iConfig )
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

void
CaloTowerConstituentsMapBuilder::fillDescriptions(edm::ConfigurationDescriptions & descriptions)
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
  desc.addUntracked<std::string>( "MapFile", "" );
  desc.addOptional<edm::ParameterSetDescription>( "hcalTopologyConstants", hcalTopologyConstants );
  descriptions.add( "caloTowerConstituents", desc );

  edm::ParameterSetDescription descSLHC;
  descSLHC.addUntracked<std::string>( "MapFile", "" );
  descSLHC.addOptional<edm::ParameterSetDescription>( "hcalTopologyConstants", hcalSLHCTopologyConstants );
  descriptions.add( "caloTowerConstituentsSLHC", descSLHC );
}

// ------------ method called to produce the data  ------------
CaloTowerConstituentsMapBuilder::ReturnType
CaloTowerConstituentsMapBuilder::produce(const IdealGeometryRecord& iRecord)
{
   HcalTopologyMode::Mode mode = HcalTopologyMode::LHC;
   int maxDepthHB = 2;
   int maxDepthHE = 3;
   if( m_pSet.exists( "hcalTopologyConstants" ))
   {
      const edm::ParameterSet hcalTopoConsts( m_pSet.getParameter<edm::ParameterSet>( "hcalTopologyConstants" ));
      StringToEnumParser<HcalTopologyMode::Mode> parser;
      mode = (HcalTopologyMode::Mode) parser.parseString(hcalTopoConsts.getParameter<std::string>("mode"));
      maxDepthHB = hcalTopoConsts.getParameter<int>("maxDepthHB");
      maxDepthHE = hcalTopoConsts.getParameter<int>("maxDepthHE");
   }

   std::auto_ptr<CaloTowerConstituentsMap> prod( new CaloTowerConstituentsMap( new HcalTopology( mode, maxDepthHB, maxDepthHE )));

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
