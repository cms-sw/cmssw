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
// $Id: CaloTowerConstituentsMapBuilder.cc,v 1.9 2013/04/26 09:38:11 yana Exp $
//
//

#include "Geometry/CaloEventSetup/plugins/CaloTowerConstituentsMapBuilder.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include <zlib.h>
#include <strings.h>

//
// constructors and destructor
//
CaloTowerConstituentsMapBuilder::CaloTowerConstituentsMapBuilder(const edm::ParameterSet& iConfig) :
    mapFile_(iConfig.getUntrackedParameter<std::string>("MapFile",""))
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
  edm::ParameterSetDescription desc;
  desc.addUntracked<std::string>( "MapFile", "" );
  descriptions.add( "caloTowerConstituents", desc );
}

// ------------ method called to produce the data  ------------
CaloTowerConstituentsMapBuilder::ReturnType
CaloTowerConstituentsMapBuilder::produce(const IdealGeometryRecord& iRecord)
{
   edm::ESHandle<HcalTopology> topology ;
   iRecord.get( topology ) ;

   std::auto_ptr<CaloTowerConstituentsMap> prod( new CaloTowerConstituentsMap( &*topology ));

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

void
CaloTowerConstituentsMapBuilder::parseTextMap( const std::string& filename, CaloTowerConstituentsMap& theMap )
{
    edm::FileInPath eff( filename );

    gzFile gzed = gzopen( eff.fullPath().c_str(), "rb" );
    
    while( !gzeof( gzed ))
    {
	char line[1024];
	int ieta, iphi, rawid;
	if( 0 != gzgets( gzed, line, 1023 ))
	{
	    if( index( line, '#' ) != 0 )*( index( line, '#' )) = 0;
	    int ct = sscanf( line, "%i %d %d", &rawid, &ieta, &iphi );
	    if( ct == 3 )
	    {
		DetId detid( rawid );
		CaloTowerDetId tid( ieta, iphi );
		theMap.assign( detid, tid );
	    }
	}
    }
    gzclose( gzed );
}
