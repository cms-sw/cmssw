#include <iostream>

#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/GEMObjects/interface/GEMEMap.h"
#include "CondFormats/DataRecord/interface/GEMEMapRcd.h"

#include <iomanip>

//using namespace std;
//using namespace edm;

// class declaration
class GEMEMapDBReader : public edm::EDAnalyzer {
public:
  explicit GEMEMapDBReader( const edm::ParameterSet& );
  ~GEMEMapDBReader();
  virtual void analyze( const edm::Event&, const edm::EventSetup& );
private:
  bool m_flag;
};

GEMEMapDBReader::GEMEMapDBReader( const edm::ParameterSet& iConfig )
{

}

GEMEMapDBReader::~GEMEMapDBReader(){}

void GEMEMapDBReader::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup ) {

  std::cout << "====== GEMEMapDBReader" << std::endl;

  edm::ESHandle<GEMEMap> readoutMapping;
  iSetup.get<GEMEMapRcd>().get(readoutMapping);
  const GEMEMap* eMap=readoutMapping.product();
  std::cout << eMap << std::endl;
  //if(eMap) return;

  std::cout << "version: " << eMap->version() << " and " << eMap->theVFatMaptype.size() << std::endl;

  // loop on the GEMEMap elements
  // for (std::vector<GEMEMapItem>::const_iterator i=eMap->theEMapItem.begin(); i<eMap->theEMapItem.end();i++){

  //   std::cout <<" Gem Map  Chamber="<<i->ChamberID<<std::endl;

  //   std::vector<int>::const_iterator p=i->positions.begin();
  //   for (std::vector<int>::const_iterator v=i->VFatIDs.begin(); v<i->VFatIDs.end();v++,p++){
  //     std::cout <<" Position="<<std::setw(2)<<std::setfill('0')<<*p
  // 		<<"  vfat=0x"<<std::setw(4)<<std::setfill('0')<<std::hex<<*v<<std::dec<<std::endl;
  //   }

  //   std::vector<GEMEMap::GEMVFatMapInPos>::const_iterator ipos;
  //   for (ipos=eMap->theVFatMapInPos.begin();ipos<eMap->theVFatMapInPos.end();ipos++){
  //     std::cout <<" In position "<<std::setw(2)<<std::setfill('0')<<ipos->position
  // 		<<" I have this map_id "<<ipos->VFATmapTypeId<<std::endl;
  //   }
    
  std::vector<GEMEMap::GEMVFatMaptype>::const_iterator imap;
  for (imap=eMap->theVFatMaptype.begin(); imap<eMap->theVFatMaptype.end();imap++){

    std::cout <<"  Map TYPE "<<imap->VFATmapTypeId<<std::endl;
    for (unsigned int ix=0;ix<imap->strip_number.size();ix++){
      std::cout <<
    	" z_direction "<<imap->z_direction[ix] <<
    	" iEta "<<imap->iEta[ix] <<
    	" iPhi "<<imap->iPhi[ix] <<
    	" depth "<<imap->depth[ix] <<
    	" vfat_position "<<imap->vfat_position[ix] <<
    	" strip_number "<<imap->strip_number[ix] <<
    	" vfat_chnnel_number "<<imap->vfat_chnnel_number[ix]<<std::endl;
    }
  }
  //}
  
}

//define this as a plug-in
DEFINE_FWK_MODULE(GEMEMapDBReader);
