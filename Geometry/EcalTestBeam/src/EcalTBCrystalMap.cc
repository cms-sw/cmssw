#include "Geometry/EcalTestBeam/interface/EcalTBCrystalMap.h"

EcalTBCrystalMap::EcalTBCrystalMap(std::string const & MapFileName) {
                                                                                                                                          
  // check if File exists
  std::ifstream * input= new std::ifstream(MapFileName.c_str(), std::ios::in);
  if(! (*input))
    {
      throw cms::Exception("FileNotFound", "Ecal TB Crystal map file not found") 
        <<  "\n" << MapFileName << " could not be opened.\n";
   }
  if (input){

    int nCrysCount = 0;

    while (nCrysCount <= NCRYSTAL ) {
      
      (*input) >> crysIndex >> crysEta >>  crysPhi;
      map_[std::pair<double,double>(crysEta,crysPhi)] = crysIndex;

      nCrysCount++;
      
    }

    input->close();
  }

}

EcalTBCrystalMap::~EcalTBCrystalMap() {
}

int EcalTBCrystalMap::CrystalIndex(double thisEta, double thisPhi) {

  int thisCrysIndex = 0;

  CrystalTBIndexMap::const_iterator mapItr = map_.find(std::make_pair(thisEta,thisPhi));
  if ( mapItr != map_.end() ) {
    thisCrysIndex = mapItr->second;
  }

  return thisCrysIndex;

}

void EcalTBCrystalMap::findCrystalAngles(const int thisCrysIndex, double & thisEta, double & thisPhi) {
  
  thisEta = thisPhi = 0.;

  if ( thisCrysIndex < 1 || thisCrysIndex > NCRYSTAL ) {
    edm::LogError("OutOfBounds") << "Required crystal number " << thisCrysIndex << " outside range";
    return;
  }

  for ( CrystalTBIndexMap::const_iterator mapItr = map_.begin() ; mapItr != map_.end() ; ++mapItr) {
    int theCrysIndex = mapItr->second;
    if ( theCrysIndex == thisCrysIndex ) {
      thisEta = (mapItr->first).first;
      thisPhi = (mapItr->first).second;
      return;
    }
  }

}
