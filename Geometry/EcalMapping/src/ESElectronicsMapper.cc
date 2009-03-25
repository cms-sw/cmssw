#include "Geometry/EcalMapping/interface/ESElectronicsMapper.h"

ESElectronicsMapper::ESElectronicsMapper(const edm::ParameterSet& ps) {

  lookup_ = ps.getUntrackedParameter<edm::FileInPath>("LookupTable");

  for (int i=0; i<2; ++i) 
    for (int j=0; j<2; ++j) 
      for (int k=0; k<40; ++k) 
	for (int m=0; m<40; ++m) 
	  fed_[i][j][k][m] = -1; 
 
  // read in look-up table
  int nLines, z, iz, ip, ix, iy, fed, kchip, pace, bundle, fiber, optorx;
  std::ifstream file;
  file.open(lookup_.fullPath().c_str());
  if( file.is_open() ) {

    file >> nLines;

    for (int i=0; i<nLines; ++i) {
      file>> iz >> ip >> ix >> iy >> fed >> kchip >> pace >> bundle >> fiber >> optorx;
      
      if (iz==-1) z = 2;
      else z = iz;
      
      fed_[z-1][ip-1][ix-1][iy-1] = fed + 519;
    }

  } else {
    std::cout<<"ESElectronicsMapper::ESElectronicsMapper : Look up table file can not be found in "<<lookup_.fullPath().c_str()<<std::endl;
  }

  // EE-ES FEDs mapping
  int eefed[18] = {601, 602, 603, 604, 605, 606, 607, 608, 609, 646, 647, 648, 649, 650, 651, 652, 653, 654};
  int nesfed[18] = {7, 5, 5, 5, 5, 8, 5, 7, 5, 8, 5, 5, 5, 5, 7, 5, 7, 5};
  int esfed[18][8] = {
    {524, 525, 528, 539, 540, 541, 542},
    {528, 529, 530, 541, 542},
    {529, 530, 541, 542, 545},
    {531, 532, 545, 546, 547},
    {531, 532, 545, 546, 547},
    {520, 522, 523, 532, 534, 535, 546, 547},
    {520, 522, 523, 534, 535},
    {522, 523, 524, 525, 535, 537, 539},
    {524, 525, 537, 539, 540},
    {548, 549, 560, 561, 563, 572, 573, 574},
    {548, 549, 551, 563, 564},
    {548, 549, 551, 563, 564},
    {551, 553, 554, 565, 566},
    {553, 554, 565, 566, 568},
    {553, 554, 555, 556, 568, 570, 571},
    {555, 556, 557, 570, 571},
    {556, 557, 560, 570, 571, 572, 573},
    {560, 561, 572, 573, 574}
  };
  
  for (int i=0; i<18; ++i) 
    for (int j=0; j<nesfed[i]; ++j)
      ee_es_map_[eefed[i]].push_back(esfed[i][j]);
  
}

int ESElectronicsMapper::getFED(const ESDetId& id) { 

  int zside;
  if (id.zside()<0) zside = 2;
  else zside = id.zside();

  return fed_[zside-1][id.plane()-1][id.six()-1][id.siy()-1]; 
} 

int ESElectronicsMapper::getFED(int zside, int plane, int x, int y) { 

  return fed_[zside-1][plane-1][x-1][y-1]; 
} 

std::vector<int> ESElectronicsMapper::GetListofFEDs(const std::vector<int> eeFEDs) const {
  std::vector<int> esFEDs;
  GetListofFEDs(eeFEDs, esFEDs);
  return esFEDs;
}

void ESElectronicsMapper::GetListofFEDs(std::vector<int> eeFEDs, std::vector<int> & esFEDs) const {

  for (uint i=0; i<eeFEDs.size(); ++i) {
    std::vector<int> esfed = ee_es_map_.find(eeFEDs[i])->second;
    for (uint j=0; j<esfed.size(); ++j) esFEDs.push_back(esfed[j]);
  }

  sort(esFEDs.begin(), esFEDs.end());
  std::vector<int>::iterator it = unique(esFEDs.begin(), esFEDs.end());
  esFEDs.erase(it, esFEDs.end());

}
