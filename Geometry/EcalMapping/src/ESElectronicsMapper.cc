#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/EcalMapping/interface/ESElectronicsMapper.h"

ESElectronicsMapper::ESElectronicsMapper(const edm::ParameterSet& ps) {
  lookup_ = ps.getParameter<edm::FileInPath>("LookupTable");

  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
      for (int k = 0; k < 40; ++k)
        for (int m = 0; m < 40; ++m) {
          fed_[i][j][k][m] = -1;
          kchip_[i][j][k][m] = -1;
        }

  // read in look-up table
  int nLines, z, iz, ip, ix, iy, fed, kchip, pace, bundle, fiber, optorx;
  std::ifstream file;
  file.open(lookup_.fullPath().c_str());
  if (file.is_open()) {
    file >> nLines;

    for (int i = 0; i < nLines; ++i) {
      file >> iz >> ip >> ix >> iy >> fed >> kchip >> pace >> bundle >> fiber >> optorx;

      if (iz == -1)
        z = 2;
      else
        z = iz;

      fed_[z - 1][ip - 1][ix - 1][iy - 1] = fed;
      kchip_[z - 1][ip - 1][ix - 1][iy - 1] = kchip;
    }

  } else {
    edm::LogVerbatim("EcalMapping") << "ESElectronicsMapper::ESElectronicsMapper : Look up table file can not be found in "
              << lookup_.fullPath().c_str();
  }

  // EE-ES FEDs mapping
  int eefed[18] = {601, 602, 603, 604, 605, 606, 607, 608, 609, 646, 647, 648, 649, 650, 651, 652, 653, 654};
  int nesfed[18] = {10, 7, 9, 10, 8, 10, 8, 10, 8, 10, 7, 8, 8, 8, 9, 8, 10, 10};
  int esfed[18][10] = {{520, 522, 523, 531, 532, 534, 535, 545, 546, 547},
                       {520, 522, 523, 534, 535, 546, 547},
                       {520, 522, 523, 524, 525, 534, 535, 537, 539},
                       {520, 522, 523, 524, 525, 534, 535, 537, 539, 540},
                       {522, 523, 524, 525, 535, 537, 539, 540},
                       {524, 525, 528, 529, 530, 537, 539, 540, 541, 542},
                       {528, 529, 530, 531, 532, 541, 542, 545},
                       {528, 529, 530, 531, 532, 541, 542, 545, 546, 547},
                       {529, 530, 531, 532, 542, 545, 546, 547},
                       {548, 549, 551, 560, 561, 563, 564, 572, 573, 574},
                       {548, 549, 560, 561, 563, 564, 574},
                       {548, 549, 551, 553, 563, 564, 565, 566},
                       {551, 553, 554, 563, 564, 565, 566, 568},
                       {553, 554, 555, 556, 565, 566, 568, 570},
                       {553, 554, 555, 556, 565, 566, 568, 570, 571},
                       {553, 554, 555, 556, 557, 568, 570, 571},
                       {555, 556, 557, 560, 561, 570, 571, 572, 573, 574},
                       {548, 549, 557, 560, 561, 570, 571, 572, 573, 574}};

  for (int i = 0; i < 18; ++i) {  // loop over EE feds
    std::vector<int> esFeds;
    esFeds.reserve(nesfed[i]);
    for (int esFed = 0; esFed < nesfed[i]; esFed++)
      esFeds.emplace_back(esfed[i][esFed]);
    ee_es_map_.insert(make_pair(eefed[i], esFeds));
  }
}

int ESElectronicsMapper::getFED(const ESDetId& id) {
  int zside;
  if (id.zside() < 0)
    zside = 2;
  else
    zside = id.zside();

  return fed_[zside - 1][id.plane() - 1][id.six() - 1][id.siy() - 1];
}

int ESElectronicsMapper::getFED(int zside, int plane, int x, int y) { return fed_[zside - 1][plane - 1][x - 1][y - 1]; }

std::vector<int> ESElectronicsMapper::GetListofFEDs(const std::vector<int>& eeFEDs) const {
  std::vector<int> esFEDs;
  GetListofFEDs(eeFEDs, esFEDs);
  return esFEDs;
}

void ESElectronicsMapper::GetListofFEDs(const std::vector<int>& eeFEDs, std::vector<int>& esFEDs) const {
  for (int eeFED : eeFEDs) {
    std::map<int, std::vector<int> >::const_iterator itr = ee_es_map_.find(eeFED);
    if (itr == ee_es_map_.end())
      continue;
    std::vector<int> fed = itr->second;
    for (int j : fed) {
      esFEDs.emplace_back(j);
    }
  }

  sort(esFEDs.begin(), esFEDs.end());
  std::vector<int>::iterator it = unique(esFEDs.begin(), esFEDs.end());
  esFEDs.erase(it, esFEDs.end());
}

int ESElectronicsMapper::getKCHIP(const ESDetId& id) {
  int zside;
  if (id.zside() < 0)
    zside = 2;
  else
    zside = id.zside();

  return kchip_[zside - 1][id.plane() - 1][id.six() - 1][id.siy() - 1];
}

int ESElectronicsMapper::getKCHIP(int zside, int plane, int x, int y) {
  return kchip_[zside - 1][plane - 1][x - 1][y - 1];
}
