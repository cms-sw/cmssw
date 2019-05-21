// prints EEDetId mappings for humans to check over

#include <iostream>
#include <string>
#include <stdexcept>
#include <iomanip>

#include "DataFormats/EcalDetId/interface/EEDetId.h"

using namespace std;

ostream& pcenter(unsigned w, string s) {
  int pad = ((int)w - (int)s.size()) / 2;
  //  if(pad<0) pad = 0;
  for (int i = 0; i < pad; ++i)
    cout << " ";
  cout << s;
  for (int i = pad + s.size(); i < (int)w; ++i)
    cout << " ";
  return cout;
}

int main(int argc, char* argv[]) {
  const int colsize = 8;
#define COL cout << setw(colsize)

  cout << right;
  pcenter(3 * (colsize + 1) - 1, "input") << "|";
  pcenter(11 * (colsize + 1), "detid") << "|";
  pcenter(4 * (colsize + 1), "detid->(isc,ic,iz)->detid") << "|";
  pcenter(3 * (colsize + 1), "sc_detid") << "\n";

  //input
  COL << "ix"
      << " ";
  COL << "iy"
      << " ";
  COL << "iz"
      << "|";

  //detId
  COL << "ix"
      << " ";
  COL << "iy"
      << " ";
  COL << "zside"
      << " ";
  COL << "iquad"
      << " ";
  COL << "Z"
      << " ";
  COL << "phi_out"
      << " ";
  COL << "hash_ind"
      << " ";
  COL << "hash_chk"
      << " ";
  COL << "dense_ind"
      << " ";
  COL << "isc"
      << " ";
  COL << "ic"
      << "|";

  //detid->isc,ic->detid
  COL << "ix"
      << " ";
  COL << "iy"
      << " ";
  COL << "iz"
      << " ";
  COL << "iscic_chk"
      << "\n";

  //sc det id
  COL << "ix"
      << " ";
  COL << "iy"
      << " ";
  COL << "iz"
      << " ";

  try {
    for (int iz = -1; iz <= 1; iz += 2) {
      COL << "========== " << (iz > 0 ? "EE+" : "EE-") << " ========== \n";
      for (int ix = 1; ix <= 100; ++ix) {
        for (int iy = 1; iy <= 100; ++iy) {
          if (!EEDetId::validDetId(ix, iy, iz))
            continue;

          //input
          COL << ix << " ";
          COL << iy << " ";
          COL << iz << " ";
          //detid
          EEDetId id(ix, iy, iz);
          COL << id.ix() << (ix != id.ix() ? "!!!" : "") << " ";
          COL << id.iy() << (iy != id.iy() ? "!!!" : "") << " ";
          COL << id.zside() << (iz != id.zside() ? "!!!" : "") << " ";
          COL << id.iquadrant() << " ";
          COL << (id.positiveZ() ? "z+" : "z-") << (id.positiveZ() != (iz > 0) ? "!!!" : "") << " ";
          COL << id.iPhiOuterRing() << " ";

          //hashed index
          int ih = id.hashedIndex();
          COL << ih << " ";
          if (!EEDetId::validHashIndex(ih) || EEDetId::unhashIndex(ih) != id ||
              EEDetId::unhashIndex(ih).rawId() != id.rawId()) {
            COL << "ERR!!!"
                << " ";
          } else {
            COL << "OK"
                << " ";
          }

          COL << id.denseIndex() << (id.denseIndex() != (uint32_t)ih ? "!!!" : "") << " ";

          //ISC
          const int isc = id.isc();
          COL << isc << " ";
          const int icInSc = id.ic();
          COL << icInSc << " ";
          EEDetId id1(isc, icInSc, iz, EEDetId::SCCRYSTALMODE);
          COL << id1.ix() << (id1.ix() != ix ? "!!!" : "") << " ";
          COL << id1.iy() << (id1.iy() != iy ? "!!!" : "") << " ";
          COL << id1.zside() << (id1.zside() != iz ? "!!!" : "") << " ";
          if (id != id1 || id.rawId() != id1.rawId()) {
            COL << "ERR!!!"
                << " ";
          } else {
            COL << "OK"
                << " ";
          }

          //SC det id
          COL << id.sc().ix() << " ";
          COL << id.sc().iy() << " ";
          COL << id.sc().zside() << " ";
          COL << "\n";
        }  //next iy
      }    //next ix
    }      //next iz
  } catch (exception& e) {
    cerr << e.what();
  }
}
