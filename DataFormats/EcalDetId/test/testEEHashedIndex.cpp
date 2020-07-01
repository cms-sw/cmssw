/**
  * \file
  * A test for EEDetId::hashedIndex()
  *
  */

#include <cassert>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>

#include "DataFormats/EcalDetId/interface/EEDetId.h"

const int nBegin[EEDetId::IX_MAX] = {41, 41, 41, 36, 36, 26, 26, 26, 21, 21, 21, 21, 21, 16, 16, 14, 14, 14, 14, 14,
                                     9,  9,  9,  9,  9,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  4,  4,  4,  4,  4,
                                     1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
                                     4,  4,  4,  4,  4,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  9,  9,  9,  9,  9,
                                     14, 14, 14, 14, 14, 16, 16, 21, 21, 21, 21, 21, 26, 26, 26, 36, 36, 41, 41, 41};

int main(int argc, char *argv[]) {
  FILE *ofile = fopen("ee_numbering.dat", "w");
  FILE *ofile_2 = fopen("ee_next_to_boundary.dat", "w");
  int hi = -1;
  try {
    for (int iz = -1; iz < 2; iz += 2) {
      for (int ix = EEDetId::IX_MIN; ix <= EEDetId::IX_MAX; ix++) {
        for (int iy = nBegin[ix - 1]; iy <= 100 - nBegin[ix - 1] + 1; iy++) {
          hi = -1;
          if (EEDetId::validDetId(ix, iy, iz)) {
            EEDetId id = EEDetId(ix, iy, iz);
            hi = id.hashedIndex();
            assert(EEDetId::unhashIndex(hi) == id);
            //std::cout << id << " " << hi << " " << EEDetId::unhashIndex( hi ) << std::endl;
            fprintf(ofile, "%d %d %d %d %d\n", ix, iy, iz, hi, 1);
            if (EEDetId::isNextToBoundary(id)) {
              fprintf(ofile_2, "%d %d %d %d %d\n", ix, iy, iz, hi, 1);
            } else {
              fprintf(ofile_2, "%d %d %d %d %d\n", ix, iy, iz, hi, 0);
            }
          } else {
            fprintf(ofile, "%d %d %d %d %d\n", ix, iy, iz, hi, 0);
            //std::cout << "Invalid detId " << ix << " " << iy << " " << iz << std::endl;
          }
        }
      }
    }
    for (int i = 0; i < 15480; i++) {
      EEDetId id = EEDetId::unhashIndex(hi);
      assert(EEDetId::validDetId(id.ix(), id.iy(), id.zside()));
    }
  } catch (std::exception &e) {
    std::cerr << e.what();
  }
  fclose(ofile);
  fclose(ofile_2);
}

// to plot the output file:
//gnuplot> set terminal postscript enhanced eps color colourtext dl 1.0 lw 1.5 "Helvetica" 21
//gnuplot> set out 'ee_numbering.eps'
//gnuplot> set size square
//gnuplot> set xlabel 'ix'
//gnuplot> set ylabel 'iy'
//gnuplot> p [0:101][0:101] 'ee_numbering.dat' u 1:($5>0 ? $2 : 1/0) not, '' u 1:($5==0 ? $2 : 1/0) not
