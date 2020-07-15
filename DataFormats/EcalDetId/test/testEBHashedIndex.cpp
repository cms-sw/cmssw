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

#include "DataFormats/EcalDetId/interface/EBDetId.h"

int main(int argc, char *argv[]) {
  FILE *ofile = fopen("eb_next_to_boundary.dat", "w");
  int hi = -1;
  try {
    for (int ieta = EBDetId::MIN_IETA; ieta <= EBDetId::MAX_IETA; ieta++) {
      for (int iphi = EBDetId::MIN_IPHI; iphi <= EBDetId::MAX_IPHI; iphi++) {
        if (EBDetId::validDetId(ieta, iphi)) {
          EBDetId id = EBDetId(ieta, iphi);
          if (EBDetId::isNextToBoundary(id)) {
            fprintf(ofile, "%d %d %d %d\n", id.ieta(), id.iphi(), hi, 1);
          } else {
            fprintf(ofile, "%d %d %d %d\n", id.ieta(), id.iphi(), hi, 0);
          }
        }
        if (EBDetId::validDetId(-ieta, iphi)) {
          EBDetId id = EBDetId(-ieta, iphi);
          if (EBDetId::isNextToBoundary(id)) {
            fprintf(ofile, "%d %d %d %d\n", id.ieta(), id.iphi(), hi, 1);
          } else {
            fprintf(ofile, "%d %d %d %d\n", id.ieta(), id.iphi(), hi, 0);
          }
        }
      }
    }
  } catch (std::exception &e) {
    std::cerr << e.what();
  }
  fclose(ofile);
}

// to plot the output file:
//gnuplot> set terminal postscript enhanced eps color colourtext dl 1.0 lw 1.5 "Helvetica" 21
//gnuplot> set out 'ee_numbering.eps'
//gnuplot> set size square
//gnuplot> set xlabel 'ix'
//gnuplot> set ylabel 'iy'
//gnuplot> p [0:101][0:101] 'ee_numbering.dat' u 1:($5>0 ? $2 : 1/0) not, '' u 1:($5==0 ? $2 : 1/0) not
