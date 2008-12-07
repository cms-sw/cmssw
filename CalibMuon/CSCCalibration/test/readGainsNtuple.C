{
gROOT->Reset();
#include "Riostream.h"

ifstream in;

in.open("goodGains.dat");

Int_t index;
Float_t gainSlope, gainIntercept, chisq;
int i=0;
Int_t nlines = 0;
TFile *f = new TFile("gains.root","RECREATE");

TNtuple *ntuple = new TNtuple("Gains","data from ascii file","index:gainSlope:gainIntercept:chisq");

while (1) {
  i++;
  in >> index >> gainSlope >> gainIntercept >> chisq;

  if (!in.good()) break;

  Gains->Fill(index,gainSlope,gainIntercept,chisq);
  nlines++;
}

std::cout<<" found nr of lines: "<<nlines<<std::endl;

in.close();
f->Write();
}
