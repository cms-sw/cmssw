{
gROOT->Reset();
#include "Riostream.h"

ifstream in;

in.open("goodXtalk.dat");

Int_t index;
Float_t leftSlope, leftIntercept, rightSlope, rightIntercept;
int i=0;
Int_t nlines = 0;
TFile *f = new TFile("xtalk.root","RECREATE");

TNtuple *ntuple = new TNtuple("Xtalk","data from ascii file","index:leftSlope:leftIntercept:rightSlope:rightIntercept");

while (1) {
  i++;
  in >> index >> leftSlope >> leftIntercept >> rightSlope >> rightIntercept;

  if (!in.good()) break;

  Xtalk->Fill(index,leftSlope,leftIntercept,rightSlope,rightIntercept);
  nlines++;
}

std::cout<<" found nr of lines: "<<nlines<<std::endl;

in.close();
f->Write();
}
