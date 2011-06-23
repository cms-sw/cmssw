{
gROOT->Reset();
#include "Riostream.h"

ifstream in;

in.open("diffGainsOct_Aug109889.dat");

Int_t index;
Float_t diffGains;
int i=0;
Int_t nlines = 0;
TFile *f = new TFile("diffGainsOct_Aug109889.root","RECREATE");

TNtuple *ntuple = new TNtuple("DiffGains","data from new ascii file","index:diffGains");

while (1) {
  i++;
  in >> index >> diffGains ;
  if (!in.good()) break;

  DiffGains->Fill(index,diffGains);
  nlines++;
}

std::cout<<" found nr of lines: "<<nlines<<std::endl;

in.close();
f->Write();
}
