{
gROOT->Reset();
#include "Riostream.h"

ifstream in;

in.open("diffPedsOct_Feb.dat");

Int_t index;
Float_t diffPeds;
int i=0;
Int_t nlines = 0;
TFile *f = new TFile("diffPedsOct_Feb.root","RECREATE");

TNtuple *ntuple = new TNtuple("DiffPeds","data from new ascii file","index:diffPeds");

while (1) {
  i++;
  in >> index >> diffPeds ;
  if (!in.good()) break;

  DiffPeds->Fill(index,diffPeds);
  nlines++;
}

std::cout<<" found nr of lines: "<<nlines<<std::endl;

in.close();
f->Write();
}
