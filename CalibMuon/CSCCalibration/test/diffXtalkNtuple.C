{
gROOT->Reset();
#include "Riostream.h"

ifstream in;

in.open("diffXtalkOct_Aug3.dat");

Int_t index;
Float_t diffXtalkR;
Float_t diffXtalkL;
Float_t diffIntL;
Float_t diffIntR;
int i=0;
Int_t nlines = 0;
TFile *f = new TFile("diffXtalkOct_109890.dat.root","RECREATE");

TNtuple *ntuple = new TNtuple("DiffXtalk","data from new ascii file","index:diffXtalkR:diffIntR:diffXtalkL:diffIntL");

while (1) {
  i++;
  in >> index >> diffXtalkR >> diffIntR >> diffXtalkL >> diffIntL ;
  if (!in.good()) break;

  DiffXtalk->Fill(index,diffXtalkR,diffIntR,diffXtalkL,diffIntL);
  nlines++;
}

std::cout<<" found nr of lines: "<<nlines<<std::endl;

in.close();
f->Write();
}
