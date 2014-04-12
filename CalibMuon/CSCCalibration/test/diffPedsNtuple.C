{
gROOT->Reset();
#include "Riostream.h"

ifstream in;

in.open("diff_Peds_FileName");

Int_t index;
Float_t diffPeds;
int i=0;
Int_t nlines = 0;
TFile *f = new TFile("RootFile","RECREATE");

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


