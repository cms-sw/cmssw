{
gROOT->Reset();
#include "Riostream.h"

ifstream in;

in.open("diff_Gains_FileName");

Int_t index;
Float_t diffGains;
int i=0;
Int_t nlines = 0;
TFile *f = new TFile("RootFile","RECREATE");

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



