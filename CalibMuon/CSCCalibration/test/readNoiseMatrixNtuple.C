{
gROOT->Reset();
#include "Riostream.h"

ifstream in;

in.open("goodMatrix2008_09_02.dat");

Int_t index;
Float_t elem33,elem34,elem44,elem35,elem45,elem55,elem46,elem56,elem66,elem57,elem67,elem77;
int i=0;
Int_t nlines = 0;
TFile *f = new TFile("matrix.root","RECREATE");

TNtuple *ntuple = new TNtuple("Matrix","data from ascii file","index:elem33:elem34:elem44:elem35:elem45:elem55:elem46:elem56:elem66:elem57:elem67:elem77");

while (1) {
  i++;
  in >> index >> elem33>>elem34>>elem44>>elem35>>elem45>>elem55>>elem46>>elem56>>elem66>>elem57>>elem67>>elem77;

  if (!in.good()) break;

  Matrix->Fill(index,elem33,elem34,elem44,elem35,elem45,elem55,elem46,elem56,elem66,elem57,elem67,elem77);
  nlines++;
}

std::cout<<" found nr of lines: "<<nlines<<std::endl;

in.close();
f->Write();
}
