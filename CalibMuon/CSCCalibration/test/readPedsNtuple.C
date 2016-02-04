{
gROOT->Reset();
#include "Riostream.h"

ifstream in1;
ifstream in2;

in1.open("pedSummary2008_07_29.dat");
in2.open("pedSummary2008_09_02.dat");

Int_t index1, index2;
Float_t peds1, rms1, peds2, rms2;
int i=0;
int j=0;
Int_t nlines1 = 0;
Int_t nlines2 = 0;

TFile *f = new TFile("peds1.root","RECREATE");

//TNtuple *ntuple1 = new TNtuple("Peds1","data from ascii file","index1:peds1:rms1:index2:peds2:rms2");
TNtuple *ntuple1 = new TNtuple("Peds2","data from new ascii file","index1:peds1:rms1");

while (1) {
  i++;
  in2 >> index1 >> peds1 >> rms1 ;
  if (!in2.good()) break;
  Peds2->Fill(index1,peds1,rms1);
  nlines1++;
  /*
  while (2) {
    j++;
    in2 >> index2 >> peds2 >> rms2 ;
    if (!in2.good()) break;
    Peds1->Fill(index1,peds1,rms1,index2,peds2,rms2);
    nlines2++;
  }
  */
}

/*
while (1) {
  j++; 
  in2 >> index2 >> peds2 >> rms2 ;
  if (!in2.good()) break;
  Peds1->Fill(index1,peds1,rms1,index2,peds2,rms2);
  nlines2++;
}
*/
std::cout<<" found nr of lines: "<<nlines1<<"  "<<nlines2<<std::endl;
in1.close();
in2.close();
f->Write();
}
