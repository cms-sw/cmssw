#include <iostream>
#include <fstream>
#include <cmath>
#include <fcntl.h>

#if !defined(__CINT__)
#include <TH2.h>
#include <TF1.h>
#include <TNtuple.h>
#endif
 

using namespace std;

//#define PRINT 

TNtuple *  ntuple;  // to see outside the macro.

int make_ntuple_lin() {   // run with .x
//int main() {   // run with .L & main()

// INput file 
  ifstream in_file;  // data file pointer
  char filename[80] = "phCalibrationFit_C0.dat";
  
  in_file.open(filename, ios::in ); // in C++
  if (in_file.bad()) {
    cout << " File not found " << endl;
    return(1); // signal error
  }
  cout << " file opened : " << filename << endl;

  char line[500];
  for (int i = 0; i < 3; i++) {
    in_file.getline(line, 500,'\n');
    cout<<line<<endl;
  }
  // Create NTuple 
  const int nvar=5;
  char *chvar = "p0:p1:roc:col:row";
  tfile = new TFile ( "phCalibrationFit_C0.root" , "RECREATE" );
  ntuple = new TNtuple("ntuple","NTUPLE",chvar);
  float p[nvar];
  

 const int max_inputs = 10000;
 float par0,par1,par2,par3;
 int rocid,colid,rowid;
 string name;
 rocid=0;

 // Read MC tracks
 for(int i=0;i<(52*80);i++)  { // loop over pixels
 //for(int i=0;i<10;i++)  { // loop over pixels
   
   //in_file >> par0 >> par1 >> par2 >> par3 >> name >> colid 
   //   >> rowid;
   in_file >> par0 >> par1 >> name >> colid >> rowid;
   if (in_file.bad()) { // check for errors
     cerr << "Cannot read data file" << endl;
     return(1);
   }
   if ( in_file.eof() != 0 ) {
     cerr << in_file.eof() << " " << in_file.gcount() << " " 
	  << in_file.fail() << " " << in_file.good() << " end of file " 
	  << endl;
     break;;
   }
   
   //cout <<" line "<<i<<" "<<par0<<" "<<par1<<" "<<colid<<" "<<rowid<<endl;
   //cout <<i<<endl;
   
   p[0]=par0;
   p[1]=par1;
   //p[2]=par2;
   //p[3]=par3;
   p[2]=float(rocid);
   p[3]=float(colid);
   p[4]=float(rowid);
   ntuple->Fill(p); 

 }   

/* Visualization */
 //ntuple->Draw("adc1:delay"); 

in_file.close();
cout << " Close input file " << endl;

tfile->Write();
tfile->Close();

return(0);
}




