// Convert calibration data (10 vcal points) to a ntuple
// Reading ascii files in CINT can be slow.
// Try to precompile.
// .x make_ntuple_data.C+
//

#include <iostream>
#include <fstream>
#include <cmath>
#include <fcntl.h>

#if !defined(__CINT__)
#include <TH2.h>
#include <TF1.h>
#include <TNtuple.h>
#endif
  
//#define PRINT 

TNtuple *  ntuple;  // to see outside the macro.

int make_ntuple_data() {   // run with .x
//int main() {   // run with .L & main()

// INput file 
  ifstream in_file;  // data file pointer
  char filename[80] = "phCalibration_C0.dat";  
  in_file.open(filename, ios::in ); // in C++

  cout << in_file.eof() << " " << in_file.bad() << " " 
       << in_file.fail() << " " << in_file.good()<<endl; 
  
  if (in_file.fail()) {
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
  const int nvar=13;
  char *chvar = "d0:d1:d2:d3:d4:d5:d6:d7:d8:d9:roc:col:row";
  tfile = new TFile ( "phCalibration_C0.root" , "RECREATE" );
  ntuple = new TNtuple("ntuple","NTUPLE",chvar);
  float p[nvar];
  
  float count[10] = {0.,0.,0.,0.,0.,0.,0.,0.,0.,0.};
  float sum[10] = {0.,0.,0.,0.,0.,0.,0.,0.,0.,0.};
  float par[10];
  int rocid=0,colid,rowid;
  string name;
  string str[10];
  //char[100] str;

  // Read data for all pixels
  for(int i=0;i<(52*80);i++)  { // loop over pixels
   //for(int i=0;i<10;i++)  { // loop over pixels
   

   //for(int i0=0;i0<10;i0++) fscanf(in_file, "%s", str);
     
   //in_file >> par[0] >> par[1] >> par[2] >> par[3] >> par[4]
   // >> par[5] >> par[6] >> par[7] >> par[8] >> par[9]
   in_file >> str[0] >> str[1] >> str[2] >> str[3] >> str[4]
	   >> str[5] >> str[6] >> str[7] >> str[8] >> str[9]
	   >> name >> colid >> rowid;

   if (in_file.fail()) { // check for errors
     cout << "Cannot read data file" << endl;
     return(1);
   }
   if ( in_file.eof() != 0 ) {
     cout<< " end of file " << endl;
     break;;
   }
   
#ifdef PRINT
   cout << " line " << i <<" ";
   for(int i1 =0;i1<10;i1++) { cout<<str[i1]<<" "; }
   cout<<colid<<" "<<rowid<<endl;
#endif
   //cout <<i<<" "<<colid<<" "<<rowid<<endl;
   
   for(int i2 =0;i2<10;i2++) { 
     if( str[i2] == "N/A" ) {
       //cout<<" skip N/A"<<endl;
       p[i2]=-9999.;
     } else {
       p[i2]=atof(str[i2].c_str());
       count[i2]++;
       sum[i2] += p[i2];
     }
   }
   
   p[10]=float(rocid);
   p[11]=float(colid);
   p[12]=float(rowid);
   ntuple->Fill(p); 
   
   //cout << " line " << i <<" ";
   //for(int i1 =0;i1<10;i1++) { cout<<p[i1]<<" "; }
   //cout<<colid<<" "<<rowid<<endl;
   
 }   
 
 /* Visualization */
 //ntuple->Draw("adc1:delay"); 
 
 in_file.close();
 cout << " Close input file " << endl;
 tfile->Write();
 tfile->Close();

 for(int i2 =0;i2<10;i2++) { 
   par[i2] = sum[i2]/count[i2];
   cout<<par[i2]<<" ";
 }
 cout<<endl;
 
 return(0);
}




