#include <cassert>
#include <iostream>
#include <iomanip>
#include <fstream>
using namespace std;

#include "CalibCalorimetry/EcalLaserAnalyzer/interface/TPNCor.h"


// ClassImp(TPNCor)

TPNCor::TPNCor( string filename )
{


  // Initialize

  isFileOK=0;

  for(int i=0;i<2;i++){
    for(int k=0;k<10;k++){
      for(int j=0;j<3;j++){
	corParams[i][k][j] = 0.0  ;
      }
    }
  }
  
  // Get values from file
  
  FILE *test;
  test = fopen(filename.c_str(),"r");
  char c;
  int gain=0;
  double aa, bb, cc;
  
  int kk=0;
  if( test ) {
    fclose( test );
    ifstream fin(filename.c_str(),ifstream::in);
   
    while( (c=fin.peek()) != EOF )
      {
	fin >> aa >> bb >> cc;
	
	if(gain<2){
	  corParams[gain][kk][0]=aa;
	  corParams[gain][kk][1]=bb;
	  corParams[gain][kk][2]=cc;
	}

	kk++;
	if(kk==10){
	  gain=1;
	  kk=0;
	}
      }
    isFileOK=1;
    fin.close();
  }else {
    cout <<" No PN linearity corrections file found, no correction will be applied "<< endl;    
  }

 
}

TPNCor::~TPNCor() 
{
 
}

double
TPNCor::getPNCorrectionFactor( double val0 , int gain )
{
  double cor=0.0;  
  double corr=1.0;

  double pn=val0;
  double xpn=val0/1000.0;

  if( isFileOK==0) return 1.0;
  assert(  gain<2 );
  
  cor=xpn*(corParams[gain][0][0] +xpn*(corParams[gain][0][1]+xpn*corParams[gain][0][2]));
  
  if(pn!=0) corr = 1.0 - cor/pn;
  else corr=1.0;

  
  return corr;
  
  
}
