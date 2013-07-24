/* 
 *  \class TPNCor
 *
 *  $Date: 2012/02/09 10:08:10 $
 *  \author: Julie Malcles - CEA/Saclay
 */

#include <CalibCalorimetry/EcalLaserAnalyzer/interface/TPNCor.h>
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <sstream>
#include <iostream>
#include <iomanip>
#include <fstream>

#include "math.h"

using namespace std;
//using namespace edm;

//ClassImp(TPNCor)

// Constructor...
TPNCor::TPNCor(string filename)
{

  // Initialize

  isFileOK=0;

  for(int i=0;i<iSizePar;i++){
    for(int j=0;j<iSizeGain;j++){
      corParams[j][i] = 0.0   ;
    }
  }
  
  // Get values from file
  
  FILE *test;
  test = fopen(filename.c_str(),"r");
  char c;
  int gain;
  double aa, bb, cc;
  ifstream fin;
  
  if( test ) {
    fclose( test );
    fin.open(filename.c_str());
    while( (c=fin.peek()) != EOF )
      {
	fin >> gain>> aa >> bb >> cc;
	
	if(gain<iSizeGain){
	  corParams[gain][0]=aa;
	  corParams[gain][1]=bb;
	  corParams[gain][2]=cc;
	}
      }
    isFileOK=1;
    fin.close();
  }else {
    cout <<" No PN linearity corrections file found, no correction will be applied "<< endl;
  }

}

// Destructor
TPNCor::~TPNCor()
{ 

  
}

double TPNCor::getPNCorrectionFactor( double val0 , int gain )
{
  
  double cor=0;
  double corr=1.0;
  double pn=val0;
  double xpn=val0/1000.0;

  if( isFileOK==0) return 1.0;

  if( gain> iSizeGain ) cout << "Unknown gain, gain has to be lower than "<<iSizeGain << endl;
  
  if( gain< iSizeGain ) {
    
    cor=xpn*(corParams[gain][0] +xpn*(corParams[gain][1]+xpn*corParams[gain][2]));
    
    if(pn!=0) corr = 1.0 - cor/pn;
    else corr=1.0;
  } 
  
  return corr;
}
