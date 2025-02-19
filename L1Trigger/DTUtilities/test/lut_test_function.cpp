/*
Main program to test function dump LUTs, called IEEE32toDSP
Author: A. Gozzelino
	INFN LNL & Padova University
Date: May 11th 2012
*/

//---------------
// C++ Headers --
//---------------
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cstring>
#include <math.h>   
#include <stdio.h>

using namespace std;

// Function under test
void IEEE32toDSP(float f, short int & DSPmantissa, short int & DSPexp);

//  Main program
int main()
{

  float ST    = 30.;
  float pitch = 4.2;
  float h     = 1.3;
  float DD    = 18.;

  for(int i=-511;i<512;i++)  {
    float fpsi =  atan( ((float)(i) * pitch) /(DD * h * ST ) );
    unsigned short int ipsi = int(fpsi*512);
    unsigned short int ipsi_9bits = ipsi & 0x1FF;
    //cout << dec << setw(3) << i << setw(10) << fpsi << hex <<
	    //setw(10) << ipsi << setw(10) << ipsi_9bits << endl;
  }

  // convert parameters from IEE32 float to DSP float format
  short int DSPmantissa = 0;
  short int DSPexp = 0;
  float d = 34.;
  short int btic = 31;
  cout << "CHECK BTIC " << btic << endl;
  short int Low_byte = (btic & 0x00FF);   // output in hex bytes format with zero padding
  short int High_byte =( btic>>8 & 0x00FF);
  
  cout << setw(2) << setfill('0') << hex << High_byte << setw(2) << setfill('0')
    << Low_byte << endl;	

  // d parameter conversion and dump
  IEEE32toDSP(d, DSPmantissa, DSPexp);

  Low_byte = (DSPmantissa & 0x00FF);   // output in hex bytes format with zero padding
  High_byte =( DSPmantissa>>8 & 0x00FF);
  cout << setw(2) << setfill('0') << hex << High_byte << setw(2) << setfill('0') << Low_byte << endl;	
  Low_byte = (DSPexp & 0x00FF);
  High_byte =( DSPexp>>8 & 0x00FF);
  cout << setw(2) << setfill('0') << High_byte << setw(2) << setfill('0') << Low_byte << endl;	

  return 0;
}

// Function under test: correct definition 
// Function used during test beam phase. Today it is not called anymore. Keep it as reference for test.

void IEEE32toDSP(float f, short int & DSPmantissa, short int & DSPexp)
{
  long int pl = 0;
  long int lm;

  bool sign=false;

  DSPmantissa = 0;
  DSPexp = 0;

  if( f!=0.0 )
  {
        memcpy (&pl,&f,sizeof(float)); 
        if((pl & 0x80000000)!=0) 
		sign=true;	  
        lm = ( 0x800000 | (pl & 0x7FFFFF)); // [1][23bit mantissa]
        lm >>= 9; //reduce to 15bits
	  lm &= 0x7FFF;
        DSPexp = ((pl>>23)&0xFF)-126;
	  DSPmantissa = (short)lm;
	  if(sign) 
		DSPmantissa = - DSPmantissa;  // convert negative value in 2.s complement	
        
	/*
	//*********************************
	// Old and wrong definition 
	//*********************************
	//long int *pl=0, lm;
  	bool sign=false;

        memcpy(pl,&f,sizeof(float));
        if((*pl & 0x80000000)!=0) 
		sign=true;	  
        lm = ( 0x800000 | (*pl & 0x7FFFFF)); // [1][23bit mantissa]
        lm >>= 9; //reduce to 15bits
	  lm &= 0x7FFF;
        DSPexp = ((*pl>>23)&0xFF)-126;
	  DSPmantissa = (short)lm;
	  if(sign) 
		DSPmantissa = - DSPmantissa;  // convert negative value in 2.s complement
	//***********************************************	
        */   
	 
  }
  return;
}

