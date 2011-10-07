#include <cstdio>
#include <cstdlib>
#include <iostream>
using namespace std;

// writing Laser primitives

#include "../../interface/ME.h"
#include "MERunManager.hh"

int main(int argc, char **argv)
{
  int type  = ME::iLaser;
  int color = ME::iBlue; 
  int ecalpart = -1; 

  int c;
  while ( (c = getopt( argc, argv, "t:c:e:" ) ) != EOF ) 
    {
      switch (c) 
	{
	case 't': type     = atoi( optarg );    break;
	case 'c': color    = atoi(optarg);      break;
	case 'e': ecalpart = atoi(optarg);      break;
	}
    }

  
  string FirstStr(getenv("PRIM_FIRST_RUN"));
  string LastStr(getenv("PRIM_LAST_RUN"));
  int First=atoi(FirstStr.c_str());
  int Last=atoi(LastStr.c_str());
  cout<<" Strings: "<<FirstStr<<" "<< LastStr<< endl;
  cout<<" Ints: "<<First<<" "<< Last<< endl;

  if(First<170000 || First>300000) First=170000;
  if(Last<170000 ) Last=900000;
  cout<<" Ints Checked: "<<First<<" "<< Last<< endl;
  
  
  MusEcal::firstRun=First;
  MusEcal::lastRun=Last;

  cout<< " First run :"<< First <<" last run:"<< Last<< endl; 

  //  cout<<" TESTJULIE: " << type<<" "<< ME::type[type]<<" "<< color<<" "<<  ME::color[color]<< endl;

  cout << "Writing primitives for Type=" << ME::type[type];
  if( type==ME::iLaser || type==ME::iLED ) cout << " Color=" << ME::color[color];
  cout << endl;

  if( type==ME::iLaser && (color!=ME::iBlue && color!= ME::iIRed)){
    cout<< "Wrong Color "<<ME::color[color]<<" for type "<<ME::type[type]<< endl;
  }

  if( type==ME::iLED && (color!=ME::iBlue && color!= ME::iRed)){
    cout<< "Wrong Color "<<ME::color[color]<<" for type "<<ME::type[type]<< endl;
  }

  
  // Barrel: Laser monitoring regions between 1 and 72
  // Endcap: 73 to 92

  int firstlmr=1;
  int lastlmr=92;

  if(type==ME::iLED ) firstlmr=73;

  if(ecalpart==0){
    // EB first part
    firstlmr=1;
    lastlmr=36;
  }else if(ecalpart==1){
    //EB second part
    firstlmr=37;
    lastlmr=72;
  }else if(ecalpart==2){
    //EE
    firstlmr=73;
    lastlmr=92;
  }
  
  
  for( unsigned int lmr=firstlmr; lmr<=lastlmr; lmr++ )
    {
      cout << "LMR=" << lmr << endl;
      MERunManager* runManager = new MERunManager( lmr, type, color );
      delete runManager;
    }
  
  return(0);
}
