#include <iostream>
#include <math.h>
#include <fstream>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <time.h>
#include <unistd.h>
#include <stdio.h>
#include <vector>
#include <map>
using namespace std;

#include "MELaserPrim.cc"  // fixme ---> put file in stubs/

// Creating laser monitoring primitives

int main(int argc, char **argv)
{
  int dcc      = 628;
  int color    = MELaserPrim::iIRed;
  TString inpath(  "/media/data/malcles/MonitoringResults/LMEB06_PedOK/Laser/Analyzed/" ); // on pcsaccms03
  TString outpath( "/afs/cern.ch/user/g/ghm/scratch0/MusEcal/EBSM06/prim/");  // on pcsaccms03
  int c;
  while ( (c = getopt( argc, argv, "r:s:d:c:i:o:" ) ) != EOF ) 
    {
      switch (c) 
	{
	case 'd': dcc      = atoi(optarg);      break;
	case 'c': color    = atoi(optarg);      break;
	  //	case 'i': inpath   = TString( optarg ); break;
	  //	case 'o': outpath  = TString( optarg ); break;
	}
    }


  TString runlistfile(outpath); runlistfile += "runlist_";
  switch( color ) 
    {
    case MELaserPrim::iBlue:  runlistfile+="Blue_"; break;
    case MELaserPrim::iRed:   runlistfile+="Red_";  break;
    case MELaserPrim::iIRed:  runlistfile+="Red_";  break;  // fixme
    default: abort();
    }
  runlistfile += "Laser";
  cout << "read " << runlistfile << endl;
  ifstream fin;
  fin.open(runlistfile);

  while( fin.peek() != EOF )
    {
      int run_, ts_;
      int dt;
      fin >> run_ >> ts_ >> dt;
      cout << "Run=" << run_ << " TS=" << ts_ << endl;
      //      if( run_!=17158 ) continue;

      TString inpath_(inpath); inpath_ += "h4b.000"; inpath_ += run_; inpath_ += ".A.0.0/";
      for( int side_=0; side_<2; side_++ )
	{
	  cout << "Make Primitives for DCC=" << dcc << " Side=" << side_ << " Color=" << color 
	       << " Run=" << run_  << " TS=" << ts_ << endl;
	  MELaserPrim prim( dcc, side_, color, run_, ts_, inpath_, outpath );
	  prim.init();
	  prim.bookHistograms();
	  prim.fillHistograms();
	  prim.writeHistograms();
	}
    }
      

  return(0);
}
