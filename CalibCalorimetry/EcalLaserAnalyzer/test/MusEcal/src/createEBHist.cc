#include <cassert>
#include <iostream>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <stdio.h>
#include <fstream>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <vector>
#include <map>
using namespace std;

// simple test program
#include "../../interface/ME.h"
#include "../../interface/MEGeom.h"
#include "MusEcal.hh"
#include "MERunManager.hh"

int main(int argc, char **argv)
{
  int type  = ME::iLaser;
  int ivar  = ME::iAPD_MEAN;
  int color = ME::iBlue; 

  int c;
  while ( (c = getopt( argc, argv, "v:t:w:" ) ) != EOF ) 
    {
      switch (c) 
	{
	case 'v': ivar     = atoi( optarg );    break;
	case 't': type     = atoi( optarg );    break;
	case 'w': color    = atoi( optarg );    break;
	}
    }
  MusEcal::verbose = false;
  MusEcal* me;
  me = new MusEcal( type, color );

  TFile* outfile_ = TFile::Open( "outfile.root", "RECREATE" );
  int ii(0);
  me->histConfig();
  me->bookHistograms();
  do
    {
      cout << "\nSequence=" << ii << " starting ";
      time_t t_ =  me->curMgr()->curKey();
      cout << ctime(&t_) << endl;
      me->fillHistograms();
      outfile_->cd();
      me->writeGlobalHistograms();	
      ii++;
    }
  while( me->nextSequence() );
  cout << "number of sequences " << ii << endl;
  outfile_->Close();

  return(0);
}
