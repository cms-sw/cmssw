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
#include "../../interface/MEChannel.h"

int main(int argc, char **argv)
{
  int ireg  = ME::iEBP;
  int isect = 5;
  int ichan = -1;
  int ieta  = -1;
  int iphi  = -1;

  int c;
  while ( (c = getopt( argc, argv, "r:s:c:e:p:" ) ) != EOF ) 
    {
      switch (c) 
	{
	case 'r': ireg     = atoi( optarg );    break;
	case 's': isect    = atoi( optarg );    break;
	case 'c': ichan    = atoi( optarg );    break;
	case 'e': ieta     = atoi( optarg );    break;
	case 'p': iphi     = atoi( optarg );    break;
	}
    }
// 1597         EB-5
// 1003         EB-6
// 421           EB-8
// 1239         EB-10
// 320           EB-16
// 1058         EB+10
// 1442         EB+16
// 682           EB+17

  if( ichan>=0 )
    {
      if( ireg==ME::iEBM ) isect+=18;
      if( ireg==ME::iEEM ) isect+=9;
      TString str_ = 
	ME::regTree( ireg )->getDescendant( ME::iSector, isect )
	->getDescendant( ME::iCrystal, ichan )->oneLine();
      cout << str_ << endl;
      return 0;
    }
  assert( iphi>0 );
  assert( ieta!=0 && std::abs(ieta)<=85 );
  if( ieta>0 )      ireg=ME::iEBP;
  else if( ieta<0 ) ireg=ME::iEBM;
  vector< MEChannel* > vec;
  ME::regTree( ireg )->getListOfChannels( vec );
  for( unsigned int ii=0; ii<vec.size(); ii++ )
    {
      MEChannel* leaf_ = vec[ii];
      if( leaf_->ix()!=ieta ) continue;
      if( leaf_->iy()!=iphi ) continue;
      cout << leaf_->oneLine() << endl;
      return(0);
    }

  return(0);
}
