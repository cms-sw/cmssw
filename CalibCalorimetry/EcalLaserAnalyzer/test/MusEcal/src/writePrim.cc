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

  int c;
  while ( (c = getopt( argc, argv, "t:c:" ) ) != EOF ) 
    {
      switch (c) 
	{
	case 't': type     = atoi( optarg );    break;
	case 'c': color    = atoi(optarg);      break;
	}
    }

  cout << "Writing primitives for Type=" << ME::type[type];
  if( type==ME::iLaser ) cout << " Color=" << ME::color[color];
  cout << endl;

  // Barrel: Laser monitoring regions between 1 and 92
  for( unsigned int lmr=1; lmr<=92; lmr++ )
    {
      cout << "LMR=" << lmr << endl;
      MERunManager* runManager = new MERunManager( lmr, type, color );
      delete runManager;
    }

  return(0);
}
