// Magnetic Field Interpolation Test Program (formerly prepareFieldInterpolation.cpp)
// by droll (03/02/04) 
// Updated N. Amapane 3/07 -former name 

#include <iostream>

#include "prepareMagneticFieldGrid.h"

using namespace std;

int main(int argc, char **argv) {
  if (argc > 4) {
    cout << "SYNOPSIS:" << endl
	 << " prepareFieldTable input.table output.bin [sector]" << endl;
    cout << "Example:" << endl
	 << " prepareFieldTable /afs/cern.ch/cms/OO/mag_field/version_85l_030919/v-xyz-217.table grid.217.bin" << endl;
      return 1;  
  }

  string filename1 = argv[1];
  string filename2 = argv[2];

  int sector=0;
  if (argc==4) {  
    sector = atoi(argv[3]);
  }
  

  prepareMagneticFieldGrid MFG001(sector);                                    // MFG001 for standard cases
  MFG001.countTrueNumberOfPoints(filename1);   // check, if file contains some points twice


  
  MFG001.fillFromFile(filename1);              // determine grid structure (standard cases)
  int type           =   MFG001.gridType();                           // grid type
  if (type == 0) {
    cout << "  standard grid sructure detection failed, retrying with special grid sructure" << endl;
  } else {
    if (MFG001.isReady())  MFG001.validateAllPoints();                  // check position of every point
    if (MFG001.isReady())  {
      MFG001.saveGridToFile(filename2);            // write grid to disk
      cout << " " << endl;
      return 0;
    }
  }

  // MFG001 anlysis was not successful. Different processing for special cases
  prepareMagneticFieldGrid MFG002(sector);                                  // MFG002 for special cases
  MFG002.fillFromFileSpecial(filename1);     // determine grid structure (special cases)
  type = MFG002.gridType();                         // grid type
  if (type == 0) cout << "  special grid sructure detection failed " << endl;
  if (MFG002.isReady())  MFG002.validateAllPoints();                // check position of every point
  if (MFG002.isReady())  {
    MFG002.saveGridToFile(filename2);          // write grid to disk
    cout << " " << endl;
    return 0;
  } 

  return 1;
}
