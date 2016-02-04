// Small Program to read Grid Files
// by droll (29/02/04)
// essential files
#include "MagneticField/Interpolation/src/binary_ifstream.h"

// used libs
#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>

using namespace std;

int main(int argc, char **argv)
{
  if (argc > 3) {
    cout << "SYNOPSIS:" << endl
	 << " GridFileReader input.bin [fullDump=true|false]" << endl;
    cout << "Example:" << endl
	 << " GridFileReader grid.217.bin false" << endl;
      return 1;  
  }
  const string filename = argv[1];

  bool fullDump = argv[2];

  binary_ifstream inFile(filename);
  if (!inFile) {
    cout << "file open failed!" << endl;
    return false;
  }

  cout << "Data File: " << filename << endl;

  // reading iterator (number of entries)
  // declaration of all variable needed for file reading
  int type;
  int nPnt[3];
  double ooPnt[3];
  double dist0[3];
  double dist1[3][3];
  double dist2[3][3];
  double rParm[4];
  bool   easyC[3];
  float Bx, By, Bz;
  // reading the type
  inFile >> type;
  // reading the header
  if (type == 1){
    inFile >>  nPnt[0]    >>  nPnt[1]    >>  nPnt[2];
    inFile >> ooPnt[0]    >> ooPnt[1]    >> ooPnt[2];
    inFile >> dist0[0]    >> dist0[1]    >> dist0[2];
  }
  if (type == 2){
    inFile >> nPnt[0]     >> nPnt[1]     >> nPnt[2];
    inFile >> ooPnt[0]    >> ooPnt[1]    >> ooPnt[2];
    inFile >> dist0[0]    >> dist0[1]    >> dist0[2];
    inFile >> dist1[0][0] >> dist1[1][0] >> dist1[2][0];
    inFile >> dist1[0][1] >> dist1[1][1] >> dist1[2][1];
    inFile >> dist1[0][2] >> dist1[1][2] >> dist1[2][2];
    inFile >> dist2[0][0] >> dist2[1][0] >> dist2[2][0];
    inFile >> dist2[0][1] >> dist2[1][1] >> dist2[2][1];
    inFile >> dist2[0][2] >> dist2[1][2] >> dist2[2][2];
    inFile >> easyC[0]    >> easyC[1]    >> easyC[2];
  }
  if (type == 3){
    inFile >>  nPnt[0]    >>  nPnt[1]    >>  nPnt[2];
    inFile >> ooPnt[0]    >> ooPnt[1]    >> ooPnt[2];
    inFile >> dist0[0]    >> dist0[1]    >> dist0[2];
  }
  if (type == 4){
    inFile >> nPnt[0]     >> nPnt[1]     >> nPnt[2];
    inFile >> ooPnt[0]    >> ooPnt[1]    >> ooPnt[2];
    inFile >> dist0[0]    >> dist0[1]    >> dist0[2];
    inFile >> dist1[0][0] >> dist1[1][0] >> dist1[2][0];
    inFile >> dist1[0][1] >> dist1[1][1] >> dist1[2][1];
    inFile >> dist1[0][2] >> dist1[1][2] >> dist1[2][2];
    inFile >> dist2[0][0] >> dist2[1][0] >> dist2[2][0];
    inFile >> dist2[0][1] >> dist2[1][1] >> dist2[2][1];
    inFile >> dist2[0][2] >> dist2[1][2] >> dist2[2][2];
    inFile >> easyC[0]    >> easyC[1]    >> easyC[2];
  }
  if (type == 5){
    inFile >>  nPnt[0]    >>  nPnt[1]    >>  nPnt[2];
    inFile >> ooPnt[0]    >> ooPnt[1]    >> ooPnt[2];
    inFile >> dist0[0]    >> dist0[1]    >> dist0[2];
    inFile >> rParm[0]    >> rParm[1]    >> rParm[2]    >> rParm[3];
  }

  //reading the field
  int nLines = nPnt[0]*nPnt[1]*nPnt[2];

  // print the stuff from above (onlt last line of field values)
  cout << "  content of " << filename << endl;
  cout << type << endl;
  if (type == 1){
    cout <<  nPnt[0] << " " <<  nPnt[1] << " " <<  nPnt[2] << endl;
    cout << ooPnt[0] << " " << ooPnt[1] << " " << ooPnt[2] << endl;
    cout << dist0[0] << " " << dist0[1] << " " << dist0[2] << endl;
  }
  if (type == 2){
    cout <<  nPnt[0]    << " " <<  nPnt[1]    << " " <<  nPnt[2]    << endl;
    cout << ooPnt[0]    << " " << ooPnt[1]    << " " << ooPnt[2]    << endl;
    cout << dist0[0]    << " " << dist0[1]    << " " << dist0[2]    << endl;
    cout << dist1[0][0] << " " << dist1[1][0] << " " << dist1[2][0] << endl;
    cout << dist1[0][1] << " " << dist1[1][1] << " " << dist1[2][1] << endl;
    cout << dist1[0][2] << " " << dist1[1][2] << " " << dist1[2][2] << endl;
    cout << dist2[0][0] << " " << dist2[1][0] << " " << dist2[2][0] << endl;
    cout << dist2[0][1] << " " << dist2[1][1] << " " << dist2[2][1] << endl;
    cout << dist2[0][2] << " " << dist2[1][2] << " " << dist2[2][2] << endl;
    cout << easyC[0]    << " " << easyC[1]    << " " << easyC[2]    << endl;
  }
  if (type == 3){
    cout <<  nPnt[0] << " " <<  nPnt[1] << " " <<  nPnt[2] << endl;
    cout << ooPnt[0] << " " << ooPnt[1] << " " << ooPnt[2] << endl;
    cout << dist0[0] << " " << dist0[1] << " " << dist0[2] << endl;
  }
  if (type == 4){
    cout <<  nPnt[0]    << " " <<  nPnt[1]    << " " <<  nPnt[2]    << endl;
    cout << ooPnt[0]    << " " << ooPnt[1]    << " " << ooPnt[2]    << endl;
    cout << dist0[0]    << " " << dist0[1]    << " " << dist0[2]    << endl;
    cout << dist1[0][0] << " " << dist1[1][0] << " " << dist1[2][0] << endl;
    cout << dist1[0][1] << " " << dist1[1][1] << " " << dist1[2][1] << endl;
    cout << dist1[0][2] << " " << dist1[1][2] << " " << dist1[2][2] << endl;
    cout << dist2[0][0] << " " << dist2[1][0] << " " << dist2[2][0] << endl;
    cout << dist2[0][1] << " " << dist2[1][1] << " " << dist2[2][1] << endl;
    cout << dist2[0][2] << " " << dist2[1][2] << " " << dist2[2][2] << endl;
    cout << easyC[0]    << " " << easyC[1]    << " " << easyC[2]    << endl;
  }
  if (type == 5){
    cout <<  nPnt[0] << " " <<  nPnt[1] << " " <<  nPnt[2] << endl;
    cout << ooPnt[0] << " " << ooPnt[1] << " " << ooPnt[2] << endl;
    cout << dist0[0] << " " << dist0[1] << " " << dist0[2] << endl;
    cout << rParm[0] << " " << rParm[1] << " " << rParm[2] << " " << rParm[3] << endl;
  }

  for (int iLine=0; iLine<nLines; ++iLine){
    inFile >> Bx >> By >> Bz;
    if (fullDump) {
      cout  << setprecision(12);
      cout  << "line: " << iLine  << " " << Bx << " " << By << " " << Bz << endl;
    }
  }
  
  if (!fullDump) {
    cout << ". . ." << endl;
    cout << Bx << " " << By << " " << Bz << " (last line of B-field only)" << endl;
  }
  

  // check completeness and close file
  string lastEntry;
  inFile >> lastEntry;
  inFile.close();

  cout << "  file is " << lastEntry;
  if (lastEntry == "complete") cout << "  -->  reading done" << endl;
  else                         cout << "  -->  reading ERROR" << endl;
  cout << endl;

  return(0);
}
