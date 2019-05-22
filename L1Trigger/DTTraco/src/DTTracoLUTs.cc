//-------------------------------------------------
//
//   Class: DTTracoLUTs
//
//   Description: Look-up tables for phi radial angle and
//   psi angle from angle and position in traco
//
//   Author :
//   Sara Vanini - 10/III/03 - INFN Padova
//   17/III/07 SV : delete SimpleConfigurable dependence
//--------------------------------------------------
// #include "Utilities/Configuration/interface/Architecture.h"

//-----------------------
// This Class's Header --
//-----------------------

#include "L1Trigger/DTTraco/interface/DTTracoLUTs.h"

//---------------
// C++ Headers --
//---------------
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "L1Trigger/DTUtilities/interface/DTTPGLutFile.h"

using namespace std;

// --------------------------------
//       class DTTracoLUTs
//---------------------------------

//----------------
// Constructors --
//----------------

DTTracoLUTs::DTTracoLUTs(string testfile) : _testfile(testfile) {}

//--------------
// Destructor --
//--------------

DTTracoLUTs::~DTTracoLUTs() {
  psi_lut.clear();
  for (int k = 0; k < 3; k++)
    phi_lut[k].clear();
}

//--------------
// Operations --
//--------------

//
// reset look-up tables
//
void DTTracoLUTs::reset() {
  psi_lut.clear();
  for (int k = 0; k < 3; k++)
    phi_lut[k].clear();
}

//
// load look-up tables for traco
//
int DTTracoLUTs::load() {
  // get file name in current directory
  string ang_file = _testfile + ".anglut";
  string pos_file = _testfile + ".poslut";

  // open file for PSI
  DTTPGLutFile filePSI(ang_file);
  if (filePSI.open() != 0)
    return -1;

  // ignore comment lines
  // filePSI.ignoreLines(14);

  // read file for PSI values --->   psi is 10 bits, 9+sign(10,11...16),
  // resolution 9 bits,
  for (int u = 0; u < 1024; u++) {
    int word = filePSI.readHex();  // read a 16 bits word
    // int psi = word & 0x01FF;    //bits  0,1,...8
    // int sgn = word & 0x0200;    //bit 9
    // if(sgn)
    // psi = -psi;
    psi_lut.push_back(word);  // positive value
  }
  filePSI.close();

  // open file for PHI
  DTTPGLutFile filePHI(pos_file);
  if (filePHI.open() != 0)
    return -1;

  // read file for PHI values    --->  phi is 12 bits, 11+sign(12..16),
  // resolution 12 bits
  for (int y = 0; y < 3; y++) {  // 3 series of values: I-outer, II-innner, III-correlated
    for (int h = 0; h < 512; h++) {
      int phi = filePHI.readHex();
      // phi &= 0x0FFF;                //get 12 bits
      // int sgn = phi;
      // sgn >> 11;                    //bit 12 for sign
      // sgn &= 0x01;
      // if(sgn==1)                    //negative value
      // phi = -phi;
      phi_lut[y].push_back(phi);  // positive value
    }
  }
  filePHI.close();
  return 0;
}

//
// print look-up tables for EMU
//
void DTTracoLUTs::print() const {
  cout << endl;
  cout << "L1 barrel Traco look-up tables :" << endl;
  cout << "====================================================" << endl;
  cout << endl;

  //  int i = 0;
  //  for debugging
  for (int x = 0; x < 1024; x++)
    cout << "K=" << x << " ---> " << hex << psi_lut[x] << dec << endl;
  for (int m = 0; m < 512; m++)
    cout << "X=" << m << " ---> " << hex << (phi_lut[0])[m] << "  " << (phi_lut[1])[m] << "  " << (phi_lut[2])[m]
         << "  " << dec << endl;
}

//
// get phi radial value for a given position
//
unsigned short int DTTracoLUTs::getPhiRad(int pos, int flag) const {
  unsigned short int phi = (phi_lut[flag])[pos] & 0xFFF;  // 12 bits
  // int sgn = (phi_lut[flag])[pos]  &  0x800;     //bit 12 for sign
  // if(sgn)
  // phi = - phi;

  return phi;
}

//
// get psi value for a given angle
//
unsigned short int DTTracoLUTs::getPsi(int ang) const {
  unsigned short int ipsi = (psi_lut)[ang + 512];  // scritto in complemento a
                                                   // due
  /*
    //debug: try with formula
    float fpsi = atan( ((float)(ang) * 4.2) /(18 * 1.3 * 30 ));
    fpsi*=512;
    if(fpsi<=0)fpsi-=1.0;
    int ipsi = (int)fpsi;
    // if outside range set to lower edge
    if( ipsi>= 512 ||
        ipsi< -512 ) ipsi=-512;
  cout << "psi is="<<ipsi <<endl;
  */
  return ipsi;
}

//
// get bending angle
//
unsigned short int DTTracoLUTs::getBendAng(int pos, int ang, int flag) const {
  // bendAng = psi - phi  : psi ha risoluzione 12, phi 9, quindi devo riportarli
  // alla stessa risoluzione con : phi/8 (scarto i 3 bit meno significativi). Il
  // risultato ha risoluzione 10 bits.
  unsigned short int BendAng = ((psi_lut)[ang + 512] - ((phi_lut[flag])[pos] / 8)) & 0x3FF;  // 10 bits

  // cout << "Bending angle is:" << hex << BendAng << endl;
  // cout << "Abs of bending angle is:" << hex << abs(BendAng) << endl;

  return BendAng;
}
