//-------------------------------------------------
//
//   Class L1MuDTChambPhDigi
//
//   Description: input data for Phase2 trigger
//
//
//   Author List: Federica Primavera  Bologna INFN
//
//
//--------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "../interface/L1MuDTChambDigi.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------


//---------------
// C++ Headers --
//---------------
using namespace std;

//-------------------
// Initializations --
//-------------------


//----------------
// Constructors --
//----------------
L1MuDTChambDigi::L1MuDTChambDigi() {

  m_bx              = -100;
  m_wheel           = 0;
  m_sector          = 0;
  m_station         = 0;
  m_phiAngle        = 0;
  m_phiBending      = 0;
  m_zCoordinate     = 0;
  m_zSlope          = 0;

  m_qualityCode     = 7;
  m_segmentIndex    = 0;

  m_t0Segment       = 0;
  m_chi2Segment     = 0;

  m_rpcFlag          = -10;
}


L1MuDTChambDigi::L1MuDTChambDigi( int ubx,  int uwh, int usc, int ust, int uphi, int uphib, int uz, int uzsl,
                                  int uqua, int uind, int ut0, int uchi2, int urpc)
{

  m_bx             = ubx;
  m_wheel          = uwh;
  m_sector         = usc;
  m_station        = ust;
  m_phiAngle       = uphi;
  m_phiBending     = uphib;
  m_zCoordinate    = uz;
  m_zSlope         = uzsl;

  m_qualityCode    = uqua;
  m_segmentIndex   = uind;

  m_t0Segment      = ut0;
  m_chi2Segment    = uchi2;

  m_rpcFlag         = urpc;
}



//--------------
// Destructor --
//--------------
L1MuDTChambDigi::~L1MuDTChambDigi() {
}

//--------------
// Operations --
//--------------
int L1MuDTChambDigi::bxNum() const {
  return m_bx;
}

int L1MuDTChambDigi::whNum() const {
  return m_wheel;
}
int L1MuDTChambDigi::scNum() const {
  return m_sector;
}
int L1MuDTChambDigi::stNum() const {
  return m_station;
}

int L1MuDTChambDigi::phi() const {
  return m_phiAngle;
}

int L1MuDTChambDigi::phiBend() const {
  return m_phiBending;
}

int L1MuDTChambDigi::z() const {
  return m_zCoordinate;
}

int L1MuDTChambDigi::zSlope() const {
  return m_zSlope;
}

int L1MuDTChambDigi::quality() const {
  return m_qualityCode;
}

int L1MuDTChambDigi::index() const{
  return m_segmentIndex;
}

int L1MuDTChambDigi::t0() const {
  return m_t0Segment;
}

int L1MuDTChambDigi::chi2() const {
  return m_chi2Segment;
}

int L1MuDTChambDigi::rpcFlag() const {
  return m_rpcFlag;
}
