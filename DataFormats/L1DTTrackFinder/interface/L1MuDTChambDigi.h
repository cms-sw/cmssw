//-------------------------------------------------
//
//   Class L1MuDTChambDigi	
//
//   Description: input data for Phase2 trigger
//
//
//   Author List: Federica Primavera  Bologna INFN
//
//
//--------------------------------------------------
#ifndef L1MuDTChambDigi_H
#define L1MuDTChambDigi_H

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------


//----------------------
// Base Class Headers --
//----------------------


//---------------
// C++ Headers --
//---------------


//              ---------------------
//              -- Class Interface --
//              ---------------------

class L1MuDTChambDigi {

 public:

  //  Constructors
  L1MuDTChambDigi();

  L1MuDTChambDigi( int ubx,  int uwh, int usc, int ust, int uphi, int uphib, int uz,  int uzsl,
		   int uqua, int uind, int ut0, int uchi2, int urpc=-10);
  
  //  Destructor
  ~L1MuDTChambDigi();

  // Operations
  int bxNum()       const;
  int whNum()       const;
  int scNum()       const;
  int stNum()       const;
  int phi()         const;
  int phiBend()     const;
  int z()           const;
  int zSlope()      const;

  int quality()     const;
  int index()       const;
   
  int t0()          const;
  int chi2()        const;

  int rpcFlag()      const;
  

 private:

  int m_bx;
  int m_wheel;
  int m_sector;
  int m_station;
  int m_phiAngle;
  int m_phiBending;
  int m_zCoordinate;
  int m_zSlope;

  int m_qualityCode;
  int m_segmentIndex;
  
  int m_t0Segment;
  int m_chi2Segment;
  
  int m_rpcFlag;
};

#endif
