#ifndef MERun_hh
#define MERun_hh

#include <iostream>
#include <map>
#include <vector>

#include <TString.h>
#include <TFile.h>
#include <TH2.h>
#include <TTree.h>

#include "../../interface/ME.h"
#include "../../interface/MEEBGeom.h"
#include "../../interface/MEEEGeom.h"

class MERun 
{
  // only a run manager can create one of those
  friend class MERunManager;
  MERun( ME::Header header, ME::Settings settings, TString fname );

public:

  virtual ~MERun();
  
  bool operator==( const MERun& o ) const;

  unsigned int run() const { return _header.run; }
  unsigned int lb() const  { return _header.lb;  }
  ME::Time time() const;
  ME::TimeStamp timestamp() const    { return _header.ts_beg; }
  TString rundir() const  { return _header.rundir; }

  TFile* laserPrimFile( bool refresh=false );
  void closeLaserPrimFile();

  float getVal( int table, int var, int ix=0, int iy=0 );

  const ME::Header   header()   const { return _header;   }
  const ME::Settings settings() const { return _settings; }

  void print( std::ostream& o ) const;

private :

  ME::Header     _header;
  ME::Settings _settings;
  int _type;
  int _color;

  TString _fname;
  TFile*  _file;  

  // get 2-D histogram from the file
  TH2* APDHist( int var );
  std::map< TString, TH2* > _h;
  
  TTree* PNTable();
  TTree* pn_t;
  std::map< TString, unsigned int    > pn_i;
  std::map< TString, float > pn_d;

  TTree* MTQTable();
  TTree* mtq_t;
  std::map< TString, unsigned int    > mtq_i;
  std::map< TString, float > mtq_d;

  ClassDef( MERun, 0 ) // MERun -- a Laser monitoring run
};

// struct SortMERun
// {
//   bool operator()( const MERun* r1, const MERun* r2 ) const
//   {
//     int color1 = r1->settings().wavelength;
//     int color2 = r2->settings().wavelength;
//     if( color1!=color2 )
//       {
// 	return color1<color2;
//       }
//     int dcc1 = r1->header().dcc;
//     int dcc2 = r2->header().dcc;
//     if( dcc1!=dcc2 )
//       {
// 	return dcc1<dcc2;
//       }
//     int side1 = r1->header().side;
//     int side2 = r2->header().side;
//     if( side1!=side2 )
//       {
// 	return side1<side2;
//       }
//     float TS1 = r1->header().ts_beg;
//     float TS2 = r2->header().ts_beg;
//     if( TS1!=TS2 )
//       {
// 	return TS1<TS2;
//       }
//   }
//};

#endif
