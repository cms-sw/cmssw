//   COCOA class header file
//Id:  EntryAngleAffAngles.h
//CAT: Model
//
//   class for the three entries that make the affine frame angles
// 
//   History: v1.0 
//   Pedro Arce

#ifndef _ENTRYANGLEAFFANGLES_HH
#define _ENTRYANGLEAFFANGLES_HH

#include "Alignment/CocoaModel/interface/EntryAngle.h"


class EntryAngleAffAngles : public EntryAngle
{
public:
  EntryAngleAffAngles( const ALIstring& type );
  ~EntryAngleAffAngles(){};

  virtual void FillName( const ALIstring& name );
  virtual void displace( ALIdouble disp );
  virtual void displaceOriginal( ALIdouble disp );
  virtual void displaceOriginalOriginal( ALIdouble disp );
  virtual ALIdouble valueDisplaced() const;
  ALIdouble checkDiff( const CLHEP::Hep3Vector& axis, const CLHEP::Hep3Vector& axisOrig,const std::vector<double>& localrot, const std::vector<double>& localrotorig ) const;
};

#endif
