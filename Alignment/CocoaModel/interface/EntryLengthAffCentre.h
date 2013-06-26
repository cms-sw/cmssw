//   COCOA class header file
//Id:  EntryLengthAffCentre.h
//CAT: Model
//
//   class for the three entries that make the affine frame centre
// 
//   History: v1.0 
//   Pedro Arce

#ifndef _ENTRYLengthAffCentre_HH
#define _ENTRYLengthAffCentre_HH

#include "Alignment/CocoaModel/interface/EntryLength.h"


class EntryLengthAffCentre : public EntryLength
{
public:
  EntryLengthAffCentre( const ALIstring& type );
  ~EntryLengthAffCentre(){};

  virtual void FillName( const ALIstring& name );
  virtual void displace( ALIdouble disp );
  virtual void displaceOriginal( ALIdouble disp );
  virtual void displaceOriginalOriginal( ALIdouble disp );
  virtual ALIdouble valueInGlobalReferenceFrame() const;
  virtual ALIdouble valueDisplaced() const;

};

#endif
