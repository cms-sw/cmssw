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
  ~EntryLengthAffCentre() override{};

  virtual void FillName( const ALIstring& name );
  void displace( ALIdouble disp ) override;
  void displaceOriginal( ALIdouble disp ) override;
  void displaceOriginalOriginal( ALIdouble disp ) override;
  ALIdouble valueInGlobalReferenceFrame() const override;
  ALIdouble valueDisplaced() const override;

};

#endif
