#ifndef Alignment_CommonAlignment_AlignableDetOrUnitPtr_H
#define Alignment_CommonAlignment_AlignableDetOrUnitPtr_H

/** \class AlignableDetOrUnitPtr
 *
 * Class to hold either a pointer to an AlignableDet or to an AlignableDetUnit.
 * As such it is like a pointer to an Alignable of that hierarchy level
 * on which hits can exists. Therefore it should be used whenever it should be
 * ensured by C++ type safety that a lowest level Alignable* is dealt with.
 * 
 * Conversion and '->' operators exist to use this class as a pointer to the base
 * class Alignable. On the other hand, the accessors alignableDet() and
 * alignableDetUnit() can be used to check which concrete type it is.
 *
 * Since this class is very light weighted, it can be used by value.
 *
 *  Original author: Gero Flucke, April 2007
 *
 *  $Date: 2009/02/28 21:04:59 $
 *  $Revision: 1.3 $
 *  (last update by $Author: flucke $)
 */

class Alignable;
class AlignableBeamSpot;
class AlignableDet;
class AlignableDetUnit;
class AlignmentPositionError;

class AlignableDetOrUnitPtr 
{  
 public:

  /// Constructor from AlignableBeamSpot* (non-explicit: for automatic conversions)
  inline AlignableDetOrUnitPtr(AlignableBeamSpot *aliBeamSpot)
    : theAliBeamSpot(aliBeamSpot), theAliDet(0), theAliDetUnit(0) {}

  /// Constructor from AlignableDet* (non-explicit: for automatic conversions)
  inline AlignableDetOrUnitPtr(AlignableDet *aliDet)
    : theAliBeamSpot(0), theAliDet(aliDet), theAliDetUnit(0) {}

  /// Constructor from AlignableDetUnit* (non-explicit: for automatic conversions)
  inline AlignableDetOrUnitPtr(AlignableDetUnit *aliDetUnit) 
    : theAliBeamSpot(0), theAliDet(0), theAliDetUnit(aliDetUnit) {}
  /// Non-virtual destructor: do not use as base class
  inline ~AlignableDetOrUnitPtr() {}

  inline AlignableDetOrUnitPtr& operator = (AlignableBeamSpot* aliBeamSpot) {
    theAliBeamSpot = aliBeamSpot; theAliDet = 0; theAliDetUnit = 0; return *this;}
  inline AlignableDetOrUnitPtr& operator = (AlignableDet* aliDet) {
    theAliBeamSpot = 0; theAliDet = aliDet; theAliDetUnit = 0; return *this;}
  inline AlignableDetOrUnitPtr& operator = (AlignableDetUnit* aliDetUnit) {
    theAliBeamSpot = 0; theAliDet = 0; theAliDetUnit = aliDetUnit; return *this;}
  // Default operator= and default copy constructor are fine, no need to code them here.

  // conversions to Alignable* etc.
  operator Alignable* ();
  operator const Alignable* () const;
  inline Alignable* operator->() { return this->operator Alignable*();}
  inline const Alignable* operator->() const { return this->operator const Alignable*();}
  // explicit call if needed:
  inline Alignable* alignable() { return this->operator Alignable*();}
  inline const Alignable* alignable() const { return this->operator const Alignable*();}

  // explicit access to specific types
  inline AlignableBeamSpot* alignableBeamSpot() { return theAliBeamSpot;}
  inline const AlignableBeamSpot* alignableBeamSpot() const { return theAliBeamSpot;}
  inline AlignableDet* alignableDet() { return theAliDet;}
  inline const AlignableDet* alignableDet() const { return theAliDet;}
  inline AlignableDetUnit* alignableDetUnit() { return theAliDetUnit;}
  inline const AlignableDetUnit* alignableDetUnit() const { return theAliDetUnit;}

  /// check for empty pointer
  inline bool isNull() const { return (!theAliBeamSpot && !theAliDet && !theAliDetUnit);}

  // interface to methods specific for AlignableDet(Unit),
  // slightly breaking idea of 'pointerness' of this class
  /// alignment position error (see comments in specific classes)
  const AlignmentPositionError* alignmentPositionError() const;

private:
  AlignableBeamSpot *theAliBeamSpot; ///< Pointer to Alignable if it is the beam spot
  AlignableDet      *theAliDet;      ///< Pointer to Alignable if it is a Det
  AlignableDetUnit  *theAliDetUnit;  ///< Pointer to Alignable if it is a DetUnit
};

#endif
