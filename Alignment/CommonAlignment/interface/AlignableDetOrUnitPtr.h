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
 *  $Date: 2007/04/30 11:33:56 $
 *  $Revision: 1.1.2.1 $
 *  (last update by $Author: flucke $)
 */

class Alignable;
class AlignableDet;
class AlignableDetUnit;

class AlignableDetOrUnitPtr 
{  
 public:
  /// Constructor from AlignableDet* (non-explicit: for automatic conversions)
  inline AlignableDetOrUnitPtr(AlignableDet *aliDet) : theAliDet(aliDet), theAliDetUnit(0) {}
  /// Constructor from AlignableDetUnit* (non-explicit: for automatic conversions)
  inline AlignableDetOrUnitPtr(AlignableDetUnit *aliDetUnit) 
    : theAliDet(0), theAliDetUnit(aliDetUnit) {}
  /// Non-virtual destructor: do not use as base class
  inline ~AlignableDetOrUnitPtr() {}

  inline AlignableDetOrUnitPtr& operator = (AlignableDet* aliDet) {
    theAliDet = aliDet; theAliDetUnit = 0; return *this;}
  inline AlignableDetOrUnitPtr& operator = (AlignableDetUnit* aliDetUnit) {
    theAliDet = 0; theAliDetUnit = aliDetUnit; return *this;}
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
  inline AlignableDet* alignableDet() { return theAliDet;}
  inline const AlignableDet* alignableDet() const { return theAliDet;}
  inline AlignableDetUnit* alignableDetUnit() { return theAliDetUnit;}
  inline const AlignableDetUnit* alignableDetUnit() const { return theAliDetUnit;}

  inline bool isNull() const { return (!theAliDet && !theAliDetUnit);}

private:
  AlignableDet     *theAliDet;      ///< Pointer to Alignable if it is a Det
  AlignableDetUnit *theAliDetUnit;  ///< Pointer to Alignable if it is a DetUnit
};

#endif
