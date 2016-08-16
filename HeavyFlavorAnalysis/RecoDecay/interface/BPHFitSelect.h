#ifndef BPHFitSelect_H
#define BPHFitSelect_H
/** \class BPHFitSelect
 *
 *  Description: 
 *     Base class for candidate selection at kinematic fit level
 *
 *
 *  $Date: 2015-07-06 18:38:16 $
 *  $Revision: 1.1 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------


//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
class BPHKinematicFit;

//---------------
// C++ Headers --
//---------------


//              ---------------------
//              -- Class Interface --
//              ---------------------

class BPHFitSelect {

 public:

  /** Constructor
   */
  BPHFitSelect();

  /** Destructor
   */
  virtual ~BPHFitSelect();

  /** Operations
   */
  /// accept function
  virtual bool accept( const BPHKinematicFit& cand ) const = 0;

 private:

  // private copy and assigment constructors
  BPHFitSelect           ( const BPHFitSelect& x );
  BPHFitSelect& operator=( const BPHFitSelect& x );

};


#endif // BPHFitSelect_H

