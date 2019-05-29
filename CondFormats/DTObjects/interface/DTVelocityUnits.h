#ifndef DTVelocityUnits_H
#define DTVelocityUnits_H
/** \class DTVelocityUnits
 *
 *  Description: 
 *       Class to contain time units identifier
 *
 *  $Date: 2006/05/04 06:54:02 $
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

//---------------
// C++ Headers --
//---------------

//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTVelocityUnits {
public:
  enum type { cm_per_count, cm_per_ns };

  /** Destructor
   */
  virtual ~DTVelocityUnits();

private:
  /** Constructor
   */
  DTVelocityUnits();
  /** Operations
   */
  ///
};

#endif  // DTVelocityUnits_H
