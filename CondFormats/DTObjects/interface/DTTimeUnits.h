#ifndef DTTimeUnits_H
#define DTTimeUnits_H
/** \class DTTimeUnits
 *
 *  Description: 
 *       Class to contain time units identifier
 *
 *  $Date: 2006-03-30 17:42:20 $
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

class DTTimeUnits {
public:
  enum type { counts, ns };

  /** Destructor
   */
  virtual ~DTTimeUnits();

private:
  /** Constructor
   */
  DTTimeUnits();
  /** Operations
   */
  ///
};

#endif  // DTTimeUnits_H
