#ifndef DTSequentialCellNumber_H
#define DTSequentialCellNumber_H
/** \class DTSequentialCellNumber
 *
 *  Description:
 *       Class to compute a sequential number for drift tube cells
 *
 *  $Date: 2012/02/07 18:34:59 $
 *  $Revision: 1.1.2.1 $
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

class DTSequentialCellNumber {
public:
  static int id(int wheel, int station, int sector, int superlayer, int layer, int cell);
  static int max();
};

#endif  // DTSequentialCellNumber_H
