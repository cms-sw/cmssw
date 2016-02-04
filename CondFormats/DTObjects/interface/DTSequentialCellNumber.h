#ifndef DTSequentialCellNumber_H
#define DTSequentialCellNumber_H
/** \class DTSequentialCellNumber
 *
 *  Description:
 *       Class to compute a sequential number for drift tube cells
 *
 *  $Date: 2010/05/06 14:42:49 $
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

class DTSequentialCellNumber {

 public:

  DTSequentialCellNumber();
  ~DTSequentialCellNumber();

  int id( int      wheel, int station, int sector,
          int superlayer, int   layer, int cell );
  static int max();

 private:

  static int cellsPerWheel;
  static int cellsPerSector;
  static int cellsIn13Sectors;
  static int cellsInTheta;
  static int cellsInMB1;
  static int cellsInMB2;
  static int cellsInMB3;
  static int cellsInMB4;

  static int* offsetChamber;
  static int* cellsPerLayer;

};


#endif // DTSequentialCellNumber_H
