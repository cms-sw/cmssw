#ifndef DTSequentialCellNumber_H
#define DTSequentialCellNumber_H
/** \class DTSequentialCellNumber
 *
 *  Description:
 *       Class to compute a sequential number for drift tube cells
 *
 *  $Date: 2012/02/18 10:46:28 $
 *  $Revision: 1.2 $
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

  static int id( int      wheel, int station, int sector,
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
