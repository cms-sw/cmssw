#ifndef DTSequentialLayerNumber_H
#define DTSequentialLayerNumber_H
/** \class DTSequentialLayerNumber
 *
 *  Description:
 *       Class to compute a sequential number for drift tube layers
 *
 *  $Date: 2010/04/30 16:20:08 $
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

class DTSequentialLayerNumber {

 public:

  DTSequentialLayerNumber();
  ~DTSequentialLayerNumber();

  static int id( int      wheel, int station, int sector,
                 int superlayer, int   layer );
  static int max();
};

#endif // DTSequentialLayerNumber_H
