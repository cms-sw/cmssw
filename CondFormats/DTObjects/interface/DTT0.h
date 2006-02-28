#ifndef DTT0_H
#define DTT0_H
/** \class DTT0
 *
 *  Description:
 *       Class to hold drift tubes T0s
 *             ( cell by cell time offsets )
 *
 *  $Date: 2006/01/27 15:21:15 $
 *  $Revision: 1.3 $
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
#include <string>
#include <vector>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTCellT0Data {

 public:

  DTCellT0Data();
  ~DTCellT0Data();

  int   wheelId;
  int stationId;
  int  sectorId;
  int      slId;
  int   layerId;
  int    cellId;
  int t0mean;
  int t0rms;

};


class DTT0 {

 public:

  /** Constructor
   */
  DTT0();
  DTT0( const std::string& version );

  /** Destructor
   */
  ~DTT0();

  /** Operations
   */
  /// read and store full content
  void initSetup() const;

  /// get content
  int cellT0( int   wheelId,
              int stationId,
              int  sectorId,
              int      slId,
              int   layerId,
              int    cellId,
              int&   t0mean,
              float& t0rms ) const;

  /// access version
  const
  std::string& version() const;
  std::string& version();

  /// reset content
  void clear();

  int setCellT0( int   wheelId,
                 int stationId,
                 int  sectorId,
                 int      slId,
                 int   layerId,
                 int    cellId,
                 int   t0mean,
                 float t0rms );

  /// Access methods to data
  typedef std::vector<DTCellT0Data>::const_iterator const_iterator;
  const_iterator begin() const;
  const_iterator end() const;

 private:

  std::string dataVersion;

  std::vector<DTCellT0Data> cellData;

  static int rmsFactor;

};


#endif // DTT0_H

