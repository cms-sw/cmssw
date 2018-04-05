#ifndef Geometry_MuonNumbering_CSCNumberingScheme_h
#define Geometry_MuonNumbering_CSCNumberingScheme_h

/** \class CSCNumberingScheme
 *
 * implementation of MuonNumberingScheme for muon endcaps,
 * converts the MuonBaseNumber to a unit id
 *  
 * \author Arno Straessner, CERN <arno.straessner@cern.ch>
 *
 */

#include "Geometry/MuonNumbering/interface/MuonNumberingScheme.h"

class MuonBaseNumber;
class DDCompactView;
class MuonDDDConstants;

class CSCNumberingScheme : public MuonNumberingScheme {
 public:

  CSCNumberingScheme( const MuonDDDConstants& muonConstants );
  CSCNumberingScheme( const DDCompactView& cpv );
  ~CSCNumberingScheme() override{};
  
  int baseNumberToUnitNumber(const MuonBaseNumber&) override;
  
 private:

  void initMe( const MuonDDDConstants& muonConstants );
  /**
   * Tim Cox - IMPORTANT - this is where we set CSC chamber labelling
   */
  int chamberIndex( int, int, int, int ) const;
  
  int theRegionLevel;
  int theStationLevel;
  int theSubringLevel;
  int theSectorLevel;
  int theLayerLevel;
  int theRingLevel;

};

#endif







