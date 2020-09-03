#ifndef Geometry_MuonNumbering_DTNumberingScheme_h
#define Geometry_MuonNumbering_DTNumberingScheme_h

/** \class DTNumberingScheme
 *
 * implementation of MuonNumberingScheme for muon barrel,
 * converts the MuonBaseNumber to a unit id
 *  
 * Original \author Arno Straessner, CERN <arno.straessner@cern.ch>
 *         Modified by Sunanda B. in different PRs (the last one is #30971)
 *
 */

#include "Geometry/MuonNumbering/interface/MuonNumberingScheme.h"

class MuonBaseNumber;
class MuBarDetBuilder;
class MuonGeometryConstants;

class DTNumberingScheme : public MuonNumberingScheme {
public:
  DTNumberingScheme(const MuonGeometryConstants& muonConstants);
  ~DTNumberingScheme() override {}

  int baseNumberToUnitNumber(const MuonBaseNumber& num) const override;

  int getDetId(const MuonBaseNumber& num) const;

private:
  void initMe(const MuonGeometryConstants& muonConstants);
  // Decode MuonBaseNumber to id: no checking
  void decode(const MuonBaseNumber& num,
              int& wire_id,
              int& layer_id,
              int& superlayer_id,
              int& sector_id,
              int& station_id,
              int& wheel_id) const;

  int theRegionLevel;
  int theWheelLevel;
  int theStationLevel;
  int theSuperLayerLevel;
  int theLayerLevel;
  int theWireLevel;

  /** Same as BaseNumberToUnitNumber but w/o check: used by MuBarDetBuilder
   * class to build the geometry from DDD */
  friend class DTGeometryBuilderFromDDD;
  friend class DTGeometryParserFromDDD;
};

#endif
