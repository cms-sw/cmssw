#ifndef MuonNumbering_RPCNumberingScheme_h
#define MuonNumbering_RPCNumberingScheme_h

/** \class RPCNumberingScheme
 *
 * implementation of MuonNumberingScheme for muon rpc,
 * converts the MuonBaseNumber to a unit id
 *  
 *  $Date: 2006/10/12 19:54:05 $
 *  $Revision: 1.4 $
 * \author Arno Straessner, CERN <arno.straessner@cern.ch>
 *
 */

#include "Geometry/MuonNumbering/interface/MuonNumberingScheme.h"

class MuonBaseNumber;
class DDCompactView;
class MuonDDDConstants;

class RPCNumberingScheme : public MuonNumberingScheme {
 public:

  RPCNumberingScheme( const DDCompactView& cpv );
  RPCNumberingScheme( const MuonDDDConstants& muonConstants );

  virtual ~RPCNumberingScheme(){};
  
  virtual int baseNumberToUnitNumber(const MuonBaseNumber);
  
 private:
  void initMe ( const MuonDDDConstants& muonConstants );

  int theRegionLevel;
  int theBWheelLevel;
  int theBStationLevel;
  int theBPlaneLevel;
  int theBChamberLevel;
  int theEPlaneLevel;
  int theESectorLevel;
  int theERollLevel;

};

#endif
