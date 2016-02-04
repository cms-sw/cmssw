#ifndef MuonNumbering_MuonSimHitNumberingScheme_h
#define MuonNumbering_MuonSimHitNumberingScheme_h

/** \class MuonSimHitNumberingScheme
 *
 * wrapper class to handle numbering schemes for the different
 * MuonSubDetector's
 *  
 *  $Date: 2006/10/12 19:54:05 $
 *  $Revision: 1.2 $
 * \author Arno Straessner, CERN <arno.straessner@cern.ch>
 *
 */

#include "Geometry/MuonNumbering/interface/MuonNumberingScheme.h"

class MuonBaseNumber;
class MuonSubDetector;
class DDCompactView; 

class MuonSimHitNumberingScheme : public MuonNumberingScheme {

 public:

  MuonSimHitNumberingScheme(MuonSubDetector*, const DDCompactView& cpv);
  ~MuonSimHitNumberingScheme();
  
  virtual int baseNumberToUnitNumber(const MuonBaseNumber);
  
 private:

  MuonSubDetector* theDetector;
  MuonNumberingScheme* theNumbering;
};

#endif
