#ifndef MuonNumbering_MuonNumberingScheme_h
#define MuonNumbering_MuonNumberingScheme_h

/** \class MuonNumberingScheme
 *
 * base class of the muon numbering scheme;
 * each implementation should supply a conversion from
 * MuonBaseNumber to a detector unit id
 *  
 *  $Date: 2006/02/15 13:21:24 $
 *  $Revision: 1.1 $
 * \author Arno Straessner, CERN <arno.straessner@cern.ch>
 *
 */

class MuonBaseNumber;
class MuonDDDConstants;

class MuonNumberingScheme {
 public:

  MuonNumberingScheme( ){};
  virtual ~MuonNumberingScheme(){};
  
  virtual int baseNumberToUnitNumber(const MuonBaseNumber);
  
 private:

};

#endif
