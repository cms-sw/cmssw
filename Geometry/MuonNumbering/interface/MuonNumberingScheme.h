#ifndef Geometry_MuonNumbering_MuonNumberingScheme_h
#define Geometry_MuonNumbering_MuonNumberingScheme_h

/** \class MuonNumberingScheme
 *
 * base class of the muon numbering scheme;
 * each implementation should supply a conversion from
 * MuonBaseNumber to a detector unit id
 *  
 * \author Arno Straessner, CERN <arno.straessner@cern.ch>
 *
 */

class MuonBaseNumber;
class MuonDDDConstants;

class MuonNumberingScheme {
 public:

  MuonNumberingScheme( ){};
  virtual ~MuonNumberingScheme(){};
  
  virtual int baseNumberToUnitNumber(const MuonBaseNumber&);
  
 private:

};

#endif
