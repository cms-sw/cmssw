#ifndef MuonNumbering_MuonNumberingScheme_h
#define MuonNumbering_MuonNumberingScheme_h

/** \class MuonNumberingScheme
 *
 * base class of the muon numbering scheme;
 * each implementation should supply a conversion from
 * MuonBaseNumber to a detector unit id
 *  
 *  $Date: 2005/10/18 16:54:41 $
 *  $Revision: 1.1 $
 * \author Arno Straessner, CERN <arno.straessner@cern.ch>
 *
 */

class MuonBaseNumber;

class MuonNumberingScheme {
 public:

  MuonNumberingScheme(){};
  virtual ~MuonNumberingScheme(){};
  
  virtual int baseNumberToUnitNumber(const MuonBaseNumber);
  
 private:

};

#endif
