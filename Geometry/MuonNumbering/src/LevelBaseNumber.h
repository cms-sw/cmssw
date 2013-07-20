#ifndef MuonNumbering_LevelBaseNumber_h
#define MuonNumbering_LevelBaseNumber_h

/** \class LevelBaseNumber
 *
 * class for extended copy numbers for muon numbering;
 * 
 * in addition to the usual copy number (Base) the
 * geometry level (Level) is specified;
 * for detectors with identical copy numbers an additional 
 * identification number (Super) can be given
 *  
 *  $Date: 2006/02/15 13:22:48 $
 *  $Revision: 1.1 $
 * \author Arno Straessner, CERN <arno.straessner@cern.ch>
 *
 */

class LevelBaseNumber {
 public:
  LevelBaseNumber(const int l,const int s,const int b){
    myLevel=l;
    mySuper=s;
    myBase=b;
  };
  ~LevelBaseNumber(){};

  int level() const {return myLevel;};
  int super() const {return mySuper;};
  int base() const {return myBase;};

 private:
  int myLevel;
  int mySuper;
  int myBase;

};

#endif
