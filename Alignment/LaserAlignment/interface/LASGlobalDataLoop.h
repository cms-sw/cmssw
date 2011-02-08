#ifndef __LASGLOBALDATALOOP_H
#define __LASGLOBALDATALOOP_H

#include <stdexcept>

#include "Alignment/LaserAlignment/interface/LASGlobalData.h"
#include <iostream>

///
/// helper class for looping over LASGlobalData objects
/// Use like this, where T can be any class or type supported by LASGlobalData:
/// \code
/// LASGlobalData<T> mydata;
/// // Fill mydata with something...
///
/// LASGlobalDataLoop theLoop();
/// do
/// {
///   T& entry_ref = theLoop<T>.GetEntry(mydata);
///   // Now entry_ref is refering to a valid entry
/// }while ( theLoop.next() );
/// 
/// // Alternative:
/// for( LASGlobalDataLoop theLoop(); ! theLoop.finished(); theLoop.next()){
///   T& entry_ref = theLoop.GetEntry<T>(mydata);
///   // Now entry_ref is refering to a valid entry
/// }
/// \endcode
///

class LASGlobalDataLoop {
 public:
  enum loop_type{ALL, TEC_PLUS, TEC_MINUS, TEC, AT, TIB, TOB, TEC_PLUS_AT, TEC_MINUS_AT, TEC_AT};
  LASGlobalDataLoop(loop_type lp_tp = ALL);
  bool next();
  bool finished(){return loop_finished;}
  template <class T> T& GetEntry(LASGlobalData<T>& data){return data.GetEntry(det, ring, beam, zpos);}
  void inspect(std::ostream & out = std::cout);

 private:
  loop_type the_loop_type;
  int det;
  int beam;
  int ring;
  int zpos;
  bool loop_finished;
  int max_det;
  int max_beam;
  int max_ring;
  int max_zpos;
};




#endif
