#ifndef ECALEBCLUSTERTRIGGERPRIMITIVESAMPLE_H
#define ECALEBCLUSTERTRIGGERPRIMITIVESAMPLE_H 1

#include <boost/cstdint.hpp>
#include <ostream>


/** \class EcalEBClusterTriggerPrimitiveSample
\author N. Marinelli - Univ of Notre Dame

*/

class EcalEBClusterTriggerPrimitiveSample {
 public:
  EcalEBClusterTriggerPrimitiveSample();
  EcalEBClusterTriggerPrimitiveSample(uint64_t data);
  EcalEBClusterTriggerPrimitiveSample(int encodedEt);
  EcalEBClusterTriggerPrimitiveSample(int encodedEt, bool isASpike);
  EcalEBClusterTriggerPrimitiveSample(int encodedEt, bool isASpike, int timing);
  

  ///Set data
  void setValue(uint64_t data){ theSample = data;}
  // The cluster trigger primitive  is a 37 bit word defined as:


  //
  //   o o o o o  o o o o o o o o  o o o o o o o o  o o o o o   o    o o o o o o o o o o
  //  |_________||_______________||_______________| |_______|       |___________________| 
  //     # xTals      eta              phi          ~60ps res  spike         Et
  //                                                  time     flag
  //

  // Uhm, cluster position and number of crystals in the cluster are not really needed for each sample. 
  // I store that stuf once in the the final digi

  /// get the raw word
  uint64_t raw() const { return theSample; }

  /// get the encoded Et (10 bits)
  int encodedEt() const { return theSample&0x3FF; }

  int l1aSpike() const  { return (theSample&0x400)!=0; }

  int time() const { return theSample>>11; }


  

  
  /// for streaming
  uint64_t operator()() { return theSample; }

 private:
  uint64_t theSample;
  
};

std::ostream& operator<<(std::ostream& s, const EcalEBClusterTriggerPrimitiveSample& samp);
  



#endif
