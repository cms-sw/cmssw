#ifndef DIGIECAL_EBDATAFRAME_H
#define DIGIECAL_EBDATAFRAME_H

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDigi/interface/EcalDataFrame.h"
#include <iosfwd>



/** \class EBDataFrame
      
*/
class EBDataFrame : public EcalDataFrame 
{
 public:
  typedef EBDetId key_type; ///< For the sorted collection
  typedef EcalDataFrame Base;

  EBDataFrame() {}
  // EBDataFrame(DetId i) :  Base(i) {}
  EBDataFrame(edm::DataFrame const & base) : Base(base) {}
  EBDataFrame(EcalDataFrame const & base) : Base(base) {}

  /** estimator for a signal being a spike
   *  based on ratios between 4th, 5th and 6th sample
   */
  float spikeEstimator() const;
    
  ~EBDataFrame() override {}

  key_type id() const { return Base::id(); }

};

std::ostream& operator<<(std::ostream&, const EBDataFrame&);


#endif
