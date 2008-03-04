#ifndef CastorPedestalWidths_h
#define CastorPedestalWidths_h

/** 
\class CastorPedestalWidths
\author Panos Katsas (UoA)
POOL container to store PedestalWidth values 4xCapId
$Author: katsas
*/

#include <vector>
#include <algorithm>

#include "CondFormats/CastorObjects/interface/CastorPedestalWidth.h"

#include "DataFormats/DetId/interface/DetId.h"

// 
class CastorPedestalWidths {
 public:
  CastorPedestalWidths();
  ~CastorPedestalWidths();

  /// get all values
  const CastorPedestalWidth* getValues (DetId fId) const;
  /// get value for given capId = 0..3
  float getWidth (DetId fId, int fCapId) const;
  /// get correlation for given capId1/2 = 0..3
  float getSigma (DetId fId, int fCapId1, int fCapId2) const;
  /// get list of all available channels
  std::vector<DetId> getAllChannels () const;
  /// check if data are sorted
  bool sorted () const {return mSorted;}
  /// add new (empty) item
  CastorPedestalWidth* setWidth (DetId fId);
  /// add new (empty) item
  void setWidth (const CastorPedestalWidth& fItem);
  /// sort values by channelId  
  void sort ();
  typedef CastorPedestalWidth Item;
 private:
  std::vector <CastorPedestalWidth> mItems;
  bool mSorted;
};

#endif
