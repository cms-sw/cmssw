/** \class CSCLayerInfo
 *
 * Auxiliary class containing vectors of comparator or wire RecDigis and
 * their matched SimHits for given Layer.
 *
 * \author Jason Mumford, Slava Valuev  21 August 2001
 * Porting from ORCA by S. Valuev in September 2006.
 *
 * $Id: CSCLayerInfo.h,v 1.2.4.1 2012/05/16 00:31:26 khotilov Exp $
 *
 */

#ifndef CSCTriggerPrimitives_CSCLayerInfo_H
#define CSCTriggerPrimitives_CSCLayerInfo_H

#include <iomanip>
#include <vector>
#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <SimDataFormats/TrackingHit/interface/PSimHitContainer.h>

template <class TYPE>
class CSCLayerInfo
{
 public:
  /** default constructor */
  CSCLayerInfo();

  /** destructor */
  ~CSCLayerInfo();

  /** clears LayerInfo */
  void clear();

  /** sets detId of this layer */
  void setId(const CSCDetId id)           {theLayerId = id;}

  /** fills RecDigi */
  void addComponent(const TYPE digi)      {RecDigis.push_back(digi);}

  /** fills SimHit */
  void addComponent(const PSimHit simHit) {SimHits.push_back(simHit);}

  /** returns the layer */
  CSCDetId getId() const                  {return theLayerId;}

  /** returns the vector of RecDigis (comparator or wire) */
  std::vector<TYPE> getRecDigis() const   {return RecDigis;}

  /** returns the vector of SimHits */
  std::vector<PSimHit> getSimHits() const {return SimHits;}

 private:
  CSCDetId theLayerId;
  std::vector<TYPE> RecDigis;
  std::vector<PSimHit> SimHits;
};


template<class TYPE> CSCLayerInfo<TYPE>::CSCLayerInfo() {
  CSCDetId tmp;        // nullify theLayerId.
  theLayerId = tmp;
  RecDigis.reserve(3); // we may have up to three RecDigis per layer.
  SimHits.reserve(3);
}

template<class TYPE> CSCLayerInfo<TYPE>::~CSCLayerInfo() {
  clear();
}

template<class TYPE> void CSCLayerInfo<TYPE>::clear() {
  CSCDetId tmp;        // nullify theLayerId.
  theLayerId = tmp;
  // Use the trick from ORCA-days "CommonDet/DetUtilities/interface/reset.h"
  // to delete the capacity of the vectors.
  std::vector<TYPE> temp_digis;
  std::vector<PSimHit> temp_hits;
  RecDigis.swap(temp_digis);
  SimHits.swap(temp_hits);
}

// overloaded << operator
template<class TYPE>
std::ostream& operator<< (std::ostream& output,
			  const CSCLayerInfo<TYPE>& info) {
  std::vector<TYPE> thisLayerDigis = info.getRecDigis();
  // vector<TYPE>::iterator prd; /* upsets pedantic compillation on LINUX */
  if (thisLayerDigis.size() > 0) {
    output << "Layer: " << std::setw(1) << info.getId().layer();
    for (unsigned int i = 0; i < thisLayerDigis.size(); i++) {
      output << " RecDigi # " << i+1 << ": " << thisLayerDigis[i] << '\t';
    }
  }
  std::vector<PSimHit> thisLayerHits = info.getSimHits();
  if (thisLayerHits.size() > 0) {
    output << "Layer: " << std::setw(1) << info.getId().layer();
    for (unsigned int i = 0; i < thisLayerHits.size(); i++) {
      output << " SimHit # " << i+1 << ": " << thisLayerHits[i] << '\t';
    }
    output << std::endl;
  }
  return output;
}
#endif // CSCTriggerPrimitives_CSCLayerInfo_H
