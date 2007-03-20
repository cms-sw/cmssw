#ifndef TrackCollectionFP420_h
#define TrackCollectionFP420_h

#include "RecoRomanPot/RecoFP420/interface/TrackFP420.h"
#include <vector>
#include <map>
#include <utility>

class TrackCollectionFP420 {

 public:

  typedef std::vector<TrackFP420>::const_iterator ContainerIterator;
  typedef std::pair<ContainerIterator, ContainerIterator> Range;
  typedef std::pair<unsigned int, unsigned int> IndexRange;
  typedef std::map<unsigned int, IndexRange> Registry;
  typedef std::map<unsigned int, IndexRange>::const_iterator RegistryIterator;

    TrackCollectionFP420() {}
    //  TrackCollectionFP420() {
    //    container_.clear();
    //}
    
    //        virtual ~TrackCollectionFP420(){
    //    delete TrackCollectionFP420
    //   cout << " TrackCollectionFP420h:     delete TrackCollectionFP420  " << endl;
    //        }

  
    void put(Range input, unsigned int stationID);
    const Range get(unsigned int stationID) const;
    const std::vector<unsigned int> stationIDs() const;
    void putclear(Range input, unsigned int stationID);
    void clear();
    
 private:
    mutable std::vector<TrackFP420> container_;
    mutable Registry map_;
    
};

#endif // 


