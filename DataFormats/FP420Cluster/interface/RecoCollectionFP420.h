#ifndef RecoCollectionFP420_h
#define RecoCollectionFP420_h

#include "DataFormats/FP420Cluster/interface/RecoFP420.h"
#include <vector>
#include <map>
#include <utility>

class RecoCollectionFP420 {

 public:

  typedef std::vector<RecoFP420>::const_iterator ContainerIterator;
  typedef std::pair<ContainerIterator, ContainerIterator> Range;
  typedef std::pair<unsigned int, unsigned int> IndexRange;
  typedef std::map<unsigned int, IndexRange> Registry;
  typedef std::map<unsigned int, IndexRange>::const_iterator RegistryIterator;
  typedef std::map< unsigned int, std::vector<RecoFP420> > RecoFP420Container;

    RecoCollectionFP420() {}
    //  RecoCollectionFP420() {
    //    container_.clear();
    //}
    
    //        virtual ~RecoCollectionFP420(){
    //    delete RecoCollectionFP420
    //   cout << " RecoCollectionFP420h:     delete RecoCollectionFP420  " << endl;
    //        }

  
    void put(Range input, unsigned int stationID);
    const Range get(unsigned int stationID) const;
    const std::vector<unsigned int> stationIDs() const;
    void putclear(Range input, unsigned int stationID);
    void clear();
    
 private:
    std::vector<RecoFP420> container_;
    Registry map_;
    
    RecoFP420Container trackMap_; 
};

#endif // 


