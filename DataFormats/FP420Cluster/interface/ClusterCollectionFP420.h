#ifndef ClusterCollectionFP420_h
#define ClusterCollectionFP420_h

#include "DataFormats/FP420Cluster/interface/ClusterFP420.h"
#include <vector>
#include <map>
#include <utility>

class ClusterCollectionFP420 {

 public:

  typedef std::vector<ClusterFP420>::const_iterator ContainerIterator;
  typedef std::pair<ContainerIterator, ContainerIterator> Range;
  typedef std::pair<unsigned int, unsigned int> IndexRange;
  typedef std::map<unsigned int, IndexRange> Registry;
  typedef std::map<unsigned int, IndexRange>::const_iterator RegistryIterator;
  typedef std::map< unsigned int, std::vector<ClusterFP420> > ClusterFP420Container; 

    ClusterCollectionFP420() {}
    //  ClusterCollectionFP420() {
    //    container_.clear();
    //}

//        virtual ~ClusterCollectionFP420(){
    //    delete ClusterCollectionFP420
    //   cout << " ClusterCollectionFP420h:     delete ClusterCollectionFP420  " << endl;
//        }

  
  void put(Range input, unsigned int detID);
  const Range get(unsigned int detID) const;
  const std::vector<unsigned int> detIDs() const;
  void putclear(Range input, unsigned int detID);
  void clear();
  
 private:
  std::vector<ClusterFP420> container_;
  Registry map_;

  ClusterFP420Container clusterMap_; 
};

#endif // 


