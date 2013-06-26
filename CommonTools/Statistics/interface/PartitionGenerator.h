#ifndef _PartitionGenerator_H_
#define _PartitionGenerator_H_

#include <vector>


/** Class to compute the possible partitions of a collection of 
 *  'collectionSize' elements. A Partition is a list of 
 *  subcollection sizes that add up to 'collectionSize', 
 *  ordered in decreasing order of subcollection size. 
 *  No subcollection size is less than 'minCollectionSize'. 
 */ 
class PartitionGenerator {

public:

  typedef std::vector<int> Partition;

  /** Return partitions in a row. 
   */
  std::vector<Partition> partitions(int collectionSize, 
			       int minCollectionSize = 1) const;

  /** Return partitions ordered according to the number of 
   *  subcollections they have. 
   *  'sortedPartitions[N]' = list of partitions with N subcollections. 
   */
  std::vector< std::vector<Partition> > 
  sortedPartitions(int collectionSize, int minCollectionSize = 1) const;

private:
  /** a private class just defining the () operator */
  class LessCollections {
  public:
    bool operator()(const Partition & a, 
		    const Partition & b) {
      return a.size() < b.size();
    }
  };

};

#endif
