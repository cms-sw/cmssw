#include "CommonTools/Statistics/interface/PartitionGenerator.h"
#include <algorithm>

using namespace std;


vector<PartitionGenerator::Partition> 
PartitionGenerator::partitions(int collectionSize, int minCollectionSize) 
const {

  std::vector<Partition> partitions;

  // at the very least, we have a single bag of size 'collectionSize'
  partitions.push_back( Partition(1, collectionSize) );

  int first = collectionSize - minCollectionSize, second = minCollectionSize;
  while( first >= second ) {
    // try to further divide the first
    std::vector<Partition> subPartitions = this->partitions(first, second);
    std::vector<Partition>::iterator isub;
    for( isub = subPartitions.begin(); isub != subPartitions.end(); isub++ ) {
      const Partition& sub = *isub;
      // reject subPartitions of first with a last element smaller than second
      if( sub.back() < second ) continue;
      Partition partition( sub.size()+1 );
      copy( sub.begin(), sub.end(), partition.begin() );
      partition[ partition.size()-1 ] = second;
      partitions.push_back( partition );
    }
    first--; second++;
  }

  return partitions;
}


vector< std::vector<PartitionGenerator::Partition> > 
PartitionGenerator::sortedPartitions(int collectionSize, 
				     int minCollectionSize) const {

  std::vector<Partition> partitions 
    = this->partitions(collectionSize, minCollectionSize);
  sort (partitions.begin(), partitions.end(), LessCollections());
  
  std::vector< std::vector<Partition> > sortedPartitions;
  sortedPartitions.resize(partitions.rbegin()->size());

  for (std::vector<Partition>::const_iterator i = partitions.begin();
       i != partitions.end(); i++) {
    sortedPartitions[(*i).size() - 1].push_back(*i);
  }

  return sortedPartitions;
}
