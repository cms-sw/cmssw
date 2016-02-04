#ifndef _CombinationGenerator_H_
#define _CombinationGenerator_H_

#include "CommonTools/Statistics/interface/PartitionGenerator.h"
#include <vector>
#include <stack>
#include <algorithm>
#include <functional>

/** Class to compute all distinct Combinations 
 *  of a collection 'data' of objects of type 'T'. 
 *  A Combination is a set of collections, each collection 
 *  containing one or more objects, with any object in 'data' 
 *  assigned to exactly one collection. 
 */


template<class T>
class CombinationGenerator {

public:

  typedef std::vector<T> Collection;

  typedef std::vector<Collection> Combination;
  typedef std::vector<Combination> VectorOfCombinations;

  /** Create combinations obtained by dividing 'data' 
   *  according to all partitions with 'numberOfCollections' collections. 
   */
  std::vector<Combination> 
  combinations(const Collection & data, int numberOfCollections) const 
  {
    std::vector<Combination> combinations;
    Collection sorted = data;

    // Sort if needed
    if (prev_permutation(sorted.begin(), sorted.end())) {
      sort(sorted.begin(), sorted.end());
    }

    // Create sorted partitions
    PartitionGenerator aPartitionGenerator;
    std::vector< std::vector<PartitionGenerator::Partition> > partitions 
      = aPartitionGenerator.sortedPartitions(data.size());
    
    // Use partitions of size 'numberOfCollections' only
    for (std::vector<PartitionGenerator::Partition>::const_iterator idiv 
	   = partitions[numberOfCollections-1].begin(); 
	 idiv != partitions[numberOfCollections-1].end(); idiv++) {

      const PartitionGenerator::Partition& partition = *idiv;
      std::vector<Combination> subCombinations 
	= this->combinations(data, partition);
      for ( typename std::vector<Combination>::const_iterator iComb 
	     = subCombinations.begin(); 
	   iComb != subCombinations.end(); iComb++) {
	combinations.push_back(*iComb);
      }
    }
    return combinations;
  }


  /** Create all combinations obtained by dividing 'data' 
   *  according to Partition 'partition'. 
   */
  std::vector<Combination> 
  combinations(const Collection & data,
	       const PartitionGenerator::Partition & partition) const
  {

    std::vector<Combination> combinations;

    // Check that sum of collection sizes in 'partition' 
    // amounts to number of elements in 'data' 
    int nElts = 0;
    for (PartitionGenerator::Partition::const_iterator iSize 
	   = partition.begin(); iSize != partition.end(); iSize++) {
      nElts += *iSize;
    }
    if (nElts != data.size()) return combinations;

    Collection sortedData = data;
    // Sort if needed
    if (prev_permutation(sortedData.begin(), sortedData.end())) {
      sort(sortedData.begin(), sortedData.end());
    }

    // Create initial combination and put it on the stack
    Combination comb;
    comb.push_back(sortedData);
    if (partition.size() == 1) {
      // Return only one combination with only one collection
      combinations.push_back(comb);
      return combinations;
    }
    std::stack<Combination> cStack;
    cStack.push(comb);

    // Sort partitions by size 
    // Let 'sortedPartition' = ( n0, n1,... nk, nk+1,... nm ) 
    // Ordering is >= to speed up partitioning: n0 >= n1 >=... >= nm 
    PartitionGenerator::Partition sortedPartition = partition;
    sort(sortedPartition.begin(), sortedPartition.end(), std::greater_equal<int>());

    while (!cStack.empty()) {

      // 'combination' popped out of the stack 
      Combination combination = cStack.top(); cStack.pop();

      // At this stage 'combination' is made of collections 
      // of sizes ( n0+n1+...+nk, nk+1,... nm ) 
      // Now generate all combinations obtained by splitting 
      // the first collection of 'combination' in two, 
      // according to Partition 'biPartition' = (n0+n1+...+nk-1, nk) 
      int k = sortedPartition.size() - combination.size();
      int size = 0;
      for (int iColl = 0; iColl != k; iColl++) {
	size += sortedPartition[iColl];
      }
      PartitionGenerator::Partition biPartition(2);
      biPartition[0] = size; 
      biPartition[1] = sortedPartition[k];

      VectorOfCombinations subCombinations 
	= splitInTwoCollections(combination[0], biPartition[0]);
      for (typename std::vector<Combination>::const_iterator iComb = subCombinations.begin();
	   iComb != subCombinations.end(); iComb++) { 

	// Check ordering of successive bins of equal size 
	if (combination.size() > 1) {
	  if ((*iComb)[1].size() == combination[1].size()) {
	    Collection adjBins;
	    adjBins.push_back((*iComb)[1][0]);
	    adjBins.push_back(combination[1][0]);
	    // Drop 'combination' if successive bins of equal size not ordered 
	    // i.e. if previous permutation exists 
	    if (prev_permutation(adjBins.begin(), adjBins.end())) continue;
	  }
	}

	// Append remaining collections of 'combination'
	Combination merged = *iComb;
	typename Combination::const_iterator it = combination.begin(); it++;
	for ( ; it != combination.end(); it++) { merged.push_back(*it); }

	// Store combination 'merged' if partitioning is complete, 
	// otherwise put it on the stack for further partitioning. 
	if (merged.size() == sortedPartition.size()) {
	  combinations.push_back(merged);
	} else {
	  cStack.push(merged);
	}
      }
    }
    return combinations;
  }


  /** Create all combinations of elements from 'data'. 
   */
  std::vector<Combination> combinations(const Collection & data) const 
  {

    std::vector<Combination> combinations;
    Collection sorted = data;
    // Sort if needed
    if (prev_permutation(sorted.begin(), sorted.end())) {
      sort(sorted.begin(), sorted.end());
    }

    PartitionGenerator aPartitionGenerator;
    std::vector<PartitionGenerator::Partition> partitions 
      = aPartitionGenerator.partitions(data.size());
    
    for (std::vector<PartitionGenerator::Partition>::const_iterator idiv 
	   = partitions.begin(); idiv != partitions.end(); idiv++) {
      const PartitionGenerator::Partition& partition = *idiv;

      std::vector<Combination> subCombinations 
	= this->combinations(data, partition);
      for ( typename std::vector<Combination>::const_iterator iComb 
	     = subCombinations.begin(); 
	   iComb != subCombinations.end(); iComb++) {
	combinations.push_back(*iComb);
      }
    }
    return combinations;
  }


private:

  /** Create all combinations obtained by dividing 'data' 
   *  in two collections, the first one being of size 'sizeFirst' 
   */
  VectorOfCombinations splitInTwoCollections(const Collection & data, 
					    int sizeFirst) const
  {
    std::vector<Combination> combinations;
    std::stack<Combination> cStack;

    // Create first combination with 2 partitions
    Combination comb; comb.push_back(data); comb.push_back(Collection());
    cStack.push(comb);

    while (!cStack.empty()) {
      Combination combination = cStack.top();
      cStack.pop();

      Collection collection = combination[0];
      std::vector<Combination> subCombinations = separateOneElement(collection);
      
      for ( typename std::vector<Combination>::const_iterator iComb = subCombinations.begin();
	   iComb != subCombinations.end(); iComb++) {

	Collection second = combination[1];
	second.push_back((*iComb)[1][0]);

	// Abandon current combination if not ordered, 
	// i.e. if previous permutation exists 
	bool ordered = !prev_permutation(second.begin(), second.end());
	if (!ordered) continue;
	next_permutation(second.begin(), second.end());

	if ((*iComb)[0].size() == second.size()) {
	  Collection adjBins;
	  adjBins.push_back((*iComb)[0][0]);
	  adjBins.push_back(second[0]);
	  // Abandon current combination if successive bins of equal size 
	  // not ordered, i.e. if previous permutation exists 
	  if (prev_permutation(adjBins.begin(), adjBins.end())) continue;
	}

	Combination stored; 
	stored.push_back((*iComb)[0]);
	stored.push_back(second);

	if (stored[0].size() == sizeFirst) {
	  combinations.push_back(stored);
	} else {
	  cStack.push(stored);
	}
      }
    }
    return combinations;
  }

  /** Create all combinations obtained by dividing 'data' in two collections, 
   *  the second one having only one element. 
   */
  std::vector<Combination> separateOneElement(const Collection & data) const
  {
    std::vector<Combination> combinations;
    for ( typename Collection::const_iterator i = data.begin(); i != data.end(); i++) {
      Combination comb;
      Collection single; single.push_back(*i);
      Collection coll; 
      typename Collection::const_iterator j = data.begin();
      for ( ; j != i; j++) { coll.push_back(*j); }
      j++;
      for ( ; j != data.end(); j++) { coll.push_back(*j); }
      comb.push_back(coll); comb.push_back(single);
      combinations.push_back(comb);
    }
    return combinations;
  }

};

#endif
