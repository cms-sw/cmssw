#ifndef SequentialPartitionGenerator_H
#define SequentialPartitionGenerator_H

#include <vector>
#include <algorithm>

/**
 * Class to compute partitions of size k of an integer n.
 */
class SequentialPartitionGenerator {
public:
  typedef std::vector<int> Partition;

public:

  /***
   *   Generates a (number-theoretic) partition of n into k partitions,
   *   the invidual "partitions" being between pmin and pmax.
   */
  SequentialPartitionGenerator(int n, int k, int pmin=1 );
  SequentialPartitionGenerator(int n, int k, int pmin, int pmax );
  /**
   *  Get the next partition, in a well-defined series of
   *  partition
   */
  Partition next_partition();

private:
  int the_n;
  int the_k;
  int the_pmin;
  int the_pmax;
  Partition the_part;
  mutable int n_first;
  mutable int n_next;

private:
  bool first_part(Partition& p, int k, int n, int pmin, int pmax) const;
  bool next_part(Partition& p) const;
};

#endif
