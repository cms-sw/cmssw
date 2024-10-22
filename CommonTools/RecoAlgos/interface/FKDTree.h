#ifndef COMMONTOOLS_RECOALGOS_FKDTREE_H
#define COMMONTOOLS_RECOALGOS_FKDTREE_H

#include <vector>
#include <array>
#include <algorithm>
#include <cmath>
#include <utility>
#include "FKDPoint.h"
#include "FQueue.h"

// Author: Felice Pantaleo
// email: felice.pantaleo@cern.ch
// date: 08/05/2017
// Description: This class provides a k-d tree implementation targeting modern architectures.
// Building each level of the FKDTree can be done in parallel by different threads.
// It produces a compact array of nodes in memory thanks to the different space partitioning method used.

// Fast version of the integer logarithm
namespace {
  const std::array<unsigned int, 32> MultiplyDeBruijnBitPosition{{0,  9,  1,  10, 13, 21, 2,  29, 11, 14, 16,
                                                                  18, 22, 25, 3,  30, 8,  12, 20, 28, 15, 17,
                                                                  24, 7,  19, 27, 23, 6,  26, 5,  4,  31}};
  unsigned int ilog2(unsigned int v) {
    v |= v >> 1;  // first round down to one less than a power of 2
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return MultiplyDeBruijnBitPosition[(unsigned int)(v * 0x07C4ACDDU) >> 27];
  }
}  // namespace

template <class TYPE, unsigned int numberOfDimensions>
class FKDTree {
public:
  FKDTree() {
    theNumberOfPoints = 0;
    theDepth = 0;
  }

  bool empty() { return theNumberOfPoints == 0; }

  // One can search for all the points which are contained in a k-dimensional box.
  // Searching is done by providing the two k-dimensional points in the minimum and maximum corners.
  // The vector that will contain the indices of the points that lay inside the box is also needed.
  // Indices are pushed into foundPoints, which is not checked for emptiness at the beginning,
  // nor memory is reserved for it.
  // Searching is done using a Breadth-first search, level after level.
  void search(const FKDPoint<TYPE, numberOfDimensions>& minPoint,
              const FKDPoint<TYPE, numberOfDimensions>& maxPoint,
              std::vector<unsigned int>& foundPoints) {
    //going down the FKDTree, one needs track which indices have to be visited in the following level.
    //a custom queue is created, since std::queue is based on lists which are sometimes not performing
    // well on computing accelerators
    // The initial size of the queue has to be a power of two for allowing fast modulo  % operation.
    FQueue<unsigned int> indicesToVisit(16);

    //The root element is pushed first
    indicesToVisit.push_back(0);
    unsigned int index;
    bool intersection;
    unsigned int dimension;
    unsigned int numberOfindicesToVisitThisDepth;
    unsigned int numberOfSonsToVisitNext;
    unsigned int firstSonToVisitNext;

    //The loop over levels of the FKDTree starts here
    for (unsigned int depth = 0; depth < theDepth + 1; ++depth) {
      // At each level, comparisons are performed for a different dimension in round robin.
      dimension = depth % numberOfDimensions;
      numberOfindicesToVisitThisDepth = indicesToVisit.size();
      // Loop over the nodes to be visit at this level
      for (unsigned int visitedindicesThisDepth = 0; visitedindicesThisDepth < numberOfindicesToVisitThisDepth;
           visitedindicesThisDepth++) {
        index = indicesToVisit[visitedindicesThisDepth];
        // check if the element's dimension lays between the two box borders
        intersection = intersects(index, minPoint, maxPoint, dimension);
        firstSonToVisitNext = leftSonIndex(index);

        if (intersection) {
          // Check if the element is contained in the box and push it to the result
          if (is_in_the_box(index, minPoint, maxPoint)) {
            foundPoints.emplace_back(theIds[index]);
          }
          //if the element is between the box borders, both the its sons have to be visited
          numberOfSonsToVisitNext =
              (firstSonToVisitNext < theNumberOfPoints) + ((firstSonToVisitNext + 1) < theNumberOfPoints);
        } else {
          // depending on the position of the element wrt the box, one son will be visited (if it exists)
          firstSonToVisitNext += (theDimensions[dimension][index] < minPoint[dimension]);

          numberOfSonsToVisitNext =
              std::min((firstSonToVisitNext < theNumberOfPoints) + ((firstSonToVisitNext + 1) < theNumberOfPoints), 1);
        }

        // the indices of the sons to be visited in the next iteration are pushed in the queue
        for (unsigned int whichSon = 0; whichSon < numberOfSonsToVisitNext; ++whichSon) {
          indicesToVisit.push_back(firstSonToVisitNext + whichSon);
        }
      }
      // a new element is popped from the queue
      indicesToVisit.pop_front(numberOfindicesToVisitThisDepth);
    }
  }

  // A vector of K-dimensional points needs to be passed in order to build the kdtree.
  // The order of the elements in the vector will be modified.
  void build(std::vector<FKDPoint<TYPE, numberOfDimensions> >& points) {
    // initialization of the data members
    theNumberOfPoints = points.size();
    theDepth = ilog2(theNumberOfPoints);
    theIntervalLength.resize(theNumberOfPoints, 0);
    theIntervalMin.resize(theNumberOfPoints, 0);
    for (unsigned int i = 0; i < numberOfDimensions; ++i)
      theDimensions[i].resize(theNumberOfPoints);
    theIds.resize(theNumberOfPoints);

    // building is done by reordering elements in a partition starting at theIntervalMin
    // for a number of elements theIntervalLength
    unsigned int dimension;
    theIntervalMin[0] = 0;
    theIntervalLength[0] = theNumberOfPoints;

    // building for each level starts here
    for (unsigned int depth = 0; depth < theDepth; ++depth) {
      // A heapified left-balanced tree can be represented in memory as an array.
      // Each level contains a power of two number of elements and starts from element 2^depth -1
      dimension = depth % numberOfDimensions;
      unsigned int firstIndexInDepth = (1 << depth) - 1;
      unsigned int maxDepth = (1 << depth);
      for (unsigned int indexInDepth = 0; indexInDepth < maxDepth; ++indexInDepth) {
        unsigned int indexInArray = firstIndexInDepth + indexInDepth;
        unsigned int leftSonIndexInArray = 2 * indexInArray + 1;
        unsigned int rightSonIndexInArray = leftSonIndexInArray + 1;

        // partitioning is done by choosing the pivotal element that keeps the tree heapified
        // and left-balanced
        unsigned int whichElementInInterval = partition_complete_kdtree(theIntervalLength[indexInArray]);
        // the elements have been partitioned in two unsorted subspaces (one containing the elements
        // smaller than the pivot, the other containing those greater than the pivot)
        std::nth_element(points.begin() + theIntervalMin[indexInArray],
                         points.begin() + theIntervalMin[indexInArray] + whichElementInInterval,
                         points.begin() + theIntervalMin[indexInArray] + theIntervalLength[indexInArray],
                         [dimension](const FKDPoint<TYPE, numberOfDimensions>& a,
                                     const FKDPoint<TYPE, numberOfDimensions>& b) -> bool {
                           if (a[dimension] == b[dimension])
                             return a.getId() < b.getId();
                           else
                             return a[dimension] < b[dimension];
                         });
        // the element is placed in its final position in the internal array representation
        // of the tree
        add_at_position(points[theIntervalMin[indexInArray] + whichElementInInterval], indexInArray);
        if (leftSonIndexInArray < theNumberOfPoints) {
          theIntervalMin[leftSonIndexInArray] = theIntervalMin[indexInArray];
          theIntervalLength[leftSonIndexInArray] = whichElementInInterval;
        }

        if (rightSonIndexInArray < theNumberOfPoints) {
          theIntervalMin[rightSonIndexInArray] = theIntervalMin[indexInArray] + whichElementInInterval + 1;
          theIntervalLength[rightSonIndexInArray] = (theIntervalLength[indexInArray] - 1 - whichElementInInterval);
        }
      }
    }
    // the last level of the tree may not be complete and needs special treatment
    dimension = theDepth % numberOfDimensions;
    unsigned int firstIndexInDepth = (1 << theDepth) - 1;
    for (unsigned int indexInArray = firstIndexInDepth; indexInArray < theNumberOfPoints; ++indexInArray) {
      add_at_position(points[theIntervalMin[indexInArray]], indexInArray);
    }
  }
  // returns the number of points in the FKDTree
  std::size_t size() const { return theNumberOfPoints; }

private:
  // returns the index of the element which makes the FKDtree a left-complete heap
  // e.g.: if we have 6 elements, the tree will be shaped like
  //                 O
  //                / '\'
  //               O    O
  //              /'\' /
  //             O   OO
  //
  // This will return for a length of 6 the 4th element, which will partition the tree so that
  // 3 elements are on its left and 2 elements are on its right
  unsigned int partition_complete_kdtree(unsigned int length) {
    if (length == 1)
      return 0;
    unsigned int index = 1 << (ilog2(length));

    if ((index / 2) - 1 <= length - index)
      return index - 1;
    else
      return length - index / 2;
  }

  // returns the index of an element left son in the array representation
  unsigned int leftSonIndex(unsigned int index) const { return 2 * index + 1; }
  // returns the index of an element right son in the array representation
  unsigned int rightSonIndex(unsigned int index) const { return 2 * index + 2; }

  //check if one element's dimension is between minPoint's and maxPoint's dimension
  bool intersects(unsigned int index,
                  const FKDPoint<TYPE, numberOfDimensions>& minPoint,
                  const FKDPoint<TYPE, numberOfDimensions>& maxPoint,
                  int dimension) const {
    return (theDimensions[dimension][index] <= maxPoint[dimension] &&
            theDimensions[dimension][index] >= minPoint[dimension]);
  }

  // check if an element is completely in the box
  bool is_in_the_box(unsigned int index,
                     const FKDPoint<TYPE, numberOfDimensions>& minPoint,
                     const FKDPoint<TYPE, numberOfDimensions>& maxPoint) const {
    for (unsigned int i = 0; i < numberOfDimensions; ++i) {
      if ((theDimensions[i][index] <= maxPoint[i] && theDimensions[i][index] >= minPoint[i]) == false)
        return false;
    }
    return true;
  }

  // places an element at the specified position in the internal data structure
  void add_at_position(const FKDPoint<TYPE, numberOfDimensions>& point, const unsigned int position) {
    for (unsigned int i = 0; i < numberOfDimensions; ++i)
      theDimensions[i][position] = point[i];
    theIds[position] = point.getId();
  }

  void add_at_position(FKDPoint<TYPE, numberOfDimensions>&& point, const unsigned int position) {
    for (unsigned int i = 0; i < numberOfDimensions; ++i)
      theDimensions[i][position] = point[i];
    theIds[position] = point.getId();
  }

  unsigned int theNumberOfPoints;
  unsigned int theDepth;

  // a SoA containing all the dimensions for each point
  std::array<std::vector<TYPE>, numberOfDimensions> theDimensions;
  std::vector<unsigned int> theIntervalLength;
  std::vector<unsigned int> theIntervalMin;
  std::vector<unsigned int> theIds;
};

#endif
