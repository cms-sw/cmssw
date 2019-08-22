#ifndef COMMONTOOLS_RECOALGOS_FKDPOINT_H
#define COMMONTOOLS_RECOALGOS_FKDPOINT_H
#include <array>
#include <utility>

//a K-dimensional point to interface with the FKDTree class
template <class TYPE, int numberOfDimensions>
class FKDPoint {
public:
  FKDPoint() : theElements(), theId(0) {}

  FKDPoint(TYPE x, TYPE y, unsigned int id = 0) {
    static_assert(numberOfDimensions == 2, "FKDPoint number of arguments does not match the number of dimensions");

    theId = id;
    theElements[0] = x;
    theElements[1] = y;
  }

  FKDPoint(TYPE x, TYPE y, TYPE z, unsigned int id = 0) {
    static_assert(numberOfDimensions == 3, "FKDPoint number of arguments does not match the number of dimensions");

    theId = id;
    theElements[0] = x;
    theElements[1] = y;
    theElements[2] = z;
  }

  FKDPoint(TYPE x, TYPE y, TYPE z, TYPE w, unsigned int id = 0) {
    static_assert(numberOfDimensions == 4, "FKDPoint number of arguments does not match the number of dimensions");
    theId = id;
    theElements[0] = x;
    theElements[1] = y;
    theElements[2] = z;
    theElements[3] = w;
  }

  // the user should check that i < numberOfDimensions
  TYPE& operator[](unsigned int const i) { return theElements[i]; }

  TYPE const& operator[](unsigned int const i) const { return theElements[i]; }

  void setDimension(unsigned int i, const TYPE& value) { theElements[i] = value; }

  void setId(const unsigned int id) { theId = id; }

  unsigned int getId() const { return theId; }

private:
  std::array<TYPE, numberOfDimensions> theElements;
  unsigned int theId;
};

#endif
