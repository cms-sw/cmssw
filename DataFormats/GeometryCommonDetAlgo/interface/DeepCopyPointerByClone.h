#ifndef DeepCopyPointerByClone_H
#define DeepCopyPointerByClone_H

#include <algorithm>
#include <cassert>

/** Same as DeepCopyPointer, except that it copies the object
 *  pointed to using the clone() virtual copy constructor.
 */

template <class T>
class DeepCopyPointerByClone {
public:
  ~DeepCopyPointerByClone() { delete theData; }
  DeepCopyPointerByClone() : theData(nullptr) {}

  DeepCopyPointerByClone(T* t) : theData(t) {}

  DeepCopyPointerByClone(const DeepCopyPointerByClone& other) {
    if (other.theData)
      theData = other->clone();
    else
      theData = nullptr;
  }

  DeepCopyPointerByClone& operator=(const DeepCopyPointerByClone& other) {
    if (theData != other.theData) {
      delete theData;
      if (other.theData)
        theData = other->clone();
      else
        theData = 0;
    }
    return *this;
  }

  // straight from http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2006/n2027.html
  DeepCopyPointerByClone(DeepCopyPointerByClone&& other) : theData(other.theData) { other.theData = nullptr; }
  DeepCopyPointerByClone& operator=(DeepCopyPointerByClone&& other) {
    std::swap(theData, other.theData);
    return *this;
  }

  T& operator*() {
    assert(theData);
    return *theData;
  }
  const T& operator*() const {
    assert(theData);
    return *theData;
  }

  T* operator->() {
    assert(theData);
    return theData;
  }
  const T* operator->() const {
    assert(theData);
    return theData;
  }

  /// to allow test like " if (p) {...}"
  operator bool() const { return theData != 0; }

  /// to allow test like " if (p == &someT) {...}"
  bool operator==(const T* otherP) const { return theData == otherP; }

private:
  T* theData;
};

#endif  // DeepCopyPointerByClone_H
