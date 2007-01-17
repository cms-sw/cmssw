#ifndef DeepCopyPointerByClone_H
#define DeepCopyPointerByClone_H

/** Same as DeepCopyPointer, except that it copies the object
 *  pointed to wsing the clone() virtual copy constructor.
 */

template <class T>
class DeepCopyPointerByClone {
public:
  
  DeepCopyPointerByClone() : theData(0) {}

  DeepCopyPointerByClone( T* t) : theData(t) {}

  DeepCopyPointerByClone( const DeepCopyPointerByClone& other) {
    if (other.theData) theData = other->clone(); else theData = 0;
  }

  ~DeepCopyPointerByClone() { delete theData;}

  DeepCopyPointerByClone& operator=( const DeepCopyPointerByClone& other) {
    if ( theData != other.theData) {
      delete theData;
      if (other.theData) theData = other->clone(); else theData = 0;
    }
    return *this;
  }

  T&       operator*()       { return *theData;}
  const T& operator*() const { return *theData;}

  T*       operator->()       { return theData;}
  const T* operator->() const { return theData;}

  /// to allow test like " if (p) {...}"
  operator bool() const { return theData != 0;}

  /// to allow test like " if (p == &someT) {...}"
  bool operator==( const T* otherP) const { return theData == otherP;}

private:
  T* theData;
};

#endif // DeepCopyPointerByClone_H
