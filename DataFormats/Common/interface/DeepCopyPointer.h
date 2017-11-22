#ifndef DeepCopyPointer_H
#define DeepCopyPointer_H


#include<algorithm>

/** A "smart" pointer that implements deep copy and ownership.
 *  In other words, when the pointer is copied it copies the 
 *  object it points to using "new". It also deletes the
 *  object it points to when it is deleted.
 *  Very useful for use as private data member of a class:
 *  it handles the copy construction, assignment, and destruction.
 */

template <class T>
class DeepCopyPointer {
public:
  
  ~DeepCopyPointer() { delete theData;}

  DeepCopyPointer() : theData(0) {}

  DeepCopyPointer( T* t) : theData(t) {}

  DeepCopyPointer( const DeepCopyPointer& other) {
    if (other.theData) theData = new T( *other); else theData = 0;
  }

  DeepCopyPointer& operator=( const DeepCopyPointer& other) {
    if ( theData != other.theData) {
      delete theData;
      if (other.theData) theData = new T( *other); else theData = 0;
    }
    return *this;
  }

  // straight from http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2006/n2027.html
  DeepCopyPointer( DeepCopyPointer&& other) : theData(other.theData) {
    other.theData=0;
  }
  DeepCopyPointer& operator=( DeepCopyPointer&& other) {
    std::swap(theData,other.theData);
    return *this;
  }


  /// Assing a new bare pointer to this DeepCopyPointer, taking ownership of it.
  /// The old content of this DeepCopyPointer is deleted 
  void replaceWith(T * otherP) {
    if ( theData != otherP ) {
        delete theData;
        theData = otherP;
    }
  }

  // assume that the replacement object is of the very same class!
  // at the moment all the work is done by the client i.e.
  // call the distructor
  // new in place
  // with c++0X a templated method can encasulate it all here... 
  T* replaceInplace() {
    return theData;
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

#endif // DeepCopyPointer_H
