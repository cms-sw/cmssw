#ifndef DeepCopyPointer_H
#define DeepCopyPointer_H

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
  
  DeepCopyPointer() : theData(0) {}

  DeepCopyPointer( T* t) : theData(t) {}

  DeepCopyPointer( const DeepCopyPointer& other) {
    if (other.theData) theData = new T( *other); else theData = 0;
  }

  ~DeepCopyPointer() { delete theData;}

  DeepCopyPointer& operator=( const DeepCopyPointer& other) {
    if ( theData != other.theData) {
      delete theData;
      if (other.theData) theData = new T( *other); else theData = 0;
    }
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
