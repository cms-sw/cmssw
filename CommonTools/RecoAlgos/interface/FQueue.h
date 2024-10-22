#ifndef COMMONTOOLS_RECOALGOS_FQUEUE_H
#define COMMONTOOLS_RECOALGOS_FQUEUE_H

#include <vector>

template <class T>
class FQueue {
public:
  FQueue() {
    theSize = 0;
    theFront = 0;
    theTail = 0;
    theCapacity = 0;
  }

  FQueue(unsigned int initialCapacity) {
    theBuffer.resize(initialCapacity);
    theSize = 0;
    theFront = 0;
    theTail = 0;
    theCapacity = initialCapacity;
  }

  unsigned int size() const { return theSize; }

  bool empty() const { return theSize == 0; }

  T front() const { return theBuffer[theFront]; }

  T& tail() { return theBuffer[theTail]; }

  constexpr unsigned int wrapIndex(unsigned int i) { return i & (theBuffer.size() - 1); }

  void push_back(const T& value) {
    if (theSize >= theCapacity) {
      theBuffer.resize(theCapacity * 2);
      if (theFront != 0) {
        std::copy(theBuffer.begin(), theBuffer.begin() + theTail, theBuffer.begin() + theCapacity);

        theTail += theSize;

      } else {
        theTail += theCapacity;
      }
      theCapacity *= 2;
    }
    theBuffer[theTail] = value;
    theTail = wrapIndex(theTail + 1);
    theSize++;
  }

  void pop_front() {
    if (theSize > 0) {
      theFront = wrapIndex(theFront + 1);
      theSize--;
    }
  }

  void pop_front(const unsigned int numberOfElementsToPop) {
    unsigned int elementsToErase = theSize > numberOfElementsToPop ? numberOfElementsToPop : theSize;
    theSize -= elementsToErase;
    theFront = wrapIndex(theFront + elementsToErase);
  }

  void reserve(unsigned int capacity) { theBuffer.reserve(capacity); }

  T& operator[](unsigned int index) { return theBuffer[wrapIndex(theFront + index)]; }

  void clear() {
    theBuffer.clear();
    theSize = 0;
    theFront = 0;
    theTail = 0;
  }

private:
  unsigned int theSize;
  unsigned int theFront;
  unsigned int theTail;
  std::vector<T> theBuffer;
  unsigned int theCapacity;
};

#endif
