#ifndef L1Trigger_TrackFindingTracklet_interface_CircularBuffer_h
#define L1Trigger_TrackFindingTracklet_interface_CircularBuffer_h

#include <cassert>
#include <vector>

namespace trklet {

  template <class T>
  class CircularBuffer {
  public:
    CircularBuffer(unsigned int nbits) {
      size_ = 1 << nbits;
      buffer_.resize(size_);
      reset();
    }

    ~CircularBuffer() = default;

    void reset() {
      rptr_ = 0;
      wptr_ = 0;
    }

    //Full if writer ptr incremented is same as read ptr
    bool full() const { return ((wptr_ + 1) % size_) == rptr_; }

    //Almost full if writer ptr incremented by 1 or 2 is same as read ptr
    bool almostfull() const { return (((wptr_ + 1) % size_) == rptr_) || (((wptr_ + 2) % size_) == rptr_); }

    //near full if writer ptr incremented by 1, 2, or 3 is same as read ptr
    bool nearfull() const {
      return (((wptr_ + 1) % size_) == rptr_) || (((wptr_ + 2) % size_) == rptr_) || (((wptr_ + 3) % size_) == rptr_);
    }

    //Empty buffer is write ptr is same as read ptr
    bool empty() const { return wptr_ == rptr_; }

    const T& read() {
      assert(!empty());
      unsigned int oldrptr = rptr_;
      rptr_ = (rptr_ + 1) % size_;
      return buffer_[oldrptr];
    }

    const T& peek() const {
      assert(!empty());
      return buffer_[rptr_];
    }

    void store(T element) {
      assert(!full());
      buffer_[wptr_++] = element;
      wptr_ = wptr_ % size_;
      assert(wptr_ != rptr_);
    }

    //these are needed for comparison of emulation with HLS FW
    unsigned int rptr() const { return rptr_; }
    unsigned int wptr() const { return wptr_; }

  private:
    std::vector<T> buffer_;

    //buffer size
    unsigned int size_;

    //read and write poiters into buffer
    unsigned int rptr_;
    unsigned int wptr_;
  };
};  // namespace trklet
#endif
