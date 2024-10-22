#ifndef RecoParticleFlow_PFProducer_FlaggedPtr_h
#define RecoParticleFlow_PFProducer_FlaggedPtr_h

template <typename T>
class FlaggedPtr {
public:
  FlaggedPtr(T* pointer, bool flag) : pointer_(pointer), flag_(flag) {}
  T& operator*() const { return *pointer_; }
  T* operator->() const { return pointer_; }
  T* get() const { return pointer_; }
  bool flag() const { return flag_; }
  void setFlag(bool flag) { flag_ = flag; }

private:
  T* pointer_;
  bool flag_;
};

#endif
