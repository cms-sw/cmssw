#ifndef FWCore_Services_SimpleProfiler_h
#define FWCore_Services_SimpleProfiler_h

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 199309
#endif

#include <vector>
#include <string>
#include "boost/thread/mutex.hpp"

class SimpleProfiler
{
 public:
  typedef std::vector<void*> VoidVec;
  typedef VoidVec::size_type size_type;

  static SimpleProfiler* instance();

  void start();
  void stop();

  void* stackTop() { return stacktop_; }
  void** tempStack() { return &tmp_stack_[0]; }
  size_type tempStackSize() { return tmp_stack_.size(); }
  
  void commitFrame(void** first, void** last);

 private:
  SimpleProfiler();
  ~SimpleProfiler();

  void complete();
  void doWrite();

  static SimpleProfiler* inst_;
  static boost::mutex lock_;
  VoidVec frame_data_;
  VoidVec tmp_stack_;
  void** high_water_;
  void** curr_;
  std::string filename_;
  int fd_;
  bool installed_;
  bool running_;
  pthread_t owner_;
  void* stacktop_;
};

#endif
