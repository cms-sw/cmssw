#ifndef SIMPLEPROFILER
#define SIMPLEPROFILER 1

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 199309
#endif

#include <vector>
#include <string>
#include "boost/thread/mutex.hpp"
#include <pthread.h>

class SimpleProfiler
{
 public:
  static SimpleProfiler* instance();

  void start();
  void stop();

  unsigned int* stackTop() { return stacktop_; }
  void** tempStack() { return &tmp_stack_[0]; }
  void commitFrame(void** first, void** last);

  typedef std::vector<void*> VoidVec;
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
  unsigned int* stacktop_;
};

#endif
