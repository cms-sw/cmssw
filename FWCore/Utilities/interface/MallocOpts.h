#ifndef MallocOpts_h
#define MallocOpts_h

// -*- C++ -*-
//
// Package:     Utilities
// Class  :     MallocOpts
// 
// Original Author:  Jim Kowalkowski
// $Id$
//
// ------------------ malloc option setter -----------------------
//
// There is a global instance of MallocOptionSetter.  Upon construction,
// it gets the CPU type.  If is it AMD or Intel, it sets the mallopt
// parameters to a set that works best for that environment.  The best
// values have been chosen based on running a simple cmsRun job.
//
// The four values that get reset are:
//   M_MMAP_MAX, M_TRIM_THRESHOLD, M_TOP_PAD, M_MMAP_THRESHOLD
//
// Current the best AMD and Intel values were calculated using:
//   AMD Opteron(tm) Processor 248
//   Intel Dual Core something or other
//
// These values will need to be checked when new CPUs are available or
// when we move to 64 bit executables.

#include <string>
#include <ostream>

namespace edm
{
  struct MallocOpts
  {
    typedef int opt_type;
    
    MallocOpts():
      mmap_max_(),trim_thr_(),top_pad_(),mmap_thr_()
    {}
    MallocOpts(opt_type max,opt_type trim,opt_type pad,opt_type mmap_thr):
      mmap_max_(max),trim_thr_(trim),top_pad_(pad),mmap_thr_(mmap_thr)
    {}
    
    opt_type mmap_max_;
    opt_type trim_thr_;
    opt_type top_pad_;
    opt_type mmap_thr_;

    bool operator==(const MallocOpts& opts) const 
    {
      return
	mmap_max_ == opts.mmap_max_ && 
	trim_thr_ == opts.trim_thr_ && 
	top_pad_ == opts.top_pad_ &&
	mmap_thr_ == opts.mmap_thr_;
    }
    bool operator!=(const MallocOpts& opts) const
    { return !operator==(opts); }
  };

  std::ostream& operator<<(std::ostream& ost,const MallocOpts&);

  class MallocOptionSetter
  {
  public:
    typedef MallocOpts::opt_type opt_type;
    MallocOptionSetter();

    bool retrieveFromCpuType(); 
    bool retrieveFromEnv();
    void adjustMallocParams();
    bool hasErrors() const { return !error_message_.empty(); }
    std::string error_message() const { return error_message_; }

    void set_mmap_max(opt_type mmap_max)
    { values_.mmap_max_=mmap_max; changed_=true; }
    void set_trim_thr(opt_type trim_thr)
    { values_.trim_thr_=trim_thr; changed_=true; }
    void set_top_pad(opt_type top_pad)
    { values_.top_pad_=top_pad; changed_=true; }
    void set_mmap_thr(opt_type mmap_thr)
    { values_.mmap_thr_=mmap_thr; changed_=true; }

    MallocOpts get() const { return values_; }

  private:
    bool changed_;
    MallocOpts values_;

    std::string error_message_;
  };

  MallocOptionSetter& getGlobalOptionSetter();

}



#endif
