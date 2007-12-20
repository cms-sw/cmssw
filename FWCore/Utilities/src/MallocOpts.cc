

// -*- C++ -*-
//
// Package:     Utilities
// Class  :     MallocOpts
// 
// Original Author:  Jim Kowalkowski
// $Id$
//
// ------------------ resetting malloc options -----------------------


#include "FWCore/Utilities/interface/MallocOpts.h"

#include <malloc.h>
#include <sstream>
#include <iostream>
#include <string>

namespace edm
{

  std::ostream& operator<<(std::ostream& ost,const MallocOpts& opts)
  {
    ost << "mmap_max=" << opts.mmap_max_
	<< " trim_threshold=" << opts.trim_thr_
	<< " top_padding=" << opts.top_pad_
	<< " mmap_threshold=" << opts.mmap_thr_;
    return ost;
  }

  namespace 
  {
    typedef enum { UNKNOWN_CPU=0, AMD_CPU=1, INTEL_CPU=2 } cpu_type;
  
    cpu_type get_cpu_type()
    {
      int op=0,a;
      int ans[4];
    
#if defined(__x86_64__) || defined(__i386__)


      __asm__ __volatile__ ("pushl %%ebx;\
 pushl %%edx;				 \
 pushl %%ecx;				 \
 cpuid;					 \
 movl  %%ecx,%0;			 \
 movl  %%edx,%1;			 \
 movl  %%ebx,%2;			 \
 movl  %%eax,%3;			 \
 popl  %%ecx;				 \
 popl  %%edx;				 \
 popl  %%ebx;"
                            : "=m"(ans[2]), "=m"(ans[1]), "=m"(ans[0]), "=m"(a)
                            : "a"(op)
			    );

    
#else
      const char* unknown_str = "Unknown";
      // int unknown_sz = strlen(unknown_str);
      strcpy((char*)&ans[0],unknown_str);
#endif
      
      const char* amd_str = "AuthenticAMD";
      int amd_sz = strlen(amd_str);
      const char* intel_str = "GenuineIntel";
      int intel_sz = strlen(intel_str);
    
      char* str = (char*)&ans[0];
      ans[3]=0;

      return strncmp(str,amd_str,amd_sz)==0?AMD_CPU:
	strncmp(str,intel_str,intel_sz)==0?INTEL_CPU:UNKNOWN_CPU;
    }

    // values determined experimentally for each architecture
    const MallocOpts intel_opts(262144, 524288, 5242880, 131072);
    const MallocOpts amd_opts(0, 8388608, 131072, 10485760);

  }


  bool MallocOptionSetter::retrieveFromCpuType()
  {
    bool rc=true;

    switch(get_cpu_type())
      {
      case AMD_CPU:
	{
	  values_ = amd_opts;
	  changed_=true;
	  break;
	}
      case INTEL_CPU:
	{
	  values_ = intel_opts;
	  changed_=true;
	  break;
	}
      case UNKNOWN_CPU:
      default:
	rc=false;
      }

    return rc;
  }

  MallocOptionSetter::MallocOptionSetter():
    changed_(false)
  {
    if(retrieveFromEnv() || retrieveFromCpuType())
      {
	adjustMallocParams();
	if(hasErrors())
	  {
	    std::cerr << "ERROR: Reset of malloc options has fails:\n"
		      << error_message_ << "\n";
	  }
      }
  }

  void MallocOptionSetter::adjustMallocParams()
  {
    if(changed_==false) return; // only adjust if they changed
    error_message_.clear();
    changed_ = false;

    if(mallopt(M_MMAP_MAX,values_.mmap_max_)<0)
      error_message_ += "Could not set M_MMAP_MAX\n"; 
    if(mallopt(M_TRIM_THRESHOLD,values_.trim_thr_)<0)
      error_message_ += "Could not set M_TRIM_THRESHOLD\n"; 
    if(mallopt(M_TOP_PAD,values_.top_pad_)<0)
      error_message_ += "ERROR: Could not set M_TOP_PAD\n";
    if(mallopt(M_MMAP_THRESHOLD,values_.mmap_thr_)<0)
      error_message_ += "ERROR: Could not set M_MMAP_THRESHOLD\n";
  }

  bool MallocOptionSetter::retrieveFromEnv()
  {
    const char* par = getenv("CMSRUN_MALLOC_RESET");
    if(par==0) return false; // leave quickly here
    std::string spar(par);
    bool rc = false;
      
    // CMSRUN_MALLOC_RESET = "mmap_max trim_thres top_pad mmap_thres"
      
    if(spar.size()>1)
      {
	std::istringstream ist(spar);
	ist >> values_.mmap_max_ >> values_.trim_thr_
	    >> values_.top_pad_ >> values_.mmap_thr_;

	if(ist.bad())
	  {
	    std::cerr << "bad malloc options in CMSRUN_MALLOC_RESET: "
		      << spar << "\n"
		      << "format is: "
		      << "CMSRUN_MALLOC_RESET=\"mmap_max trim_thres top_pad mmap_thres\"\n";
	  }
	else
	  {
	    std::cout << "MALLOC_OPTIONS> Reset options: "
		      << "CMSRUN_MALLOC_RESET=" << par << "\n";
	  }
	rc=true;
	changed_=true;
      }

    return rc;
  }

  MallocOptionSetter global_malloc_options;

  MallocOptionSetter& getGlobalOptionSetter()
  {
    return global_malloc_options;
  }


  // ----------------------------------------------------------------
}
