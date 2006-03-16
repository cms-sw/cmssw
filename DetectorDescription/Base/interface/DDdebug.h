#ifndef DDdebug_h
#define DDdebug_h

// Message logger.
#include "FWCore/MessageLogger/interface/MessageLogger.h"

 // If DDEBUG is defined, the debug macros DCOUT and DCOUT_V are active
 // Undefined DDEBUG redefines DCOUT and DCOUT_V to not to produce any code.
#define DDEBUG 
 
 #ifdef DDEBUG
  // Extended debug macros: DDEBUG(a,b) and DDEBUG_V(a,b) for extended verbosity
  // Both macros are kind of CPU intensive (calls to getenv, loop over a std::string)
  //
  // Usage: DEBUGOUT(char, std::ostream )
  //  char (i.e. 'P') defines the module which should produce debug-output
  //  At runtime the environment variables DDEBUG and  DDEBUG_V specify which
  //  module should activate its debug-output.
  //
  //  Example:
  //   file DDCore/src/a.cc: DCOUT('C',"Debuginfo of moule C" << i_ << j_ );
  //   file Parser/src/b.cc: DCOUT('P',"Parser debug");
  //   runtime: setenv DDEBUG P  > activates debug-output of a.cc
  //            setenv DDEBUG C  > activates debug-output of b.cc
  //            setenv DDEBUG PC > activated debug-output of a.cc and b.cc  
  

  #include <cstdlib>
  #include <iostream>
  
      
  #define DCOUT(M_v_Y,M_v_S) if (char* M_v_c = getenv("DDEBUG")){ std::string M_v_t(M_v_c); for(std::string::size_type M_v_i=0; M_v_i < M_v_t.size(); M_v_i++) if(M_v_t[M_v_i]==M_v_Y) LogDebug("DDdebug") << M_v_t[M_v_i] << " : " << M_v_S << std::endl; }   
  #define DCOUT_V(M_v_Y,M_v_S) if (char* M_v_c = getenv("DDEBUG_V")){ std::string M_v_t(M_v_c); for(std::string::size_type M_v_i=0; M_v_i < M_v_t.size(); M_v_i++) if(M_v_t[M_v_i]==M_v_Y) LogDebug("DDdebug") << M_v_t[M_v_i] << "v: " << M_v_S << std::endl; }         
  
  // backward compatiblility, don't use anymore! 
  #define DEBUGOUT(s) if (getenv("DDEBUG")) { LogDebug("DDdebug") << s << std::endl; }
  #define DEBUGOUT_V(s) if (getenv("DDEBUG_V")) { LogDebug("DDdebug") << s << std::endl; }
 
 #else
  
  #define DCOUT(M_v_Y, M_v_s)
  #define DCOUT_V(M_v_Y, M_v_s)
  #define DEBUGOUT(s)
  #define DEBUGOUT_V(s)
 
 #endif

/** only for LINUX
  returns the size of the running program in kbytes
*/    
int DDmem();
int DDtime();

#endif
