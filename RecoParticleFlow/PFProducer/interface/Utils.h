#ifndef RecoParticleFlow_PFProducer_Utils_h
#define RecoParticleFlow_PFProducer_Utils_h

#include <string>
#include <vector>
#include <glob.h>


/// \brief Utility functions
///
/// \author Colin Bernet
/// \date January 2006



class Utils {

 public:
  
  /// match a string to a regexp 
  static bool     stringMatch(const char* string, const char* pattern);

  /// get all files matching pattern
  static std::vector<std::string>  myGlob(const char* pattern);

  /// returns the date
  static std::string   date();

  /// returns angle between mpi and pi
  static double mpi_pi(double angle);
};

#endif


