#ifndef __Utils__
#define __Utils__

#include <string>
#include <vector>


class TCanvas;
class TVector3;

/// \brief Utility functions
///
/// \author Colin Bernet
/// \date January 2006
class Utils {

 public:
  
  /// match a string to a regexp 
  static bool     StringMatch(const char* string, const char* pattern);

  /// divide a TCanvas in a nice way
  static TCanvas* DivideCanvas( TCanvas *cv, int nPads );

  /// get all files matching pattern
  static std::vector<std::string>  Glob(const char* pattern);

  /// returns the date
  static std::string   Date();

  /// converts a vector (in eta,phi,R) to a vector in (x,y,z)
  static TVector3 VectorEPRtoXYZ( const TVector3& posepr );
};

#endif


