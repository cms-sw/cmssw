#ifndef DataFormats_ParticleFlowReco_PFResolutionMap_h
#define DataFormats_ParticleFlowReco_PFResolutionMap_h

#include <iostream>
#include <string>
#include <stdexcept>

#include <TH2.h>



/// \brief Resolution Map (resolution as a function of eta and E)
///
/// Basically just a TH2D with text I/O
/// \author Colin Bernet
/// \todo extrapolation
/// \date January 2006
class PFResolutionMap : public TH2D {

 public:

  /// default constructor
  PFResolutionMap() : TH2D() {}
  
  /// create a map from text file mapfile
  PFResolutionMap(const char* name, const char* mapfile);
  
  /// create an empty map and initialize it 
  PFResolutionMap(const char* name, 
		  unsigned nbinseta, double mineta, double maxeta,
		  unsigned nbinse, double mine, double maxe, double value=-1);
  
  /// create a map from a 2d histogram
  PFResolutionMap(const TH2D& h) : TH2D(h) {}
 

  /// read text file
  bool ReadMapFile(const char* mapfile);

  /// write text file
  /// is not const because mapFile_ will be updated
  bool WriteMapFile(const char* mapfile);

  ///  extrapolation requires overloading of this function
  int  FindBin(double eta, double e, double z = 0 ) override;

  double getRes(double eta, double phi, double e,int MapEta = -1); 

  const char* GetMapFile() const {return mapFile_.c_str();}

  /// print this map
  friend std::ostream& operator<<(std::ostream& out, const PFResolutionMap& rm);
  
 private:
  bool IsInAPhiCrack(double phi, double eta);
  double minimum(double a,double b);
  double dCrackPhi(double phi, double eta);
  static const unsigned lineSize_;
  std::string           mapFile_;
};

#endif
