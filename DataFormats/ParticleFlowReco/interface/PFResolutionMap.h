#ifndef DataFormats_ParticleFlowReco_PFResolutionMap_h
#define DataFormats_ParticleFlowReco_PFResolutionMap_h

#include <iostream>

#include <TH2.h>



/// \brief Resolution Map (resolution as a function of eta and E)
///
/// Basically just a TH2D with text I/O
/// \author Colin Bernet
/// \todo extrapolation
/// \date January 2006

namespace reco {
  
  class PFResolutionMap : public TH2D {
  
  private:

  public:
    static const unsigned lineSize_;

    /// default constructor
    PFResolutionMap() : TH2D() {}

    /// create a map from text file mapfile
    PFResolutionMap(const char* name, const char* mapfile) throw(std::string) ;

    /// create an empty map and initialize it 
    PFResolutionMap(const char* name, 
		    unsigned nbinseta, double mineta, double maxeta,
		    unsigned nbinse, double mine, double maxe, double value=-1);

    /// create a map from a 2d histogram
    PFResolutionMap(const TH2D& h) : TH2D(h) {}
 

    /// read text file
    bool ReadMapFile(const char* mapfile);

    /// write text file
    bool WriteMapFile(const char* mapfile) const;

    ///  extrapolation requires overloading of this function
    int  FindBin(double eta, double e);

    /// print this map
    friend std::ostream& operator<<(std::ostream& out, const PFResolutionMap& rm);
  
  };
}
#endif
