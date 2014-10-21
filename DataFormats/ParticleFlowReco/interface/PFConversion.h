#ifndef DataFormats_ParticleFlowReco_PFConversion_h
#define DataFormats_ParticleFlowReco_PFConversion_h


#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrackFwd.h"

#include <iostream>
#include <vector>

class Conversion;

namespace reco {

  /**\class PFConversion
     \author  Nancy Marinelli - Univ. of Notre Dame
     \date   
  */
  class PFConversion {
  public:


 // Default constructor
    PFConversion() {}
  

    //    PFConversion(const reco::ConversionRef c);
    // PFConversion(const reco::ConversionRef c, const std::vector<reco::PFRecTrackRef>&  tr   );

    PFConversion( reco::ConversionRef c);
    PFConversion( const reco::ConversionRef& c, const std::vector<reco::PFRecTrackRef>&  tr   );


    /// destructor
    ~PFConversion();

    const reco::ConversionRef& originalConversion() const   {return originalConversion_;} 
    const std::vector<reco::PFRecTrackRef>& pfTracks() const {return pfTracks_ ;} 
    

  private:

    void addPFTrack( const reco::PFRecTrackRef & tr ) { pfTracks_.push_back(tr); }    
    reco::ConversionRef originalConversion_;
    std::vector<reco::PFRecTrackRef>  pfTracks_;
    

  };

}

#endif
