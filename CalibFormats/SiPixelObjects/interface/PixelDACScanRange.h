#ifndef PixelDACScanRange_h
#define PixelDACScanRange_h
/*! \file CalibFormats/SiPixelObjects/interface/PixelConfigurationVerifier.h
*   \brief This class collects the information about the range of DAC settings used in scans of the DACs
*
*   A longer explanation will be placed here later
*/
//
// This class collects the information
// about the range of DAC settings used
// in scans of the DACs.
//
//
//

#include <string>

namespace pos{
/*! \class PixelConfigurationVerifier PixelConfigurationVerifier.h "interface/PixelConfigurationVerifier.h"
*   \brief This class collects the information about the range of DAC settings used in scans of the DACs
*
*   A longer explanation will be placed here later
*/
  class PixelDACScanRange {

  public:

    PixelDACScanRange(){;}
    PixelDACScanRange(std::string dacname, unsigned int first, 
		      unsigned int last, unsigned int step,
		      unsigned int index, bool mixValuesAcrossROCs);

    std::string name() const { return name_;}
    unsigned int dacchannel() const { return dacchannel_; }
    unsigned int step() const { return step_; }
    unsigned int first() const { return first_; }
    unsigned int last() const { return last_; }
    unsigned int index() const { return index_; }
    unsigned int getNPoints() const { return (last_-first_)/step_+1; }
    bool mixValuesAcrossROCs() const { return mixValuesAcrossROCs_; }

  private:

    std::string name_;
    unsigned int dacchannel_;
    unsigned int first_;
    unsigned int last_;
    unsigned int step_;
    unsigned int index_;

    bool mixValuesAcrossROCs_; // whether to spread the DAC values across the entire range on each iteration for different ROCs on a channel

  };
}
#endif
