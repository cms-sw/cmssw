#ifndef SiPixelCalibConfiguration_h
#define SiPixelCalibConfiguration_h
//
// This class imnplement the steps
// that are used in a calibration  such
// as e.g. an S-curve
// 
//
// some additional functionality added by F.Blekman, dd April 18, 2007.
//

#include <vector>
#include <string>
#include <utility>
#include "CalibFormats/SiPixelObjects/interface/PixelConfigBase.h"

class PixelHdwAddress;
class PixelFECInterface;
class PixelDetectorConfig;
class PixelNameTranslation;
class PixelFEDConfig;

class SiPixelCalibConfiguration: public PixelConfigBase {

 public:
    SiPixelCalibConfiguration(std::string filename="");

/*    void nextFECState(PixelFECConfigInterface* pixelFEC,
		      PixelDetectorConfig* detconfig,
		      PixelNameTranslation* trans, 
		      uint32_t state) const; 
*/
    //return vector of fed# and channels controlled by this
    //fed supervisor
    /*  std::vector<std::pair<uint32_t,std::vector<uint32_t> > >& fedCardsAndChannels(uint32_t crate,
												   PixelNameTranslation* translation,
												   PixelFEDConfig* fedconfig) const;*/

    //void nextFEDStep(PixelFEDConfigInterface* pixelFEd) const; 

    //void nextTTCStep(PixelFECConfigInterface* pixelFEd) const; 

    uint32_t nPixelPatterns() const { return rows_.size()*cols_.size(); }
    uint32_t nTriggersPerPattern() const { return ntrigger_; }
    uint32_t vcal_first() {return vcal_first_;}
    uint32_t vcal_last()  {return vcal_last_;}
    uint32_t vcal_step() {return vcal_step_;}
    double vcal_step2() {return vcal_step_;}
    uint32_t nVcal() const { return (vcal_last_-vcal_first_)/vcal_step_+1; }    
    uint32_t nConfigurations() const {return nPixelPatterns()*nVcal();}
    uint32_t nTriggersTotal() const {return nConfigurations()*nTriggersPerPattern();}
    uint32_t vcal(uint32_t state) const;
    uint32_t vcal_fromeventno(uint32_t evtno) const;
    
    uint32_t nTriggers() {return ntrigger_;}
    const std::vector<std::string>& rocList(){return rocs_;}

    friend std::ostream& operator<<(std::ostream& s, const SiPixelCalibConfiguration& calib);

    void getRowsAndCols(uint32_t state,
			std::vector<uint32_t>& rows,
			std::vector<uint32_t>& cols) const;

 private:

    std::vector<std::vector<uint32_t> > rows_;
    std::vector<std::vector<uint32_t> > cols_;

    mutable std::vector<std::string> rocs_;
    bool roclistfromconfig_;


    mutable std::vector<std::pair<uint32_t, std::vector<uint32_t> > > fedCardsAndChannels_;
    
    uint32_t vcal_first_;
    uint32_t vcal_last_;
    uint32_t vcal_step_;
    uint32_t ntrigger_;

/*    void enablePixels(PixelFECConfigInterface* pixelFEC,
		      uint32_t irows, uint32_t icols,
		      PixelHdwAddress theROC) const;*/
    mutable int old_irows;
    mutable int old_icols;
    
};

#endif
