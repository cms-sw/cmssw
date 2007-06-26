#ifndef PixelCalib_h
#define PixelCalib_h
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

class PixelCalib: public PixelConfigBase {

 public:
    PixelCalib(std::string filename="");

/*    void nextFECState(PixelFECConfigInterface* pixelFEC,
		      PixelDetectorConfig* detconfig,
		      PixelNameTranslation* trans, 
		      unsigned int state) const; 
*/
    //return vector of fed# and channels controlled by this
    //fed supervisor
    /*  std::vector<std::pair<unsigned int,std::vector<unsigned int> > >& fedCardsAndChannels(unsigned int crate,
												   PixelNameTranslation* translation,
												   PixelFEDConfig* fedconfig) const;*/

    //void nextFEDStep(PixelFEDConfigInterface* pixelFEd) const; 

    //void nextTTCStep(PixelFECConfigInterface* pixelFEd) const; 

    unsigned int nPixelPatterns() const { return rows_.size()*cols_.size(); }
    unsigned int nTriggersPerPattern() const { return ntrigger_; }
    unsigned int vcal_first() {return vcal_first_;}
    unsigned int vcal_last()  {return vcal_last_;}
    unsigned int vcal_step() {return vcal_step_;}
    double vcal_step2() {return vcal_step_;}
    unsigned int nVcal() const { return (vcal_last_-vcal_first_)/vcal_step_+1; }    
    unsigned int nConfigurations() const {return nPixelPatterns()*nVcal();}
    unsigned int nTriggersTotal() const {return nConfigurations()*nTriggersPerPattern();}
    unsigned int vcal(unsigned int state) const;
    unsigned int vcal_fromeventno(unsigned int evtno) const;
    
    unsigned int nTriggers() {return ntrigger_;}
    const std::vector<std::string>& rocList(){return rocs_;}

    friend std::ostream& operator<<(std::ostream& s, const PixelCalib& calib);

    void getRowsAndCols(unsigned int state,
			std::vector<unsigned int>& rows,
			std::vector<unsigned int>& cols) const;

 private:

    std::vector<std::vector<unsigned int> > rows_;
    std::vector<std::vector<unsigned int> > cols_;

    mutable std::vector<std::string> rocs_;
    bool roclistfromconfig_;


    mutable std::vector<std::pair<unsigned int, std::vector<unsigned int> > > fedCardsAndChannels_;
    
    unsigned int vcal_first_;
    unsigned int vcal_last_;
    unsigned int vcal_step_;
    unsigned int ntrigger_;

/*    void enablePixels(PixelFECConfigInterface* pixelFEC,
		      unsigned int irows, unsigned int icols,
		      PixelHdwAddress theROC) const;*/
    mutable int old_irows;
    mutable int old_icols;
    
};

#endif
