#ifndef PixelCalib_h
#define PixelCalib_h
//
// This class inplement the steps
// that are used in a scan over
// Threshold and CalDelay
//
//
//
//

#include <vector>
#include <set>
#include <map>
#include <string>
#include <utility>
#include "CalibFormats/SiPixelObjects/interface/PixelCalibBase.h"
#include "CalibFormats/SiPixelObjects/interface/PixelNameTranslation.h"
#include "CalibFormats/SiPixelObjects/interface/PixelDetectorConfig.h"
#include "CalibFormats/SiPixelObjects/interface/PixelFEDConfig.h"
#include "CalibFormats/SiPixelObjects/interface/PixelFECConfig.h"
#include "CalibFormats/SiPixelObjects/interface/PixelTKFECConfig.h"
#include "CalibFormats/SiPixelObjects/interface/PixelFECConfigInterface.h"
#include "CalibFormats/SiPixelObjects/interface/PixelROCName.h"
#include "CalibFormats/SiPixelObjects/interface/PixelModuleName.h"
#include "CalibFormats/SiPixelObjects/interface/PixelModuleName.h"
#include "CalibFormats/SiPixelObjects/interface/PixelMaskBase.h"
#include "CalibFormats/SiPixelObjects/interface/PixelTrimBase.h"
#include "CalibFormats/SiPixelObjects/interface/PixelROCMaskBits.h"
#include "CalibFormats/SiPixelObjects/interface/PixelROCTrimBits.h"
#include "CalibFormats/SiPixelObjects/interface/PixelConfigBase.h"
#include "CalibFormats/SiPixelObjects/interface/PixelDACScanRange.h"
#include "CalibFormats/SiPixelObjects/interface/PixelPortcardMap.h"
#include "CalibFormats/SiPixelObjects/interface/PixelPortCardConfig.h"


namespace pos{
  class PixelHdwAddress;

  class PixelCalibConfiguration : public PixelCalibBase, public PixelConfigBase {

  public:

    PixelCalibConfiguration(std::string filename="");

    virtual ~PixelCalibConfiguration();

    // This must be run before using commands that require the ROC list.
    void buildROCAndModuleLists(const PixelNameTranslation* translation, const PixelDetectorConfig* detconfig);

    void nextFECState(PixelFECConfigInterface* pixelFEC,
		      PixelDetectorConfig* detconfig,
		      PixelNameTranslation* trans,
		      std::map<pos::PixelModuleName,pos::PixelMaskBase*>* masks,
		      std::map<pos::PixelModuleName,pos::PixelTrimBase*>* trims,
		      unsigned int state) const; 

    //return vector of fed# and channels controlled by this fed supervisor
    std::vector<std::pair<unsigned int,std::vector<unsigned int> > >& fedCardsAndChannels(unsigned int crate, PixelNameTranslation* translation, PixelFEDConfig* fedconfig,PixelDetectorConfig* detconfig) const;

    std::map <unsigned int, std::set<unsigned int> > getFEDsAndChannels (PixelNameTranslation *translation);
    
    // Returns a std::set of FED crates that are used by this Calib object
    std::set <unsigned int> getFEDCrates(const PixelNameTranslation *translation, const PixelFEDConfig *fedconfig) const;

    // Returns a std::set of FEC crates that are used by this Calib object
    std::set <unsigned int> getFECCrates(const PixelNameTranslation *translation, const PixelFECConfig* fecconfig) const;

    // Returns a std::set of TKFEC crates that are used by this Calib object
    std::set <unsigned int> getTKFECCrates(const PixelPortcardMap *portcardmap, const std::map<std::string,PixelPortCardConfig*>& mapNamePortCard, const PixelTKFECConfig* tkfecconfig) const;

    unsigned int nROC() const { assert(rocAndModuleListsBuilt_); return nROC_; }
    unsigned int nPixelPatterns() const { return rows_.size()*cols_.size(); }
    unsigned int nTriggersPerPattern() const { return ntrigger_; }
    unsigned int nScanPoints(unsigned int iscan) const { return (dacs_[iscan].last()-dacs_[iscan].first())/dacs_[iscan].step()+1; }    
    unsigned int nScanPoints(std::string dac) const { return nScanPoints(iScan(dac)); }    

    unsigned int nScanPoints() const {unsigned int points=1;
      for(unsigned int i=0;i<dacs_.size();i++) {
	points*=nScanPoints(i);
      }
      return points;
    }
    unsigned int nConfigurations() const { assert(rocAndModuleListsBuilt_); return nPixelPatterns()*nScanPoints()*nROC();}
    unsigned int nTriggersTotal() const {return nConfigurations()*nTriggersPerPattern();}

    unsigned int scanValue(unsigned int iscan, unsigned int state) const;
    unsigned int scanValue(std::string dac, unsigned int state) const{
      return scanValue(iScan(dac),state);
    }

    unsigned int scanCounter(unsigned int iscan, unsigned int state) const;
    unsigned int scanCounter(std::string dac, unsigned int state) const{
      return scanCounter(iScan(dac),state);
    }

    double scanValueMin(unsigned int iscan) const {return dacs_[iscan].first();}
    double scanValueMin(std::string dac) const {return scanValueMin(iScan(dac));}
    double scanValueMax(unsigned int iscan) const {return dacs_[iscan].first()+
	dacs_[iscan].step()*(nScanPoints(iscan)-1);}
    double scanValueMax(std::string dac) const {return scanValueMax(iScan(dac));}
    double scanValueStep(unsigned int iscan) const {return dacs_[iscan].step();}
    double scanValueStep(std::string dac) const {return scanValueStep(iScan(dac));}

    unsigned int iScan(std::string dac) const;

    const std::vector<std::vector<unsigned int> > &columnList() const {return cols_;}
    const std::vector<std::vector<unsigned int> > &rowList() const {return rows_;}
    const std::vector<PixelROCName>& rocList() const {assert(rocAndModuleListsBuilt_); return rocs_;}
    const std::set <PixelModuleName>& moduleList() const {assert(rocAndModuleListsBuilt_); return modules_;}
    const std::set <PixelChannel>& channelList(const PixelNameTranslation* aNameTranslation);

    virtual std::string mode() {return mode_;}

    unsigned int nParameters() const {return parameters_.size();}
    // get the value of parameter parameterName, or "" if parameterName is not in the list
    std::string parameterValue(std::string parameterName) const;

    friend std::ostream& pos::operator<<(std::ostream& s, const PixelCalibConfiguration& calib);

    virtual void writeASCII(std::string dir="") const;


  private:

    // Used in constructor or in buildROCAndModuleLists()
    void buildROCAndModuleListsFromROCSet(const std::set<PixelROCName>& rocSet);

    //Mode is one of the following: 
    //  ThresholdCalDelay
    //  FEDChannelOffsetDAC
    //  FEDAddressLevelDAC
    //  FEDChannelOffsetPixel
    //  FEDAddressLevelPixel
    //  GainCalibration
    //  PixelAlive
    //  SCurve
    //  ClockPhaseCalibration

    bool singleROC_;

    std::vector<std::vector<unsigned int> > rows_;
    std::vector<std::vector<unsigned int> > cols_;

    mutable std::vector<PixelROCName> rocs_;
    std::set <PixelModuleName> modules_;
    std::map <PixelModuleName,unsigned int> countROC_;
    bool rocAndModuleListsBuilt_;
    std::vector<std::string> rocListInstructions_;
    
    // Channel list, filled from ROC list only when needed.
    std::set <PixelChannel> channels_;
    bool channelListBuilt_;

    mutable std::vector<std::pair<unsigned int, std::vector<unsigned int> > > fedCardsAndChannels_;

    //unsigned int vcal_;

    std::vector<PixelDACScanRange> dacs_;

    //std::vector<std::string> dacname_;
    //std::vector<unsigned int> dacchannel_;
    //std::vector<unsigned int> dac_first_;
    //std::vector<unsigned int> dac_last_;
    //std::vector<unsigned int> dac_step_;

    unsigned int ntrigger_;
    unsigned int nROC_; //This is the maximal #ROCs on a given TBM

    bool highVCalRange_;

    void enablePixels(PixelFECConfigInterface* pixelFEC,
		      unsigned int irows, unsigned int icols,
		      pos::PixelROCMaskBits* masks,
		      pos::PixelROCTrimBits* trims,	
		      PixelHdwAddress theROC) const;

    void disablePixels(PixelFECConfigInterface* pixelFEC,
		       unsigned int irows, unsigned int icols,
		       PixelHdwAddress theROC) const;

    void disablePixels(PixelFECConfigInterface* pixelFEC,
		       PixelHdwAddress theROC) const;

    mutable std::vector<int> old_irows;
    mutable std::vector<int> old_icols;

    std::map<std::string, std::string> parameters_;
    //       name         value

    bool _bufferData;

  };
}
#endif
