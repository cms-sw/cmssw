#ifndef PixelCalibConfiguration_h
#define PixelCalibConfiguration_h
/*! \file CalibFormats/SiPixelObjects/interface/PixelCalibConfiguration.h
*   \brief This class implements the steps that are used in a scan over Threshold and CalDelay
*
*   A longer explanation will be placed here later
*/

#include <vector>
#include <set>
#include <map>
#include <string>
#include <sstream>
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
#include "CalibFormats/SiPixelObjects/interface/PixelDACSettings.h"
#include "CalibFormats/SiPixelObjects/interface/PixelConfigBase.h"
#include "CalibFormats/SiPixelObjects/interface/PixelDACScanRange.h"
#include "CalibFormats/SiPixelObjects/interface/PixelPortcardMap.h"
#include "CalibFormats/SiPixelObjects/interface/PixelPortCardConfig.h"


namespace pos{
  class PixelHdwAddress;

  //This class contains info about a ROC
  class PixelROCInfo {    
  public:
    bool use_;
    const PixelHdwAddress* hdwadd_;
    //FIXME these should be const but it has ripple effects...
    PixelROCTrimBits* trims_;
    PixelROCMaskBits* masks_;
    std::vector<std::pair<unsigned int, unsigned int> > defaultDACs_;
    std::string tbmChannel_;
  };


/*!  \ingroup ConfigurationObjects "Configuration Objects"
*    \ingroup CalibrationObjects "Calibration Objects"
*    \brief This class implements the steps that are used in a scan over Threshold and CalDelay
*
*    It features a double inheritance, both from ConfigurationObjects and CalibrationObjects
*
*  @{
*
*   \class PixelCalibConfiguration PixelCalibConfiguration.h "interface/PixelCalibConfiguration.h"
*
*   A longer explanation will be placed here later
*/
  class PixelCalibConfiguration;
  std::ostream& operator<<(std::ostream& s, const PixelCalibConfiguration& calib);

  class PixelCalibConfiguration : public PixelCalibBase, public PixelConfigBase {

  public:

    PixelCalibConfiguration(std::string filename="");
    PixelCalibConfiguration(std::vector<std::vector<std::string> > &);

    virtual ~PixelCalibConfiguration();

    // This must be run before using commands that require the ROC list.
    void buildROCAndModuleLists(const PixelNameTranslation* translation, const PixelDetectorConfig* detconfig);

    void nextFECState(std::map<unsigned int, PixelFECConfigInterface*>& pixelFECs,
		      PixelDetectorConfig* detconfig,
		      PixelNameTranslation* trans,
		      std::map<pos::PixelModuleName,pos::PixelMaskBase*>* masks,
		      std::map<pos::PixelModuleName,pos::PixelTrimBase*>* trims,
		      std::map<pos::PixelModuleName,pos::PixelDACSettings*>* dacss,
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
    unsigned int nScanPoints(std::string dac) const { return nScanPoints(iScan(dac)); }    

    unsigned int nScanPoints() const {unsigned int points=1;
      for(unsigned int i=0;i<dacs_.size();i++) {
	points*=nScanPoints(i);
      }
      return points;
    }
    unsigned int nConfigurations() const { return nPixelPatterns()*nScanPoints()*nROC();}
    unsigned int nTriggersTotal() const {return nConfigurations()*nTriggersPerPattern();}

    bool noHits() const {return (maxNumHitsPerROC()==0);} // returns true if no hits will be produced
    unsigned int maxNumHitsPerROC() const; // returns the maximum number of hits that will be produced in any pixel pattern

    // Return all the pixels that are enabled for this state.
    std::set< std::pair<unsigned int, unsigned int> > pixelsWithHits(unsigned int state) const;
    //                  column #      row #

    // Whether this ROC is currently being scanned.  (Always true when not in SingleROC mode.)
    bool scanningROCForState(PixelROCName roc, unsigned int state) const;

    unsigned int scanCounter(std::string dac, unsigned int state) const{
      return scanCounter(iScan(dac),state);
    }

    unsigned int scanValue(std::string dac, unsigned int state, PixelROCName roc) const {
      return scanValue(iScan(dac), state, roc);
    }

    // This function should not be used -- provided for backwards compatibility only.  It asserts if the scan values for this dac are mixed across different ROCs.
    unsigned int scanValue(std::string dac, unsigned int state) const {
      assert( !(dacs_[iScan(dac)].mixValuesAcrossROCs()) );
      return scanValue(iScan(dac), state, 0, 1);
    }

    unsigned int numberOfScanVariables() const {return dacs_.size();}

    bool containsScan(std::string name) const;

    std::string scanName(unsigned int iscan) const {return dacs_[iscan].name();}
    std::vector<unsigned int> scanValues(std::string dac) const {return scanValues(iScan(dac));}

    double scanValueMin(std::string dac) const {return scanValueMin(iScan(dac));}
    double scanValueMax(std::string dac) const {return scanValueMax(iScan(dac));}
    double scanValueStep(std::string dac) const {return scanValueStep(iScan(dac));}
    bool scanValuesMixedAcrossROCs(std::string dac) const {return scanValuesMixedAcrossROCs(iScan(dac));}

    unsigned int iScan(std::string dac) const;

    const std::vector<std::vector<unsigned int> > &columnList() const {return cols_;}
    const std::vector<std::vector<unsigned int> > &rowList() const {return rows_;}
    const std::vector<PixelROCName>& rocList() const {assert(rocAndModuleListsBuilt_); return rocs_;}
    const std::set <PixelModuleName>& moduleList() const {assert(rocAndModuleListsBuilt_); return modules_;}
    const std::set <PixelChannel>& channelList() const {assert( objectsDependingOnTheNameTranslationBuilt_ ); return channels_;}

    std::string mode() const {return mode_;}

    bool singleROC() const {return singleROC_;}

    unsigned int nParameters() const {return parameters_.size();}

    // Added by Dario Apr 24th, 2008
    std::map<std::string, std::string> parametersList() const {return parameters_;}
    // get the value of parameter parameterName, or "" if parameterName is not in the list
    std::string parameterValue(std::string parameterName) const;

    // Added by Dario May 8th, 2008
    std::string getStreamedContent(void) const {return calibFileContent_;} ;

    friend std::ostream& pos::operator<<(std::ostream& s, const PixelCalibConfiguration& calib);

    virtual void writeASCII(std::string dir="") const;
    void 	 writeXML(        pos::PixelConfigKey key, int version, std::string path) const {;}
    virtual void writeXMLHeader(  pos::PixelConfigKey key, 
				  int version, 
				  std::string path, 
				  std::ofstream *out,
				  std::ofstream *out1 = NULL,
				  std::ofstream *out2 = NULL
				  ) const ;
    virtual void writeXML(        std::ofstream *out,			                                    
			   	  std::ofstream *out1 = NULL ,
			   	  std::ofstream *out2 = NULL ) const ;
    virtual void writeXMLTrailer( std::ofstream *out, 
				  std::ofstream *out1 = NULL,
				  std::ofstream *out2 = NULL
				  ) const ;


  private:

    // Which set of rows we're on.
    unsigned int rowCounter(unsigned int state) const;
    
    // Which set of columns we're on.
    unsigned int colCounter(unsigned int state) const;

    // In SingleROC mode, which ROC we're on.  In normal mode, this equals 1.
    unsigned int scanROC(unsigned int state) const;

    unsigned int nScanPoints(unsigned int iscan) const { return dacs_[iscan].getNPoints(); }

    unsigned int scanCounter(unsigned int iscan, unsigned int state) const;

    unsigned int scanValue(unsigned int iscan, unsigned int state, unsigned int ROCNumber, unsigned int ROCsOnChannel) const;
    unsigned int scanValue(unsigned int iscan, unsigned int state, PixelROCName roc) const;

    std::vector<unsigned int> scanValues(unsigned int iscan) const {return dacs_[iscan].values();}

    double scanValueMin(unsigned int iscan) const {return dacs_[iscan].first();}
    double scanValueMax(unsigned int iscan) const {return dacs_[iscan].first()+
                                                   dacs_[iscan].step()*(nScanPoints(iscan)-1);}
    double scanValueStep(unsigned int iscan) const {return dacs_[iscan].step();}
    bool scanValuesMixedAcrossROCs(unsigned int iscan) const {return dacs_[iscan].mixValuesAcrossROCs();}

    // Used in constructor or in buildROCAndModuleLists()
    void buildROCAndModuleListsFromROCSet(const std::set<PixelROCName>& rocSet);

    void buildObjectsDependingOnTheNameTranslation(const PixelNameTranslation* aNameTranslation);
    
    unsigned int ROCNumberOnChannelAmongThoseCalibrated(PixelROCName roc) const;
    unsigned int numROCsCalibratedOnChannel(PixelROCName roc) const;

    bool singleROC_;

    std::vector<std::vector<unsigned int> > rows_;
    std::vector<std::vector<unsigned int> > cols_;

    mutable std::vector<PixelROCName> rocs_;
    mutable std::vector<PixelROCInfo> rocInfo_;
    std::set <PixelModuleName> modules_;
    bool rocAndModuleListsBuilt_;
    std::vector<std::string> rocListInstructions_;
    
    // Objects built using the name translation.
    std::set <PixelChannel> channels_;
    std::map <PixelROCName, unsigned int> ROCNumberOnChannelAmongThoseCalibrated_;
    std::map <PixelROCName, unsigned int> numROCsCalibratedOnChannel_;
    bool objectsDependingOnTheNameTranslationBuilt_;
    
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
		      const PixelHdwAddress& theROC) const;

    void disablePixels(PixelFECConfigInterface* pixelFEC,
		       unsigned int irows, unsigned int icols,
		       pos::PixelROCTrimBits* trims,	
		       const PixelHdwAddress& theROC) const;

    void disablePixels(PixelFECConfigInterface* pixelFEC,
		       pos::PixelROCTrimBits* trims,	
		       const PixelHdwAddress& theROC) const;

    mutable std::vector<int> old_irows;
    mutable std::vector<int> old_icols;

    std::map<std::string, std::string> parameters_;
    //       name         value

    bool _bufferData;

    bool usesROCList_;

    // Added by Dario May 8th, 2008
    std::string calibFileContent_ ;
  };
}
/* @} */
#endif
