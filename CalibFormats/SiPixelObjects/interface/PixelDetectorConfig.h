#ifndef PixelDetectorConfig_h
#define PixelDetectorConfig_h
/**
*   \file CalibFormats/SiPixelObjects/interface/PixelDetectorConfig.h
*   \brief This class specifies which detector components are used in the 
*          configuration (and eventually should specify which  xdaq process 
*          controls which components).
*
*   A longer explanation will be placed here later
*/
//
// This class specifies which detector
// components are used in the configuration
// (and eventually should specify which
// xdaq process controlls which components).
//
//
//
//

#include <vector>
#include <set>
#include <map>
#include <string>
#include "CalibFormats/SiPixelObjects/interface/PixelConfigBase.h"
#include "CalibFormats/SiPixelObjects/interface/PixelModuleName.h"
#include "CalibFormats/SiPixelObjects/interface/PixelHdwAddress.h"
#include "CalibFormats/SiPixelObjects/interface/PixelNameTranslation.h"
#include "CalibFormats/SiPixelObjects/interface/PixelROCStatus.h"

namespace pos {
  /*!  \ingroup ConfigurationObjects "Configuration Objects"
*    
*  @{
*
*  \class PixelDetectorConfig PixelDetectorConfig.h
*  \brief This is the documentation about PixelDetectorConfig...
*/
  class PixelDetectorConfig : public PixelConfigBase {
  public:
    PixelDetectorConfig(std::vector<std::vector<std::string> > &tableMat);
    PixelDetectorConfig(std::string filename);

    unsigned int getNModules() const;

    PixelModuleName getModule(unsigned int i) const;

    const std::vector<PixelModuleName> &getModuleList() const { return modules_; }

    void addROC(PixelROCName &, std::string statusLabel);
    void addROC(PixelROCName &);
    void removeROC(PixelROCName &);
    const std::map<PixelROCName, PixelROCStatus> &getROCsList() const { return rocs_; };

    void writeASCII(std::string dir = "") const override;
    void writeXML(pos::PixelConfigKey key, int version, std::string path) const override;
    void writeXMLHeader(pos::PixelConfigKey key,
                        int version,
                        std::string path,
                        std::ofstream *out,
                        std::ofstream *out1 = nullptr,
                        std::ofstream *out2 = nullptr) const override;
    void writeXML(std::ofstream *out, std::ofstream *out1 = nullptr, std::ofstream *out2 = nullptr) const override;
    void writeXMLTrailer(std::ofstream *out,
                         std::ofstream *out1 = nullptr,
                         std::ofstream *out2 = nullptr) const override;

    bool containsModule(const PixelModuleName &moduleToFind) const;

    std::set<unsigned int> getFEDs(PixelNameTranslation *translation) const;
    std::map<unsigned int, std::set<unsigned int> > getFEDsAndChannels(PixelNameTranslation *translation) const;

    //friend std::ostream& operator<<(std::ostream& s, const PixelDetectorconfig& config);

  private:
    std::vector<PixelModuleName> modules_;

    std::map<PixelROCName, PixelROCStatus> rocs_;
  };
}  // namespace pos
/* @} */
#endif
