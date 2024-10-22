#ifndef PixelMaxVsf_h
#define PixelMaxVsf_h
/**
* \file CalibFormats/SiPixelObjects/interface/PixelMaxVsf.h
* \brief This class specifies the maximum Vsf setting that should be used
*        for each ROC.
*/

#include <map>
#include <string>
#include <vector>
#include "CalibFormats/SiPixelObjects/interface/PixelConfigBase.h"
#include "CalibFormats/SiPixelObjects/interface/PixelROCName.h"

namespace pos {
  /*!  \ingroup ConfigurationObjects "Configuration Objects"
*    
*  @{
*
*  \class PixelMaxVsf PixelMaxVsf.h
*  \brief This is the documentation about PixelMaxVsf...
*
*   This class specifies the maximum Vsf setting that should be used
*   for each ROC.
*/
  class PixelMaxVsf : public PixelConfigBase {
  public:
    PixelMaxVsf(std::vector<std::vector<std::string> > &tableMat);
    PixelMaxVsf(std::string filename);

    void writeASCII(std::string dir = "") const override;
    void writeXML(pos::PixelConfigKey key, int version, std::string path) const override { ; }
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

    bool getVsf(PixelROCName roc, unsigned int &Vsf) const;

    void setVsf(PixelROCName roc, unsigned int Vsf);

  private:
    std::map<PixelROCName, unsigned int> rocs_;
  };

}  // namespace pos
/* @} */
#endif
