#ifndef CalibTracker_SiStripChannelGain_SiStripGainFromAsciiFile_h
#define CalibTracker_SiStripChannelGain_SiStripGainFromAsciiFile_h

#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "CommonTools/ConditionDBWriter/interface/ConditionDBWriter.h"
#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"

#include <vector>
#include <memory>
#include <unordered_map>

class SiStripGainFromAsciiFile : public ConditionDBWriter<SiStripApvGain> {

public:

  explicit SiStripGainFromAsciiFile(const edm::ParameterSet&);
  ~SiStripGainFromAsciiFile() override;

private:

  std::unique_ptr<SiStripApvGain> getNewObject() override;

private:

  struct ModuleGain{
    float apv[6];

    void soft_reset(){ for (int i=0;i<6;++i) if(apv[i]==-1)apv[i]=1; }
    void hard_reset(float val){ for (int i=0;i<6;++i) apv[i]=val; }
    
  };

  std::string Asciifilename_;
  float referenceValue_;
  edm::FileInPath fp_;

  std::unordered_map< unsigned int,ModuleGain>  GainsMap;

};
#endif
