#ifndef CalibTracker_SiStripChannelGain_SiStripGainFromAsciiFile_h
#define CalibTracker_SiStripChannelGain_SiStripGainFromAsciiFile_h

#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "CommonTools/ConditionDBWriter/interface/ConditionDBWriter.h"
#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"
#include <vector>

#include <unordered_map>

class SiStripGainFromAsciiFile : public ConditionDBWriter<SiStripApvGain> {

public:

  explicit SiStripGainFromAsciiFile(const edm::ParameterSet&);
  ~SiStripGainFromAsciiFile();

private:

  SiStripApvGain * getNewObject();

private:

  struct ModuleGain{
    float apv[6];

    void soft_reset(){ for (float & i : apv) if(i==-1)i=1; }
    void hard_reset(float val){ for (float & i : apv) i=val; }
    
  };

  std::string Asciifilename_;
  float referenceValue_;
  edm::FileInPath fp_;

  std::unordered_map< unsigned int,ModuleGain>  GainsMap;

};
#endif
