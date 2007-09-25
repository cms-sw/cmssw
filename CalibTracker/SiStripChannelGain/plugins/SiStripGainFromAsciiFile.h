#ifndef CalibTracker_SiStripChannelGain_SiStripGainFromAsciiFile_h
#define CalibTracker_SiStripChannelGain_SiStripGainFromAsciiFile_h


#include "CommonTools/ConditionDBWriter/interface/ConditionDBWriter.h"
#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"
#include <vector>

#include <ext/hash_map>

class SiStripGainFromAsciiFile : public ConditionDBWriter<SiStripApvGain> {

public:

  explicit SiStripGainFromAsciiFile(const edm::ParameterSet&);
  ~SiStripGainFromAsciiFile();

  void algoAnalyze(const edm::Event & event, const edm::EventSetup& iSetup);

private:

  SiStripApvGain * getNewObject(){return obj;}

private:

  struct FibersGain{
    float fiber[3];

    void soft_reset(){ for (int i=0;i<3;++i) if(fiber[i]==-1)fiber[i]=1; }
    void hard_reset(float val){ for (int i=0;i<3;++i) fiber[i]=val; }
    
  };

  std::string Asciifilename_;
  float referenceValue_;

  SiStripApvGain * obj;

  std::vector< std::pair<uint32_t, unsigned short> > detid_apvs_;
    
  __gnu_cxx::hash_map< unsigned int,FibersGain>  GainsMap;
  //std::map< unsigned int,FibersGain>  GainsMap;

};
#endif
