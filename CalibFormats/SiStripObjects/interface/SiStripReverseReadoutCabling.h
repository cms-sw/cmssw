#ifndef CALIBFORMATS_SISTRIPOBJECTS_SISTRIPREVERSEREADOUTCABLING_H
#define CALIBFORMATS_SISTRIPOBJECTS_SISTRIPREVERSEREADOUTCABLING_H
#include "CondFormats/SiStripObjects/interface/SiStripReadoutCabling.h"
#include <boost/cstdint.hpp>
//#include <vector>
#include <map>

//class SiStripReadoutCabling; 

//using namespace std;

class SiStripReverseReadoutCabling {

public:

  SiStripReverseReadoutCabling();

  SiStripReverseReadoutCabling(const SiStripReadoutCabling *);

  ~SiStripReverseReadoutCabling();

  const SiStripReadoutCabling::FEDChannelId & getFEDChannel(uint32_t det_id , unsigned short apvpair_id) const;

  const SiStripReadoutCabling::FEDChannelId & getFEDChannel(const SiStripReadoutCabling::APVPairId & apvpair) const;

  const  std::map<unsigned short, SiStripReadoutCabling::FEDChannelId> & getAPVPairs(const uint32_t det_id) const;

  const std::map<uint32_t, std::map<unsigned short, SiStripReadoutCabling::FEDChannelId> > & getDetConnections() const;

  void debug() const;

private:

  std::map<uint32_t, std::map<unsigned short, SiStripReadoutCabling::FEDChannelId> > theDetConnections;

};

#endif
