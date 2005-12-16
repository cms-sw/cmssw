#ifndef SISTRIPREADOUTCABLING_H
#define SISTRIPREADOUTCABLING_H
//#include <map>
#include <vector>
#include <boost/cstdint.hpp>

using namespace std;

class SiStripReadoutCabling{
public:

  //APV pairs identifier (DetId according to convention in DataFormats/SiStripId, apv on Det [0,2])
  typedef pair<uint32_t, unsigned short>  APVPairId;  
  typedef pair<unsigned short, unsigned short> FEDChannelId;

  SiStripReadoutCabling();
  SiStripReadoutCabling(const vector< vector< APVPairId > > & cabling);
  virtual ~SiStripReadoutCabling();

  const APVPairId & getAPVPair(unsigned short fed_id, unsigned short fed_channel) const;		

  const APVPairId & getAPVPair(const FEDChannelId & fedch_id) const;

  const vector<unsigned short> & getFEDs() const;

  const vector< APVPairId > & getFEDAPVPairs(unsigned short fed_id) const;

private:

  //matching of fed channels ([0,1023] feds (only the Tk strip ones have non empty vector<APVPairId>), [0,95] channels) to APV pairs)
  vector< vector< pair<uint32_t, unsigned short> >  > theFEDConnections;
  //active si strip feds
  vector<unsigned short> theFEDs;

};
#endif
