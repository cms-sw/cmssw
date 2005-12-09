#ifndef CALIBFORMATS_SISTRIPOBJECTS_SISTRIPREADOUTCONNECTIVITY_H
#define CALIBFORMATS_SISTRIPOBJECTS_SISTRIPREADOUTCONNECTIVITY_H

#include "DataFormats/DetId/interface/DetId.h"


#include <vector>
#include <map>

class SiStripReadoutCabling; 

using namespace std;


class SiStripReadoutConnectivity {

public:

  typedef pair<DetId,unsigned short> APVPairId;
  typedef pair<unsigned short, unsigned short> FedChannelId; 

  SiStripReadoutConnectivity();

  SiStripReadoutConnectivity(const SiStripReadoutCabling *);

  ~SiStripReadoutConnectivity(){}

  inline const APVPairId & getAPVPair(unsigned short fed_id, unsigned short fed_channel) const {return ;

  const APVPairId & getAPVPair(const FEDChannelId & fedch_id) const;

  const vector<unsigned short> & getFeds() const;

  const FEDChannelId & getFEDChannel(const DetId & det, unsigned short apvpair) const;

  const FEDChannelId & getFEDChannel(const APVPairId & apvpair) const;

  //  int getStripNumber(SiStripReadoutConnectivity::FEDChannel&,int);
  //  int getStripNumber(unsigned short& fed_id, unsigned short& fed_channel);
  //  void addConnection(FEDChannelId&, APVPairId&);
  //  void clean(){detUnitMap_.clear();}

  void debug() const;

private:

  map<DetId, vector<FEDChannelId> > theDetConnections;
  vector< vector<APVPairId> > theFEDConnections; 
  vector<unsigned short> theFEDs;

};

#endif
