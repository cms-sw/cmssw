#ifndef CALIBFORMATS_SISTRIPOBJECTS_SISTRIPCONTROLCONNECTIVITY_H
#define CALIBFORMATS_SISTRIPOBJECTS_SISTRIPCONTROLCONNECTIVITY_H

#include "DataFormats/DetId/interface/DetId.h"

#include <vector>
#include <map>
using namespace std;

class SiStripControlConnectivity {
public:
  typedef pair<DetId,unsigned short> DetPair;
  typedef pair<unsigned short, unsigned short> FedReference; 
  typedef map<FedReference, DetPair>  MapType;

  SiStripControlConnectivity(){}
  ~SiStripControlConnectivity(){}

  DetId getDetId(FedReference& in);
  DetId getDetId(unsigned short fed_id, unsigned short fed_channel);

  int getDetIds(unsigned short fed_num, unsigned short max_channels, vector<DetId>&);
  unsigned short getFedIdAndChannels(DetId id,vector<unsigned short>& fedChannels);

  unsigned short getPairNumber(SiStripControlConnectivity::FedReference& );
  unsigned short getPairNumber(unsigned short fed_id, unsigned short fed_channel);

  void getDetPair(SiStripControlConnectivity::FedReference& fed_ref, DetPair& det_pair);

  /** Returns (by reference in the argument list) DetPair information
      (DetUnit*, APVpair) for given FED id and channel. */
  void getDetPair(unsigned short fed_id, unsigned short fed_channel, DetPair& det_pair);

  int getStripNumber(SiStripControlConnectivity::FedReference&,int);
  int getStripNumber(unsigned short& fed_id, unsigned short& fed_channel);

  void setPair(FedReference&, DetPair&);
  
  void clean(){theMap.clear();}
  void debug();

  const MapType& getFedList();
  pair<int, int> getFedRange();

  void getConnectedFedNumbers(vector<unsigned short>& feds);
  void getDetPartitions(map<unsigned short, vector<DetId> >& partitions);

 private:
  
  map< unsigned short, FecRings * > theFecConnections;

};

#endif
