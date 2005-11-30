#ifndef CALIBFORMATS_SISTRIPOBJECTS_SISTRIPREADOUTCONNECTIVITY_H
#define CALIBFORMATS_SISTRIPOBJECTS_SISTRIPREADOUTCONNECTIVITY_H

#include "DataFormats/DetId/interface/DetId.h"

#include <vector>
#include <map>
using namespace std;

class SiStripReadoutConnectivity {
public:
  typedef pair<DetId,unsigned short> DetPair;
  typedef pair<unsigned short, unsigned short> FedReference; 
  //  typedef map<FedReference, DetPair>  MapType;

  SiStripReadoutConnectivity(){}
  ~SiStripReadoutConnectivity(){}

  DetId getDetId(FedReference& in);
  DetId getDetId(unsigned short fed_id, unsigned short fed_channel);

  int getDetIds(unsigned short fed_num, unsigned short max_channels, vector<DetId>&);
  unsigned short getFedIdAndChannels(DetId id,map<unsigned short, 
                                        unsigned short>& fedChannels);

  unsigned short getPairNumber(SiStripReadoutConnectivity::FedReference& );
  unsigned short getPairNumber(unsigned short fed_id, unsigned short fed_channel);

  void getDetPair(SiStripReadoutConnectivity::FedReference& fed_ref, DetPair& det_pair);

  /** Returns (by reference in the argument list) DetPair information
      (DetUnit*, APVpair) for given FED id and channel. */
  void getDetPair(unsigned short fed_id, unsigned short fed_channel, DetPair& det_pair);

  int getStripNumber(SiStripReadoutConnectivity::FedReference&,int);
  int getStripNumber(unsigned short& fed_id, unsigned short& fed_channel);

  void setPair(FedReference&, DetPair&);
  
  void clean(){detUnitMap_.clear();}
  void debug();


  void getConnectedFedNumbers(vector<unsigned short>& feds);
  //  void getDetPartitions(map<unsigned short, vector<DetId> >& partitions);

 private:
  //  MapType theMap;
  
  /** DetPair info for FED id (1st dim) and channel (2nd dim). */
  vector< vector<DetPair> > detUnitMap_; // M.W, R.B 

};

#endif
