#ifndef RPCObject_RPCelecIndex_h
#define RPCObject_RPCelecIndex_h
/** \class RPCelecIndex 
 *
 *  Description"
 *       Class to index the RPC read out electronic channel
 *  \author Marcello Maggi -- INFN Bari
 *
 */

class RPCelecIndex{
 public:
  RPCelecIndex();
  RPCelecIndex(int dccId, int tbId, int lboxId, int mbId, 
	       int lboardId, int channelId);
  bool operator <(const RPCelecIndex& einx) const;
  int dcc() const;
  int tb() const;
  int lbox() const;
  int mb() const;
  int lboard() const;
  int channel() const;

 private:
  int dat;
  int trb;
  int lbb;
  int mbo;
  int lbo;
  int cha;
};
#endif
