#ifndef L1RpcLogHitH
#define L1RpcLogHitH

/** \class L1RpcLogHit
 * 
 * Class to store logical hit data: tower number (0, 16), coneNumber (in phi, 0 ,144),
 * logical plane number (1, 6), strip number in cone (0, to maximum cone width in givel plane
 * see L1RpcConst)
 *
 * \author  Marcin Konecki, Warsaw
 *          Artur Kalinowski, Warsaw
 *          Karol Bunkowski, Warsaw
 *
 ********************************************************************/

#include <vector>
//#include "L1Trigger/RPCTrigger/src/L1RpcConeCrdnts.h"
#include "L1Trigger/RPCTrigger/src/L1RpcParameters.h"

using namespace std;
//#include "Muon/MCommonData/interface/MRpcDigi.h"

//#include "Trigger/L1RpcTrigger/src/L1RpcParametersDef.h"

class L1RpcLogHit {

public:

  ///
  ///Default ctor.
  ///
  L1RpcLogHit() {};

  
  L1RpcLogHit(int tower, int PAC, int logplane, int posInCone);
  ///
  ///Default dctor.
  ///
  ~L1RpcLogHit(){ }

  
  rpcparam::L1RpcConeCrdnts GetConeCrdnts() const;

  int getTower() const;

  int getLogSector() const;

  int getLogSegment() const;

  int getlogPlaneNumber() const;

  int getStripNumberInCone() const;

  void  setDigiIdx(int _digiIdx);

  int getDigiIdx() const;

private:
  rpcparam::L1RpcConeCrdnts ConeCrdnts;
  
  int logPlaneNumber, stripNumberInCone;

  int digiIdx;
};
#endif
