#ifndef L1Trigger_RPCLogHit_h
#define L1Trigger_RPCLogHit_h

/** \class RPCLogHit
 * 
 * Class to store logical hit data: m_tower number (0, 16), coneNumber (in phi, 0 ,144),
 * logical plane number (1, 6), strip number in cone (0, to maximum cone width in givel plane
 * see RPCConst)
 *
 * \author  Marcin Konecki, Warsaw
 *          Artur Kalinowski, Warsaw
 *          Karol Bunkowski, Warsaw
 *
 ********************************************************************/

#include <vector>
//#include "L1Trigger/RPCTrigger/src/l1RpcConeCrdnts.h"
#include "L1Trigger/RPCTrigger/interface/RPCConst.h"

class RPCLogHit {

public:

  ///
  ///Default ctor.
  ///
  RPCLogHit() {};

  
  RPCLogHit(int m_tower, int m_PAC, int m_logplane, int m_posInCone);
  ///
  ///Default dctor.
  ///
  ~RPCLogHit(){ }

  
  RPCConst::l1RpcConeCrdnts getConeCrdnts() const;

  int getTower() const;

  int getLogSector() const;

  int getLogSegment() const;

  int getlogPlaneNumber() const;

  int getStripNumberInCone() const;

  void  setDigiIdx(int);

  int getDigiIdx() const;

private:
  RPCConst::l1RpcConeCrdnts m_ConeCrdnts;
  
  int m_logPlaneNumber, m_stripNumberInCone;

  int m_digiIdx;
};
#endif
