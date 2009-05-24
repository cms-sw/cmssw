// $Id: ProcessDigiGlobalSignal.h,v 1.1 2009/05/16 19:43:30 aosorio Exp $
#ifndef PROCESSDIGIGLOBALSIGNAL_H 
#define PROCESSDIGIGLOBALSIGNAL_H 1

// Include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "Geometry/RPCGeometry/interface/RPCRoll.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"

#include "DataFormats/RPCDigi/interface/RPCDigi.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"

#include "L1Trigger/RPCTechnicalTrigger/interface/TTUInput.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/RPCInputSignal.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/RPCData.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/ProcessInputSignal.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/RPCWheelMap.h"

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <ios>
#include <cmath>
#include <map>
#include <vector>

/** @class ProcessDigiGlobalSignal ProcessDigiGlobalSignal.h
 *  
 *
 *  @author Andres Felipe Osorio Oliveros
 *  @date   2008-11-21
 */
class ProcessDigiGlobalSignal : public ProcessInputSignal {
public: 
  /// Standard constructor
  ProcessDigiGlobalSignal( ) {};
  
  ProcessDigiGlobalSignal( const edm::ESHandle<RPCGeometry> &, 
                           const edm::Handle<RPCDigiCollection> & );
  
  virtual ~ProcessDigiGlobalSignal( ); ///< Destructor
  
  int  next();
  
  void reset();
  
  RPCInputSignal * retrievedata() {
    return  m_wmin;
  };
  
  void rewind() {};
  void showfirst() {};
  
protected:
  
private:

  int getBarrelLayer(const int &, const int &);
    
  TTUInput * m_ttuwheelmap;
  RPCInputSignal * m_wmin;
  
  std::map<int, TTUInput*> m_data;
  std::map<int, RPCWheelMap*> m_wheelMapVec;
  
  const edm::ESHandle<RPCGeometry>     * m_ptr_rpcGeom;
  const edm::Handle<RPCDigiCollection> * m_ptr_digiColl;
  
  RPCDigiCollection::const_iterator m_digiItr;
  RPCDigiCollection::DigiRangeIterator m_detUnitItr;

  int m_maxBx;
    
  bool m_debug;
        
};
#endif // PROCESSDIGIGLOBALSIGNAL_H
