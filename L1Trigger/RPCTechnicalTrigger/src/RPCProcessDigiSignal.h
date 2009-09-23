// $Id: RPCProcessDigiSignal.h,v 1.2 2009/02/16 09:11:30 aosorio Exp $
#ifndef RPCPROCESSDIGISIGNAL_H 
#define RPCPROCESSDIGISIGNAL_H 1

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
#include "L1Trigger/RPCTechnicalTrigger/src/RPCWheelMap.h"

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <ios>
#include <cmath>
#include <map>
#include <vector>

/** @class RPCProcessDigiSignal RPCProcessDigiSignal.h
 *  
 *
 *  @author Andres Felipe Osorio Oliveros
 *  @date   2008-11-21
 */
class RPCProcessDigiSignal : public ProcessInputSignal {
public: 
  /// Standard constructor
  RPCProcessDigiSignal( ) {};
  
  RPCProcessDigiSignal( const edm::ESHandle<RPCGeometry> &, 
                        const edm::Handle<RPCDigiCollection> & );
  
  virtual ~RPCProcessDigiSignal( ); ///< Destructor
  
  int  next();
  
  void rewind() {};
  
  void showfirst() {};
  
  void reset() {};
  
  RPCInputSignal * retrievedata() {
    return  m_wmin;
  };
  
protected:
  
private:

  int getBarrelLayer(const int &, const int &);
    
  TTUInput * m_ttuwheelmap;
  RPCInputSignal * m_wmin;
  
  std::vector<RPCWheelMap*> m_wheelmapvec;
  std::map<int, TTUInput*> m_data;
  
  const edm::ESHandle<RPCGeometry>     * m_ptr_rpcGeom;
  const edm::Handle<RPCDigiCollection> * m_ptr_digiColl;
  
  RPCDigiCollection::const_iterator m_digiItr;
  RPCDigiCollection::DigiRangeIterator m_detUnitItr;

      
};
#endif // RPCPROCESSDIGISIGNAL_H
