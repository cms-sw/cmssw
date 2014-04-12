// $Id: 
#ifndef RBCPROCESSRPCDIGIS_H 
#define RBCPROCESSRPCDIGIS_H 1

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

// From project
#include "L1Trigger/RPCTechnicalTrigger/interface/RBCInput.h" 
#include "L1Trigger/RPCTechnicalTrigger/interface/RPCInputSignal.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/RPCData.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/ProcessInputSignal.h"

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <ios>
#include <cmath>
#include <map>
#include <vector>

/** @class RBCProcessRPCDigis RBCProcessRPCDigis.h
 *  
 *
 *  @author Andres Felipe Osorio Oliveros
 *  @date   2009-04-15
 */

class RBCProcessRPCDigis : public ProcessInputSignal {
public: 
  /// Standard constructor
  RBCProcessRPCDigis( ) {};
  
  RBCProcessRPCDigis( const edm::ESHandle<RPCGeometry> &, 
                      const edm::Handle<RPCDigiCollection> & );
  
  virtual ~RBCProcessRPCDigis( ); ///< Destructor
  
  int  next();
  
  void reset();
  
  void configure();
    
  void initialize( std::vector<RPCData*> & );
  
  void builddata();
  
  void print_output();
  
  RPCInputSignal * retrievedata() {
    return  m_lbin;
  };
  
  void rewind() {};
  void showfirst() {};

protected:
  
private:
  
  int getBarrelLayer(const int &, const int &);
  
  void setDigiAt( int , int  );
  
  void setInputBit( std::bitset<15> & , int );
  
  const edm::ESHandle<RPCGeometry>     * m_ptr_rpcGeom;
  const edm::Handle<RPCDigiCollection> * m_ptr_digiColl;
  
  RPCDigiCollection::const_iterator m_digiItr;
  RPCDigiCollection::DigiRangeIterator m_detUnitItr;
  
  RPCData  * m_block;
  
  RPCInputSignal * m_lbin;
  
  std::map<int, int> m_layermap;
  
  std::map<int, RBCInput*> m_data;
  
  std::map<int, std::vector<RPCData*> > m_vecDataperBx;
  
  bool m_debug;
  int m_maxBxWindow;
  
  std::vector<int> m_wheelid;
  std::vector<int> m_sec1id;
  std::vector<int> m_sec2id;
  
  std::map<int, l1trigger::Counters*> m_digiCounters;
      
};
#endif // RBCPROCESSRPCDIGIS_H
