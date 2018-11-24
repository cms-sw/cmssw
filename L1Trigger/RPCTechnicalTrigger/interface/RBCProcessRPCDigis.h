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

#include <cstdlib>
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
  RBCProcessRPCDigis( const edm::ESHandle<RPCGeometry> &, 
                      const edm::Handle<RPCDigiCollection> & );
  
  ~RBCProcessRPCDigis( ) override; ///< Destructor
  
  int  next() override;
  
  void reset();
  
  void configure();
    
  
  void builddata();
  
  void print_output();
  
  RPCInputSignal * retrievedata() override {
    return  m_lbin.get();
  };
  
  void rewind() {};
  void showfirst() {};

protected:
  
private:
  void initialize( std::vector<RPCData> & ) const;
  
  int getBarrelLayer(const int &, const int &);
  
  void setDigiAt( int , int , RPCData& );
  
  void setInputBit( std::bitset<15> & , int );
  
  const edm::ESHandle<RPCGeometry>     * m_ptr_rpcGeom;
  const edm::Handle<RPCDigiCollection> * m_ptr_digiColl;
  
  std::unique_ptr<RPCInputSignal> m_lbin;
  
  std::map<int, RBCInput*> m_data;
  
  std::map<int, std::vector<RPCData> > m_vecDataperBx;
  

  std::map<int, l1trigger::Counters> m_digiCounters;
  const int m_maxBxWindow;
  const bool m_debug;
      
};
#endif // RBCPROCESSRPCDIGIS_H
