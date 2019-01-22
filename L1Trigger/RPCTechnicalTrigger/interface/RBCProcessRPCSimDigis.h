#ifndef INTERFACE_RBCPROCESSRPCSIMDIGIS_H 
#define INTERFACE_RBCPROCESSRPCSIMDIGIS_H 1

// Include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "Geometry/RPCGeometry/interface/RPCRoll.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"

#include "DataFormats/MuonDetId/interface/RPCDetId.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "SimDataFormats/RPCDigiSimLink/interface/RPCDigiSimLink.h"

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
#include <memory>

/** @class RBCProcessRPCSimDigis RBCProcessRPCSimDigis.h interface/RBCProcessRPCSimDigis.h
 *  
 *
 *  @author Andres Felipe Osorio Oliveros
 *  @date   2009-09-20
 */

class RBCProcessRPCSimDigis : public ProcessInputSignal {
public: 
  /// Standard constructor
  RBCProcessRPCSimDigis(); 
  
  RBCProcessRPCSimDigis( const edm::ESHandle<RPCGeometry> &, 
                         const edm::Handle<edm::DetSetVector<RPCDigiSimLink> > & );
  
  int  next() override;
  
  void reset();
  
  void builddata();
  
  void print_output();
  
  RPCInputSignal * retrievedata() override {
    return  m_lbin.get();
  };
  
  void rewind() {};
  void showfirst() {};
  
  ~RBCProcessRPCSimDigis( ) override; ///< Destructor
  
protected:
  
private:
  
  int getBarrelLayer(const int &, const int &);
  
  void setDigiAt( int , int , RPCData& );
  
  void setInputBit( std::bitset<15> & , int );
  
  void initialize( std::vector<RPCData> & );
  
  const edm::ESHandle<RPCGeometry> * m_ptr_rpcGeom;
  const edm::Handle<edm::DetSetVector<RPCDigiSimLink> > * m_ptr_digiSimLink;
  
  edm::DetSetVector<RPCDigiSimLink>::const_iterator m_linkItr;
  edm::DetSet<RPCDigiSimLink>::const_iterator m_digiItr;
    
  std::unique_ptr<RPCInputSignal> m_lbin;
  
  std::map<int, RBCInput*> m_data;
  
  std::map<int, std::vector<RPCData> > m_vecDataperBx;
  
  bool m_debug;
  int m_maxBxWindow;
  
  
};
#endif // INTERFACE_RBCPROCESSRPCSIMDIGIS_H
