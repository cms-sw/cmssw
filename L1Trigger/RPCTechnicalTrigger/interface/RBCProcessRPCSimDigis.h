// $Id: RBCProcessRPCSimDigis.h,v 1.2 2009/12/25 07:05:21 elmer Exp $
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

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <ios>
#include <cmath>
#include <map>
#include <vector>

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
  
  virtual ~RBCProcessRPCSimDigis( ); ///< Destructor
  
protected:
  
private:
  
  int getBarrelLayer(const int &, const int &);
  
  void setDigiAt( int , int  );
  
  void setInputBit( std::bitset<15> & , int );
  
  const edm::ESHandle<RPCGeometry> * m_ptr_rpcGeom;
  const edm::Handle<edm::DetSetVector<RPCDigiSimLink> > * m_ptr_digiSimLink;
  
  edm::DetSetVector<RPCDigiSimLink>::const_iterator m_linkItr;
  edm::DetSet<RPCDigiSimLink>::const_iterator m_digiItr;
    
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

  
};
#endif // INTERFACE_RBCPROCESSRPCSIMDIGIS_H
