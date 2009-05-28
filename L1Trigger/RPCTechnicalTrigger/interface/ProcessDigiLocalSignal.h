// $Id: ProcessDigiLocalSignal.h,v 1.2 2009/05/24 21:45:39 aosorio Exp $
#ifndef PROCESSDIGILOCALSIGNAL_H 
#define PROCESSDIGILOCALSIGNAL_H 1

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

/** @class ProcessDigiLocalSignal ProcessDigiLocalSignal.h
 *  
 *
 *  @author Andres Felipe Osorio Oliveros
 *  @date   2009-04-15
 */
class ProcessDigiLocalSignal : public ProcessInputSignal {
public: 
  /// Standard constructor
  ProcessDigiLocalSignal( ) {};

  ProcessDigiLocalSignal( const edm::ESHandle<RPCGeometry> &, 
                          const edm::Handle<RPCDigiCollection> & );

  virtual ~ProcessDigiLocalSignal( ); ///< Destructor
  
  int  next();
  
  void reset();
  
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
    
};
#endif // PROCESSDIGILOCALSIGNAL_H
