// $Id: RPCTechnicalTrigger.h,v 1.1 2009/02/05 13:46:21 aosorio Exp $
#ifndef RPCTECHNICALTRIGGER_H 
#define RPCTECHNICALTRIGGER_H 1

// system include files
#include <memory>
#include <bitset>

// Include files From CMSSW

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "Geometry/RPCGeometry/interface/RPCRoll.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"

#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"

#include "DataFormats/RPCDigi/interface/RPCDigi.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"

#include "PhysicsTools/UtilAlgos/interface/TFileService.h"

//Local to project
#include "L1Trigger/RPCTechnicalTrigger/interface/ProcessInputSignal.h"
#include "L1Trigger/RPCTechnicalTrigger/src/TTUEmulator.h"
#include "L1Trigger/RPCTechnicalTrigger/src/HistoOutput.h"

#include "CondFormats/RPCObjects/interface/RBCBoardSpecs.h"
#include "CondFormats/DataRecord/interface/RBCBoardSpecsRcd.h"

#include "CondFormats/RPCObjects/interface/TTUBoardSpecs.h"
#include "CondFormats/DataRecord/interface/TTUBoardSpecsRcd.h"

//From ROOT

/** @class RPCTechnicalTrigger RPCTechnicalTrigger.h
 *  
 *
 *  @author Andres Osorio
 *
 *  email: aosorio@uniandes.edu.co
 *
 *  @date   2008-10-15
 */

class RPCTechnicalTrigger : public edm::EDAnalyzer {
public: 
  /// Standard constructor
  explicit RPCTechnicalTrigger(const edm::ParameterSet&);
  
  ~RPCTechnicalTrigger( ); ///< Destructor
  
  
protected:
  
private:
  
  virtual void beginJob(const edm::EventSetup&);
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  void validate( const edm::Event&, const edm::EventSetup&, int & );
  
  int  discriminateGMT( const edm::Event&, const edm::EventSetup& );
  
  void makeGMTFilterDist( );
      
  void printinfo();
  
  TTUEmulator * m_ttu[3];
  
  RPCInputSignal * m_input;
  
  ProcessInputSignal * m_signal;

  std::bitset<1> m_trigger;
  std::bitset<6> m_triggerbits;
  std::bitset<8> m_gmtfilter;
  
  edm::ESHandle<RPCGeometry> m_rpcGeometry;
  
  int m_debugmode;
  int m_validation;
  std::string m_rbclogictype;
  std::string m_ttulogictype;
  
  edm::InputTag m_GMTLabel;
  
  const TTUBoardSpecs * m_ttuspecs;
  const RBCBoardSpecs * m_rbcspecs;
  
  HistoOutput * m_houtput;
  
  int m_ievt;
  int m_cand;
  int m_valflag;
  
};
#endif // RPCTECHNICALTRIGGER_H
