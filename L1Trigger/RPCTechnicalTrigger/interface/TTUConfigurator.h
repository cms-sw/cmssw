// $Id: TTUConfigurator.h,v 1.2 2009/10/29 20:04:03 ghete Exp $
#ifndef TTUCONFIGURATOR_H 
#define TTUCONFIGURATOR_H 1

// Include files

/** @class TTUConfigurator TTUConfigurator.h
 *
 *  
 *  This is an auxiliary class to read an ascii or xml configuration file
 *  for the RPC Technical Trigger - to by pass reading configuration from
 *  database via EventSetup
 *
 *
 *  @author Andres Felipe Osorio Oliveros
 *  @date   2009-06-02
 */

#include <iostream>
#include <fstream>
#include <ios>

#include "CondFormats/RPCObjects/interface/RBCBoardSpecs.h"
#include "CondFormats/DataRecord/interface/RBCBoardSpecsRcd.h"
#include "CondFormats/RPCObjects/interface/TTUBoardSpecs.h"
#include "CondFormats/DataRecord/interface/TTUBoardSpecsRcd.h"

#include "L1Trigger/RPCTechnicalTrigger/interface/RBCBoardSpecsIO.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/TTUBoardSpecsIO.h"

class TTUConfigurator {
public: 
  /// Standard constructor
  TTUConfigurator( ) { };
  
  TTUConfigurator( const std::string& );
  
  virtual ~TTUConfigurator( ); ///< Destructor
  
  RBCBoardSpecs * getRbcSpecs(){ return m_rbcspecs; };
  
  TTUBoardSpecs * getTtuSpecs(){ return m_ttuspecs; };
  
  void process();
  
  bool m_hasConfig;
  
protected:
  
private:
  
  std::ifstream * m_in;
  
  void addData( RBCBoardSpecs *  );
  void addData( TTUBoardSpecs *  );
  
  RBCBoardSpecs * m_rbcspecs;
  TTUBoardSpecs * m_ttuspecs;
  
};
#endif // TTUCONFIGURATOR_H
