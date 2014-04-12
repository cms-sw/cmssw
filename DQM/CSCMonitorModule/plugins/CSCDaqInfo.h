/*
 * =====================================================================================
 *
 *       Filename:  CSCDaqInfo.h
 *
 *    Description:  CSC DAQ Information
 *
 *        Version:  1.0
 *        Created:  12/09/2008 10:53:27 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Valdas Rapsevicius (VR), valdas.rapsevicius@cern.ch
 *        Company:  CERN, CH
 *
 * =====================================================================================
 */

#ifndef CSCDaqInfo_H
#define CSCDaqInfo_H

// system include files
#include <memory>
#include <iostream>
#include <fstream>

// FWCore
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// DQM
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

class CSCDaqInfo : public edm::EDAnalyzer {

  public:

    explicit CSCDaqInfo(const edm::ParameterSet&);
    ~CSCDaqInfo() { }

  private:

    virtual void beginJob();

    virtual void beginLuminosityBlock(const edm::LuminosityBlock& , const  edm::EventSetup&) { }
    virtual void analyze(const edm::Event&, const edm::EventSetup&) { }
    virtual void endLuminosityBlock(const edm::LuminosityBlock& , const  edm::EventSetup&) { }
    virtual void endJob() { }
                    
    std::map<std::string, MonitorElement*> mos;
    DQMStore *dbe;  
                          
};

#endif
