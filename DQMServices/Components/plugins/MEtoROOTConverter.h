#ifndef MEtoROOTConverter_h
#define MEtoROOTConverter_h

/** \class MEtoROOTConverter
 *  
 *  Class to take dqm monitor elements and convert into a
 *  ROOT dataformat stored in Run tree of edm file
 *
 *  $Date: 2007/11/20 23:53:44 $
 *  $Revision: 1.2 $
 *  \author M. Strang SUNY-Buffalo
 */

// framework & common header files
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//DQM services
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

// data format
#include "DataFormats/Histograms/interface/MEtoROOTFormat.h"

// helper files
#include <iostream>
#include <stdlib.h>
#include <string>
#include <memory>
#include <vector>

#include "TString.h"
#include "TH1F.h"

class MEtoROOTConverter : public edm::EDProducer
{

 public:

  explicit MEtoROOTConverter(const edm::ParameterSet&);
  virtual ~MEtoROOTConverter();
  virtual void beginJob(const edm::EventSetup&);
  virtual void endJob();  
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void beginRun(edm::Run&, const edm::EventSetup&);
  virtual void endRun(edm::Run&, const edm::EventSetup&);

 private:
  
  std::string fName;
  int verbosity;
  int frequency;

  DaqMonitorBEInterface *dbe;

  // private statistics information
  unsigned int count;

}; // end class declaration

#endif
