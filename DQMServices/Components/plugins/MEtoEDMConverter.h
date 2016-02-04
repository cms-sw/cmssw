#ifndef MEtoEDMConverter_h
#define MEtoEDMConverter_h

/** \class MEtoEDMConverter
 *  
 *  Class to take dqm monitor elements and convert into a
 *  ROOT dataformat stored in Run tree of edm file
 *
 *  $Date: 2010/09/15 15:50:01 $
 *  $Revision: 1.19 $
 *  \author M. Strang SUNY-Buffalo
 */

// framework & common header files
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//DQM services
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"

// data format
#include "DataFormats/Histograms/interface/MEtoEDMFormat.h"

// helper files
#include <iostream>
#include <stdlib.h>
#include <string>
#include <memory>
#include <vector>
#include <map>
#include <assert.h>
#include <stdint.h>

#include "TString.h"
#include "TH1F.h"
#include "TH1S.h"
#include "TH1D.h"
#include "TH2F.h"
#include "TH2S.h"
#include "TH2D.h"
#include "TH3F.h"
#include "TProfile.h"
#include "TProfile2D.h"
#include "TObjString.h"

class MEtoEDMConverter : public edm::EDProducer
{
public:
  explicit MEtoEDMConverter(const edm::ParameterSet&);
  virtual ~MEtoEDMConverter();
  virtual void beginJob();
  virtual void endJob();  
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void beginRun(edm::Run&, const edm::EventSetup&);
  virtual void endRun(edm::Run&, const edm::EventSetup&);
  virtual void beginLuminosityBlock(edm::LuminosityBlock&, const edm::EventSetup&);
  virtual void endLuminosityBlock(edm::LuminosityBlock&, const edm::EventSetup&);

  template <class T>
  void putData(T& iPutTo, bool iLumiOnly);

  typedef std::vector<uint32_t> TagList;

private:
  std::string fName;
  int verbosity;
  int frequency;
  bool deleteAfterCopy;
  std::string path;

  DQMStore* dbe;

  // private statistics information
  std::map<int,int> iCount;

}; // end class declaration

#endif
