#ifndef MEtoEDMConverter_h
#define MEtoEDMConverter_h

/** \class MEtoEDMConverter
 *  
 *  Class to take dqm monitor elements and convert into a
 *  ROOT dataformat stored in Run tree of edm file
 *
 *  $Date: 2013/05/17 21:56:01 $
 *  $Revision: 1.20 $
 *  \author M. Strang SUNY-Buffalo
 */

// framework & common header files
#include "FWCore/Framework/interface/one/EDProducer.h"
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

class MEtoEDMConverter : public edm::one::EDProducer<edm::one::WatchRuns,
                                                     edm::EndLuminosityBlockProducer,
                                                     edm::EndRunProducer>
{
public:
  explicit MEtoEDMConverter(const edm::ParameterSet&);
  virtual ~MEtoEDMConverter();
  virtual void beginJob() override;
  virtual void endJob() override;  
  virtual void produce(edm::Event&, const edm::EventSetup&) override;
  virtual void beginRun(edm::Run const&, const edm::EventSetup&) override;
  virtual void endRun(edm::Run const&, const edm::EventSetup&) override;
  virtual void endRunProduce(edm::Run&, const edm::EventSetup&) override;
  virtual void endLuminosityBlockProduce(edm::LuminosityBlock&, const edm::EventSetup&) override;

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
