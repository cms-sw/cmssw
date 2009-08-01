#ifndef MEtoEDMConverter_h
#define MEtoEDMConverter_h

/** \class MEtoEDMConverter
 *  
 *  Class to take dqm monitor elements and convert into a
 *  ROOT dataformat stored in Run tree of edm file
 *
 *  $Date: 2009/06/24 17:00:01 $
 *  $Revision: 1.13 $
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
#include "FWCore/Version/interface/GetReleaseVersion.h"

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

#include "TString.h"
#include "TH1F.h"
#include "TH1S.h"
#include "TH2F.h"
#include "TH2S.h"
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

  typedef std::vector<uint32_t> TagList;

private:
  template <class T> struct mestorage
  {
    std::vector<std::string> name;
    std::vector<std::vector<uint32_t> > tags;
    std::vector<T> object;
    std::vector<std::string> release;
    std::vector<int> run;
    std::vector<std::string> datatier;
  };

  
  std::string fName;
  int verbosity;
  int frequency;
  bool deleteAfterCopy;
  std::string path;

  DQMStore *dbe;

  // private statistics information
  std::map<int,int> count;
  std::string datatier;
  bool firstevent;

}; // end class declaration

#endif
