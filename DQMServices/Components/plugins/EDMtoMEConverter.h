#ifndef EDMtoMEConverter_h
#define EDMtoMEConverter_h

/** \class EDMtoMEConverter
 *  
 *  Class to take dqm monitor elements and convert into a
 *  ROOT dataformat stored in Run tree of edm file
 *
 *  $Date: 2008/03/26 22:11:30 $
 *  $Revision: 1.7 $
 *  \author M. Strang SUNY-Buffalo
 */

// framework & common header files
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/FileBlock.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
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

#include "TString.h"
#include "TH1F.h"

#include "classlib/utils/StringList.h"
#include "classlib/utils/StringOps.h"

using namespace lat;

class EDMtoMEConverter : public edm::EDAnalyzer
{

 public:

  explicit EDMtoMEConverter(const edm::ParameterSet&);
  virtual ~EDMtoMEConverter();
  virtual void beginJob(const edm::EventSetup&);
  virtual void endJob();  
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void beginRun(const edm::Run&, const edm::EventSetup&);
  virtual void endRun(const edm::Run&, const edm::EventSetup&);
  virtual void respondToOpenInputFile(const edm::FileBlock&);

  typedef std::vector<uint32_t> TagList;

 private:
  
  std::string name;
  int verbosity;
  int frequency;

  DQMStore *dbe;
  std::vector<MonitorElement*> me1, me2, me3, me4, me5, me6, me7, me8;

  // release tag
  bool releaseTag;
  
  // private statistics information
  unsigned int countf;
  std::map<int,int> count;

  std::vector<std::string> classtypes;

}; // end class declaration

#endif


