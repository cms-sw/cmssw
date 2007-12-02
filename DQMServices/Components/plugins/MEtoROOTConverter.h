#ifndef MEtoROOTConverter_h
#define MEtoROOTConverter_h

/** \class MEtoROOTConverter
 *  
 *  Class to take dqm monitor elements and convert into a
 *  ROOT dataformat stored in Run tree of edm file
 *
 *  $Date: 2007/11/29 13:36:36 $
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
#include "DQMServices/CoreROOT/interface/MonitorElementRootT.h"

#include "DQMServices/Core/interface/MonitorElement.h"
//#include "DQMServices/Core/interface/MonitorUserInterface.h"
//#include "DQMServices/CoreROOT/interface/DaqMonitorROOTBackEnd.h"
//#include "DQMServices/UI/interface/MonitorUIRoot.h"

//#include "VisMonitoring/DQMServer/interface/Objects.h"

// data format
#include "DataFormats/Histograms/interface/MEtoROOTFormat.h"

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

#include "classlib/utils/StringList.h"
#include "classlib/utils/StringOps.h"
#include "classlib/utils/Time.h"
#include "classlib/utils/TimeInfo.h"

using namespace lat;

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

  typedef std::vector<uint64_t> Int64Vector;
  typedef std::vector<std::string> StringVector;
  typedef std::vector<MEtoROOT::TagList> TagListVector;
  typedef std::vector<TObject*> TObjectVector;
  typedef std::vector<MEtoROOT::QReports> QReportsVector;
  typedef std::vector<uint32_t> Int32Vector;

  typedef MonitorElementT<TNamed> ROOTObj;

 private:
  
  std::string fName;
  int verbosity;
  int frequency;

  DaqMonitorBEInterface *dbe;
  StringList items;
  StringList::iterator i, e, n, m;

  std::map<std::string,int> packages; //keep track just of package names
  std::map<std::string,int>::iterator pkgIter;

  // for each monitor element
  StringList pkgvec;  //package name
  StringList pathvec; //path (without me name)
  StringList mevec;   //monitor element name

  // persistent MERoot information
  MEtoROOT::TagList taglist; // to be stored in tags
  MEtoROOT::QReports qreportsmap; // to be stored in qreports
  uint32_t flag; // to be stored in flags

  Int64Vector version;
  StringVector name;
  TagListVector tags;
  TObjectVector object;
  TObjectVector reference;
  QReportsVector qreports;
  Int32Vector flags;

  // private statistics information
  unsigned int count;

}; // end class declaration

static Time s_version;

static const uint32_t FLAG_ERROR		= 0x1;
static const uint32_t FLAG_WARNING		= 0x2;
static const uint32_t FLAG_REPORT		= 0x4;
static const uint32_t FLAG_SCALAR		= 0x8;
static const uint32_t FLAG_TEXT			= 0x4000000;
static const uint32_t FLAG_DEAD			= 0x8000000;

//static const Regexp s_rxmeval("<(.*)>(i|f|s|qr)=(.*)</\\1>");
//static const Regexp s_rxmeqr("^st\\.(\\d+)\\.(.*)$");

#endif
