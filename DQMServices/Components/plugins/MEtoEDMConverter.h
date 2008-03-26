#ifndef MEtoEDMConverter_h
#define MEtoEDMConverter_h

/** \class MEtoEDMConverter
 *  
 *  Class to take dqm monitor elements and convert into a
 *  ROOT dataformat stored in Run tree of edm file
 *
 *  $Date: 2008/02/13 22:40:49 $
 *  $Revision: 1.3 $
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
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/CoreROOT/interface/MonitorElementRootT.h"
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
#include "TH2F.h"
#include "TH3F.h"
#include "TProfile.h"
#include "TProfile2D.h"
#include "TObjString.h"

#include "classlib/utils/StringList.h"
#include "classlib/utils/StringOps.h"

using namespace lat;

template <class T>
class mestorage {

 public:
  std::vector<std::string> name;
  std::vector<std::vector<uint32_t> > tags;
  std::vector<T> object;
};

class MEtoEDMConverter : public edm::EDProducer
{

 public:

  explicit MEtoEDMConverter(const edm::ParameterSet&);
  virtual ~MEtoEDMConverter();
  virtual void beginJob(const edm::EventSetup&);
  virtual void endJob();  
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void beginRun(edm::Run&, const edm::EventSetup&);
  virtual void endRun(edm::Run&, const edm::EventSetup&);

  typedef std::vector<uint32_t> TagList;

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
  StringList fullpathvec; //full path of monitor element
  StringList metype; //monitor element type
  bool hasTH1F;
  bool hasTH2F;
  bool hasTH3F;
  bool hasTProfile;
  bool hasTProfile2D;
  bool hasFloat;
  bool hasInt;
  bool hasString;
  int nTH1F;
  int nTH2F;
  int nTH3F;
  int nTProfile;
  int nTProfile2D;
  int nFloat;
  int nInt;
  int nString;

  // persistent MERoot information
  TagList taglist; // to be stored in tags

  mestorage<TH1F> TH1FME;
  mestorage<TH2F> TH2FME;
  mestorage<TH3F> TH3FME;
  mestorage<TProfile> TProfileME;
  mestorage<TProfile2D> TProfile2DME;
  mestorage<float> FloatME;
  mestorage<int> IntME;
  mestorage<TString> StringME;

  // private statistics information
  std::map<int,int> count;

}; // end class declaration

#endif
