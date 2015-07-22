#ifndef EDMtoMEConverter_h
#define EDMtoMEConverter_h

/** \class EDMtoMEConverter
 *  
 *  Class to take dqm monitor elements and convert into a
 *  ROOT dataformat stored in Run tree of edm file
 *
 *  \author M. Strang SUNY-Buffalo
 */

// framework & common header files
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/FileBlock.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"

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
#include "TList.h"

#include "classlib/utils/StringList.h"
#include "classlib/utils/StringOps.h"

class EDMtoMEConverter : public edm::EDAnalyzer
{

 public:

  explicit EDMtoMEConverter(const edm::ParameterSet&);
  virtual ~EDMtoMEConverter();
  virtual void beginJob();
  virtual void endJob();  
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void beginRun(const edm::Run&, const edm::EventSetup&);
  virtual void endRun(const edm::Run&, const edm::EventSetup&);
  virtual void beginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&);
  virtual void endLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&);
  virtual void respondToOpenInputFile(const edm::FileBlock&);

  template <class T>
  void getData(T& iGetFrom, bool iEndRun);

  typedef std::vector<uint32_t> TagList;

 private:
  
  std::string name;
  int verbosity;
  int frequency;

  bool convertOnEndLumi;
  bool convertOnEndRun;

  DQMStore *dbe;
  std::vector<MonitorElement*> me1, me2, me3, me4, me5, me6, me7, me8;

  // private statistics information
  unsigned int iCountf;
  std::map<int,int> iCount;

  std::vector<std::string> classtypes;

  edm::EDGetTokenT<MEtoEDM<TH1F> > runInputTagTH1F_;
  edm::EDGetTokenT<MEtoEDM<TH1F> > lumiInputTagTH1F_;

  edm::EDGetTokenT<MEtoEDM<TH1S> > runInputTagTH1S_;
  edm::EDGetTokenT<MEtoEDM<TH1S> > lumiInputTagTH1S_;

  edm::EDGetTokenT<MEtoEDM<TH1D> > runInputTagTH1D_;
  edm::EDGetTokenT<MEtoEDM<TH1D> > lumiInputTagTH1D_;

  edm::EDGetTokenT<MEtoEDM<TH2F> > runInputTagTH2F_;
  edm::EDGetTokenT<MEtoEDM<TH2F> > lumiInputTagTH2F_;

  edm::EDGetTokenT<MEtoEDM<TH2S> > runInputTagTH2S_;
  edm::EDGetTokenT<MEtoEDM<TH2S> > lumiInputTagTH2S_;

  edm::EDGetTokenT<MEtoEDM<TH2D> > runInputTagTH2D_;
  edm::EDGetTokenT<MEtoEDM<TH2D> > lumiInputTagTH2D_;

  edm::EDGetTokenT<MEtoEDM<TH3F> > runInputTagTH3F_;
  edm::EDGetTokenT<MEtoEDM<TH3F> > lumiInputTagTH3F_;

  edm::EDGetTokenT<MEtoEDM<TProfile> > runInputTagTProfile_;
  edm::EDGetTokenT<MEtoEDM<TProfile> > lumiInputTagTProfile_;

  edm::EDGetTokenT<MEtoEDM<TProfile2D> > runInputTagTProfile2D_;
  edm::EDGetTokenT<MEtoEDM<TProfile2D> > lumiInputTagTProfile2D_;

  edm::EDGetTokenT<MEtoEDM<double> > runInputTagDouble_;
  edm::EDGetTokenT<MEtoEDM<double> > lumiInputTagDouble_;

  edm::EDGetTokenT<MEtoEDM<int> > runInputTagInt_;
  edm::EDGetTokenT<MEtoEDM<int> > lumiInputTagInt_;

  edm::EDGetTokenT<MEtoEDM<long long> > runInputTagInt64_;
  edm::EDGetTokenT<MEtoEDM<long long> > lumiInputTagInt64_;

  edm::EDGetTokenT<MEtoEDM<TString> > runInputTagString_;
  edm::EDGetTokenT<MEtoEDM<TString> > lumiInputTagString_;


}; // end class declaration

#endif
