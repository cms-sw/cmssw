#ifndef L1TdeCSCTF_h
#define L1TdeCSCTF_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "CondFormats/L1TObjects/interface/L1MuTriggerScales.h"
#include "L1Trigger/CSCTrackFinder/interface/CSCSectorReceiverLUT.h"
#include "L1Trigger/CSCTrackFinder/interface/CSCTFPtLUT.h"

//data formats
#include "DataFormats/L1CSCTrackFinder/interface/L1CSCTrackCollection.h"
#include "DataFormats/L1CSCTrackFinder/interface/CSCTriggerContainer.h"


#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "L1Trigger/CSCTrackFinder/src/CSCTFDTReceiver.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <unistd.h>

#include "TTree.h"
#include "TFile.h"

class L1TdeCSCTF : public thread_unsafe::DQMEDAnalyzer {
private:
  edm::EDGetTokenT<L1CSCTrackCollection> dataTrackProducer;
  edm::EDGetTokenT<L1CSCTrackCollection> emulTrackProducer;
  edm::EDGetTokenT<CSCTriggerContainer<csctf::TrackStub> > dataStubProducer;
  edm::EDGetTokenT<L1MuDTChambPhContainer> emulStubProducer;

  const L1MuTriggerScales *ts;
  CSCTFPtLUT* ptLUT_;
  edm::ParameterSet ptLUTset;
  CSCTFDTReceiver* my_dtrc;
	
  // Define Monitor Element Histograms
  ////////////////////////////////////
  MonitorElement* phiComp, *etaComp, *occComp, *ptComp, *qualComp;
  MonitorElement* pt1Comp, *pt2Comp, *pt3Comp, *pt4Comp, *pt5Comp, *pt6Comp;
  MonitorElement* dtStubPhi, *badDtStubSector;
	
  MonitorElement* phiComp_1d, *etaComp_1d, *occComp_1d, *ptComp_1d, *qualComp_1d;
  MonitorElement* pt1Comp_1d, *pt2Comp_1d, *pt3Comp_1d, *pt4Comp_1d, *pt5Comp_1d, *pt6Comp_1d;
  MonitorElement* dtStubPhi_1d;
	
  // dqm folder name
  //////////////////
   std::string m_dirName;
   std::string outFile;
	

protected:
  void analyze(edm::Event const& e, edm::EventSetup const& iSetup);
  virtual void bookHistograms(DQMStore::IBooker &ibooker, edm::Run const&, edm::EventSetup const&) override;
  virtual void dqmBeginRun(edm::Run const&, edm::EventSetup const&);
  virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);

public:
  explicit L1TdeCSCTF(edm::ParameterSet const& pset);
  virtual ~L1TdeCSCTF() {}
};

#endif

