// -*- C++ -*-
//
// Package:    MEtoMEComparitor
// Class:      MEtoMEComparitor
// 
/**\class MEtoMEComparitor MEtoMEComparitor.cc DQMOffline/MEtoMEComparitor/src/MEtoMEComparitor.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  jean-roch Vlimant,40 3-A28,+41227671209,
//         Created:  Tue Nov 30 18:55:50 CET 2010
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"


#include <TH1F.h>
#include <TH1D.h>

//
// class declaration
//

class MEtoMEComparitor : public edm::EDAnalyzer {
   public:
      explicit MEtoMEComparitor(const edm::ParameterSet&);
      ~MEtoMEComparitor() override;


   private:
      void beginJob() override ;
      void analyze(const edm::Event&, const edm::EventSetup&) override{}
      void beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) override;
      void endRun(const edm::Run& iRun, const edm::EventSetup& iSetup) override;
      void endLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&) override;
      void endJob() override ;

  template <class W,class T> void compare(const W& where,const std::string & instance);
  template <class T> void book(const std::string & directory,const std::string & type, const T * h);
  template <class T> void keepBadHistograms(const std::string & directory, const T * h_new, const T * h_ref);

  DQMStore * _dbe;
  std::string _moduleLabel;
  
  std::string _lumiInstance;
  std::string _runInstance;

  std::string _process_ref;
  std::string _process_new;
  bool _autoProcess;

  double _KSgoodness;
  double _diffgoodness;
  unsigned int _dirDepth;
  double _overallgoodness;

};
