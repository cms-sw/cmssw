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
// $Id$
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
#include "DataFormats/Histograms/interface/MEtoEDMFormat.h"

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
      ~MEtoMEComparitor();


   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&){}
      virtual void endRun(const edm::Run& iRun, const edm::EventSetup& iSetup);
      virtual void endLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&);
      virtual void endJob() ;

  template <class W, 
	    //class Wto,
	    class T> void compare(const W& where,const std::string & instance);
  template <class T,class where> void product();

  DQMStore * _dbe;
  edm::InputTag _MEtoEDMTag_ref;
  edm::InputTag _MEtoEDMTag_new;
  std::string _lumiInstance;
  std::string _runInstance;

  double _KSgoodness;
  
      // ----------member data ---------------------------
};
