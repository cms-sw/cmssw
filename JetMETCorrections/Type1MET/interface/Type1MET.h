#ifndef Type1MET_Type1MET_h
#define Type1MET_Type1MET_h
// -*- C++ -*-
//
// Package:    Type1MET
// Class:      Type1MET
// 
/**\class Type1MET Type1MET.cc JetMETCorrections/Type1MET/src/Type1MET.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Oct 12 08:23
//         Created:  Wed Oct 12 12:16:04 CDT 2005
// $Id: Type1MET.h,v 1.2 2010/05/16 15:21:59 jdamgov Exp $
//
//


// system include files
#include <memory>
#include <string.h>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "JetMETCorrections/Type1MET/interface/Type1METAlgo.h"



namespace cms 
{
  // PRODUCER CLASS DEFINITION -------------------------------------
  class Type1MET : public edm::EDProducer 
  {
  public:
    explicit Type1MET( const edm::ParameterSet& );
    explicit Type1MET();
    virtual ~Type1MET();
    virtual void produce( edm::Event&, const edm::EventSetup& );
  private:
    Type1METAlgo alg_;
    std::string metType;
    std::string inputUncorMetLabel;
    std::string inputUncorJetsLabel;
    std::string correctorLabel;
    double jetPTthreshold;
    double jetEMfracLimit;
    double UscaleA;
    double UscaleB;
    double UscaleC;
    bool useTypeII;
    bool hasMuonsCorr;
  };
}
#endif
