// -*- C++ -*-
//
// Package:    JetCorrectorDBReader
// Class:      
// 
/**\class JetCorrectorDBReader

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Benedikt Hegner 
//         Created:  Tue Mar 09 01:32:51 CET 2010
// $Id: JetCorrectorDBReader.cc,v 1.3.2.2 2010/10/05 14:22:26 kkousour Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/JetMETObjects/interface/JetCorrectorParameters.h"
#include "JetMETCorrections/Objects/interface/JetCorrectionsRecord.h"
//
// class declaration
//

class JetCorrectorDBReader : public edm::EDAnalyzer {
public:
  explicit JetCorrectorDBReader(const edm::ParameterSet&);
  ~JetCorrectorDBReader();
  
  
private:
  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
 
  std::string mPayloadName;
  bool mCreateTextFile,mPrintScreen;
};


JetCorrectorDBReader::JetCorrectorDBReader(const edm::ParameterSet& iConfig)
{
  mPayloadName    = iConfig.getUntrackedParameter<std::string>("payloadName");
  mPrintScreen    = iConfig.getUntrackedParameter<bool>("printScreen");
  mCreateTextFile = iConfig.getUntrackedParameter<bool>("createTextFile");
}


JetCorrectorDBReader::~JetCorrectorDBReader()
{
 
}

void JetCorrectorDBReader::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  edm::ESHandle<JetCorrectorParametersCollection> JetCorParamsColl;
  std::cout <<"Inspecting JEC payload with label: "<< mPayloadName <<std::endl;
  iSetup.get<JetCorrectionsRecord>().get(mPayloadName,JetCorParamsColl);
  std::vector<JetCorrectorParametersCollection::key_type> keys;
  JetCorParamsColl->validKeys( keys );
  for ( std::vector<JetCorrectorParametersCollection::key_type>::const_iterator ibegin = keys.begin(),
	  iend = keys.end(), ikey = ibegin; ikey != iend; ++ikey ) {
    JetCorrectorParameters const & JetCorParams = (*JetCorParamsColl)[*ikey];
    std::cout<<"object label: "<<JetCorParamsColl->findLabel(*ikey)<<std::endl;
    if (mCreateTextFile)
      {
	std::cout<<"Creating txt file: "<<mPayloadName+"_"+JetCorParamsColl->findLabel(*ikey)+".txt"<<std::endl;
	JetCorParams.printFile(mPayloadName+"_"+JetCorParamsColl->findLabel(*ikey)+".txt");
      }
    if (mPrintScreen)
      JetCorParams.printScreen();
  }
}

void 
JetCorrectorDBReader::beginJob()
{
}

void 
JetCorrectorDBReader::endJob() 
{
}

//define this as a plug-in
DEFINE_FWK_MODULE(JetCorrectorDBReader);
