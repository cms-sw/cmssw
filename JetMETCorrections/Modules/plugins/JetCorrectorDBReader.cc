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
// $Id: JetCorrectorDBReader.cc,v 1.3 2010/03/17 15:19:21 kkousour Exp $
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
 
  std::string mLabel;
  bool mCreateTextFile;
};


JetCorrectorDBReader::JetCorrectorDBReader(const edm::ParameterSet& iConfig)
{
  mLabel          = iConfig.getUntrackedParameter<std::string>("label");
  mCreateTextFile = iConfig.getUntrackedParameter<bool>("createTextFile");
}


JetCorrectorDBReader::~JetCorrectorDBReader()
{
 
}

void JetCorrectorDBReader::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  edm::ESHandle<JetCorrectorParametersCollection> JetCorParamsColl;
  std::cout <<"Inspecting jet correction parameters with label: "<< mLabel <<std::endl;
  iSetup.get<JetCorrectionsRecord>().get(mLabel,JetCorParamsColl);

  std::vector<JetCorrectorParametersCollection::key_type> keys;
  JetCorParamsColl->validKeys( keys );
  for ( std::vector<JetCorrectorParametersCollection::key_type>::const_iterator ibegin = keys.begin(),
	  iend = keys.end(), ikey = ibegin; ikey != iend; ++ikey ) {
    JetCorrectorParameters const & JetCorParams = 
      (*JetCorParamsColl)[*ikey];
    if (mCreateTextFile)
      {
	std::cout<<"Creating txt file: "<<mLabel+".txt"<<std::endl;
	JetCorParams.printFile(mLabel+".txt");
      }
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
