// -*- C++ -*-
//
// Package:    METCorrectorDBReader
// Class:      
// 
/**\class METCorrectorDBReader

 Description: Reads out *.db format for MET corrections 

 Implementation:
     <Notes on implementation>
*/
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
#include "CondFormats/JetMETObjects/interface/METCorrectorParameters.h"
#include "JetMETCorrections/Objects/interface/METCorrectionsRecord.h"
#include "JetMETCorrections/Objects/interface/JetCorrectionsRecord.h"


//
// class declaration
//

class METCorrectorDBReader : public edm::EDAnalyzer {
public:
  explicit METCorrectorDBReader(const edm::ParameterSet&);
  ~METCorrectorDBReader();
  
  
private:
  virtual void beginJob() override ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
  virtual void endJob() override ;
 
  std::string mPayloadName,mGlobalTag;
  bool mCreateTextFile,mPrintScreen;
};


METCorrectorDBReader::METCorrectorDBReader(const edm::ParameterSet& iConfig)
{
  mPayloadName    = iConfig.getUntrackedParameter<std::string>("payloadName");
  mGlobalTag      = iConfig.getUntrackedParameter<std::string>("globalTag");  
  mPrintScreen    = iConfig.getUntrackedParameter<bool>("printScreen");
  mCreateTextFile = iConfig.getUntrackedParameter<bool>("createTextFile");
}


METCorrectorDBReader::~METCorrectorDBReader()
{
 
}

void METCorrectorDBReader::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  edm::ESHandle<METCorrectorParametersCollection> METCorParamsColl;
  std::cout <<"Inspecting MET payload with label: "<< mPayloadName <<std::endl;
  //iSetup.get<JetCorrectionsRecord>().get(mPayloadName,METCorParamsColl);
  iSetup.get<METCorrectionsRecord>().get(mPayloadName,METCorParamsColl);
  std::cout<<"hahahahahahahah"<<std::endl;
  //METCorParamsColl.~ESHandle();

  // get the sections from Collection (pair of section and METCorr.Par class)
  std::vector<METCorrectorParametersCollection::key_type> keys;
  // save level to keys for each METParameter in METParameter collection
  METCorParamsColl->validKeys( keys );
  //METCorParamsColl->validSections( sections );
  for ( std::vector<METCorrectorParametersCollection::key_type>::const_iterator ibegin = keys.begin(),
	  iend = keys.end(), ikey = ibegin; ikey != iend; ++ikey ) {
    std::cout<<"--------------------------------------" << std::endl;
    std::cout<<"Processing key = " << *ikey << std::endl;
    std::string sectionName= METCorParamsColl->findLabel(*ikey);
    std::cout<<"object label: "<<sectionName<<std::endl;
    METCorrectorParameters const & METCorParams = (*METCorParamsColl)[*ikey];

    if (mCreateTextFile)
      {
        if(METCorParamsColl->isXYshift(*ikey) )
        {
	  std::string outFileName(mGlobalTag+"_XYshift_"+mPayloadName+".txt");
	  std::cout<<"outFileName: "<<outFileName<<std::endl;
          //std::cout<<"Writing METCorrectorParameter to txt file: "<<mGlobalTag+"_XYshift_"+mPayloadName+".txt"<<std::endl;
          METCorParams.printFile(outFileName, sectionName);
        }
      }
    
    if (mPrintScreen)
    {
      METCorParams.printScreen(sectionName);
    }
  }

  std::cout<<"Finished hahaha--------------------------" << std::endl;
}

void 
METCorrectorDBReader::beginJob()
{
}

void 
METCorrectorDBReader::endJob() 
{

}

//define this as a plug-in
DEFINE_FWK_MODULE(METCorrectorDBReader);
