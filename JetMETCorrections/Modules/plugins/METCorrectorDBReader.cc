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
#include "FWCore/MessageLogger/interface/MessageLogger.h"
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
namespace{

class METCorrectorDBReader : public edm::one::EDAnalyzer<> {
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

}


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
  edm::LogInfo("METCorrectorDBReader") <<"Inspecting MET payload with label: "<< mPayloadName;
  iSetup.get<METCorrectionsRecord>().get(mPayloadName,METCorParamsColl);

  // get the sections from Collection (pair of section and METCorr.Par class)
  std::vector<METCorrectorParametersCollection::key_type> keys;
  // save level to keys for each METParameter in METParameter collection
  METCorParamsColl->validKeys( keys );
  //METCorParamsColl->validSections( sections );
  for ( std::vector<METCorrectorParametersCollection::key_type>::const_iterator 
	  ikey = keys.begin(); ikey != keys.end(); ++ikey ) {
    std::string sectionName= METCorParamsColl->findLabel(*ikey);
    edm::LogInfo("METCorrectorDBReader")
    	<<"Processing key = " << *ikey
    	<<"object label: "<<sectionName;
    METCorrectorParameters const & METCorParams = (*METCorParamsColl)[*ikey];

    if (mCreateTextFile)
    {
	std::string outFileName(mGlobalTag+"_XYshift");
	std::string shiftType("MC");

        if(METCorParamsColl->isXYshiftMC(*ikey) )
        {
	  shiftType = "MC";
        }else if(METCorParamsColl->isXYshiftDY(*ikey) )
        {
	  shiftType = "DY";
        }else if(METCorParamsColl->isXYshiftTTJets(*ikey) )
        {
	  shiftType = "TTJets";
        }else if(METCorParamsColl->isXYshiftWJets(*ikey) )
        {
	  shiftType = "WTJets";
        }else if(METCorParamsColl->isXYshiftData(*ikey) )
        {
	  shiftType = "Data";
	}else{
	  throw cms::Exception("InvalidKey") <<
	    "************** Can't interpret the stored key: "<<*ikey<<std::endl;
	}
	outFileName += shiftType + "_" + mPayloadName + ".txt";
	edm::LogInfo("METCorrectorDBReader")<<"outFileName: "<<outFileName;
        METCorParams.printFile(outFileName, sectionName);
    }
    
    if (mPrintScreen)
    {
      edm::LogInfo("METCorrectorDBReader")<<"Level: "<<METCorParamsColl->levelName(*ikey);
      METCorParams.printScreen(sectionName);
    }
  }

  edm::LogInfo("METCorrectorDBReader")<<"Finished --------------------------";
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
