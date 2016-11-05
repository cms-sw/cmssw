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
#include "CondFormats/JetMETObjects/interface/MEtXYcorrectParameters.h"
#include "JetMETCorrections/Objects/interface/MEtXYcorrectRecord.h"
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
  edm::ESHandle<MEtXYcorrectParametersCollection> MEtXYcorParaColl;
  edm::LogInfo("METCorrectorDBReader") <<"Inspecting MET payload with label: "<< mPayloadName;
  iSetup.get<MEtXYcorrectRecord>().get(mPayloadName,MEtXYcorParaColl);

  // get the sections from Collection (pair of section and METCorr.Par class)
  std::vector<MEtXYcorrectParametersCollection::key_type> keys;
  // save level to keys for each METParameter in METParameter collection
  MEtXYcorParaColl->validKeys( keys );
  //MEtXYcorParaColl->validSections( sections );
  for ( std::vector<MEtXYcorrectParametersCollection::key_type>::const_iterator 
	  ikey = keys.begin(); ikey != keys.end(); ++ikey ) {
    std::string sectionName= MEtXYcorParaColl->findLabel(*ikey);
    edm::LogInfo("METCorrectorDBReader")
    	<<"Processing key = " << *ikey
    	<<"object label: "<<sectionName;
    MEtXYcorrectParameters const & MEtXYcorParams = (*MEtXYcorParaColl)[*ikey];

    if (mCreateTextFile)
    {
	std::string outFileName(mGlobalTag+"_Shift");
	std::string shiftType("MC");

        if(MEtXYcorParaColl->isShiftMC(*ikey) )
        {
	  shiftType = "MC";
        }else if(MEtXYcorParaColl->isShiftDY(*ikey) )
        {
	  shiftType = "DY";
        }else if(MEtXYcorParaColl->isShiftTTJets(*ikey) )
        {
	  shiftType = "TTJets";
        }else if(MEtXYcorParaColl->isShiftWJets(*ikey) )
        {
	  shiftType = "WTJets";
        }else if(MEtXYcorParaColl->isShiftData(*ikey) )
        {
	  shiftType = "Data";
	}else{
	  throw cms::Exception("InvalidKey") <<
	    "************** Can't interpret the stored key: "<<*ikey<<std::endl;
	}
	outFileName += shiftType + "_" + mPayloadName + ".txt";
	edm::LogInfo("METCorrectorDBReader")<<"outFileName: "<<outFileName;
        MEtXYcorParams.printFile(outFileName, sectionName);
    }
    
    if (mPrintScreen)
    {
      edm::LogInfo("METCorrectorDBReader")<<"Level: "<<MEtXYcorParaColl->levelName(*ikey);
      MEtXYcorParams.printScreen(sectionName);
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
