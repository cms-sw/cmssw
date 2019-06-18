#include "L1Trigger/RPCTriggerPrimitives/interface/PrimitivePreprocess.h"

PrimitivePreprocess::PrimitivePreprocess(const edm::ParameterSet& iConfig, edm::ConsumesCollector&& iConsumes):
  rpcToken_(iConsumes.consumes<RPCDigiCollection>(iConfig.getParameter<edm::InputTag>("Primitiverechitlabel"))),
  processorvector_(),
  Mapsource_(iConfig.getParameter<std::string>("Mapsource")),
  ApplyLinkBoardCut_(iConfig.getParameter<bool>("ApplyLinkBoardCut")),
  LinkBoardCut_(iConfig.getParameter<int>("LinkBoardCut")),
  ClusterSizeCut_(iConfig.getParameter<int>("ClusterSizeCut")),
  theRPCMaskedStripsObj(nullptr),
  theRPCDeadStripsObj(nullptr),
  // Get the concrete reconstruction algo from the factory
  theAlgorithm{PrimitiveAlgoFactory::get()->create(iConfig.getParameter<std::string>("recAlgo"),
							 iConfig.getParameter<edm::ParameterSet>("recAlgoConfig"))},
  maskSource_(MaskSource::EventSetup), 
  deadSource_(MaskSource::EventSetup){ 
  
  
  //Get LUT for linkboard map
  std::ifstream inputFile(Mapsource_.fullPath().c_str(), std::ios::in);
  
  if(!inputFile){
    throw cms::Exception("No LUT file") << "Error: Linkboard mapping cannot be opened";
    exit(1);
  }
  
  while( inputFile.good() ){
    RPCProcessor::Map_structure temp; 
    inputFile >> temp.linkboard_ >> temp.linkboard_ID >> temp.chamber1_ >> temp.chamber2_;
    Final_MapVector.push_back(temp);  
  }
  Final_MapVector.pop_back();
  
  inputFile.close();
  
  
  
  const std::string maskSource = iConfig.getParameter<std::string>("maskSource");
  
  if (maskSource == "File") {
    maskSource_ = MaskSource::File;
    edm::FileInPath fp1 = iConfig.getParameter<edm::FileInPath>("maskvecfile");
    std::ifstream inputFile_1(fp1.fullPath().c_str(), std::ios::in);
    if ( !inputFile_1 ) {
      std::cerr << "Masked Strips File cannot not be opened" << std::endl;
      exit(1);
    }
    while ( inputFile_1.good() ) {
      RPCMaskedStrips::MaskItem Item;
      inputFile_1 >> Item.rawId >> Item.strip;
      if ( inputFile_1.good() ) MaskVec.push_back(Item);
    }
    inputFile_1.close();
  }
  
  const std::string deadSource = iConfig.getParameter<std::string>("deadSource");
  
  if (deadSource == "File") {
    deadSource_ = MaskSource::File;
    edm::FileInPath fp2 = iConfig.getParameter<edm::FileInPath>("deadvecfile");
    std::ifstream inputFile_2(fp2.fullPath().c_str(), std::ios::in);
    if ( !inputFile_2 ) {
      std::cerr << "Dead Strips File cannot not be opened" << std::endl;
      exit(1);
    }
    while ( inputFile_2.good() ) {
      RPCDeadStrips::DeadItem Item;
      inputFile_2 >> Item.rawId >> Item.strip; 
      if ( inputFile_2.good() ) DeadVec.push_back(Item);
    }
    inputFile_2.close();
    
  } 
  
  //Closing the input files
  }

PrimitivePreprocess::~PrimitivePreprocess(){
}

void PrimitivePreprocess::beginRun(const edm::EventSetup& iSetup){
  
  // Get masked- and dead-strip information
  theRPCMaskedStripsObj = std::make_unique<RPCMaskedStrips>();
  theRPCDeadStripsObj = std::make_unique<RPCDeadStrips>();
  
  // Getting the masked-strip information
  if ( maskSource_ == MaskSource::EventSetup ) {
    edm::ESHandle<RPCMaskedStrips> readoutMaskedStrips;
    iSetup.get<RPCMaskedStripsRcd>().get(readoutMaskedStrips);
    const RPCMaskedStrips* tmp_obj = readoutMaskedStrips.product();
    theRPCMaskedStripsObj->MaskVec = tmp_obj->MaskVec;
    delete tmp_obj;
  }  
  
  
  else if ( maskSource_ == MaskSource::File ) {
    std::vector<RPCMaskedStrips::MaskItem>::iterator posVec;
    for ( posVec = MaskVec.begin(); posVec != MaskVec.end(); ++posVec ) {
      RPCMaskedStrips::MaskItem Item; 
      Item.rawId = (*posVec).rawId;
      Item.strip = (*posVec).strip; 
      theRPCMaskedStripsObj->MaskVec.push_back(Item);
    }
  }
  
  // Getting the dead-strip information
  
  if ( deadSource_ == MaskSource::EventSetup ) {
    edm::ESHandle<RPCDeadStrips> readoutDeadStrips;
    iSetup.get<RPCDeadStripsRcd>().get(readoutDeadStrips);
    const RPCDeadStrips* tmp_obj = readoutDeadStrips.product();
    theRPCDeadStripsObj->DeadVec = tmp_obj->DeadVec;
    delete tmp_obj;
  }
  else if ( deadSource_ == MaskSource::File ) {
    std::vector<RPCDeadStrips::DeadItem>::iterator posVec;
    for ( posVec = DeadVec.begin(); posVec != DeadVec.end(); ++posVec ) {
      RPCDeadStrips::DeadItem Item;
      Item.rawId = (*posVec).rawId;
      Item.strip = (*posVec).strip;
      theRPCDeadStripsObj->DeadVec.push_back(Item);
    }
  }
  
} 

void PrimitivePreprocess::Preprocess(const edm::Event& iEvent, const edm::EventSetup& iSetup, RPCRecHitCollection& primitivedigi){
  //loop over rpcdigis and cluster algorithm
  
  
  std::map<std::string, std::string> LBName_ChamberID_Map_1;
  std::map<std::string, std::string> LBID_ChamberID_Map_1;
  std::map<std::string, std::string> LBName_ChamberID_Map_2;
  std::map<std::string, std::string> LBID_ChamberID_Map_2;
  
  std::vector<RPCProcessor::Map_structure>::iterator it;
  for(it = Final_MapVector.begin(); it != Final_MapVector.end(); it++){
    LBName_ChamberID_Map_1[(*it).chamber1_]=(*it).linkboard_;
    LBID_ChamberID_Map_1[(*it).chamber1_]=(*it).linkboard_ID;
    if((*it).chamber2_ != "-"){
      LBName_ChamberID_Map_2[(*it).chamber2_]=(*it).linkboard_;
      LBID_ChamberID_Map_2[(*it).chamber2_]=(*it).linkboard_ID;
    }
  }
  // map_2 is only necessary for barrel
  
  
  
  for(auto& iterator_ : processorvector_){
    iterator_.Process(iEvent, iSetup, rpcToken_, primitivedigi, 
		      theRPCMaskedStripsObj, theRPCDeadStripsObj, theAlgorithm, 
		      LBName_ChamberID_Map_1, LBID_ChamberID_Map_1, LBName_ChamberID_Map_2, LBID_ChamberID_Map_2, 
		      ApplyLinkBoardCut_, LinkBoardCut_, ClusterSizeCut_);
  }
}


