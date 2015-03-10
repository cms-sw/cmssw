#include "CalibCalorimetry/EcalTiming/plugins/EcalTimeCalibrationValidator.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//#include "CondFormats/DataRecord/interface/EcalTimeCalibConstantsRcd.h"
//#include "CondTools/Ecal/interface/EcalTimeCalibConstantsXMLTranslator.h"
//#include "CondTools/Ecal/interface/EcalCondHeader.h"

#include "CondFormats/DataRecord/interface/EcalTimeCalibConstantsRcd.h"
#include "CondFormats/DataRecord/interface/EcalTimeOffsetConstantRcd.h"
#include "CondTools/Ecal/interface/EcalTimeCalibConstantsXMLTranslator.h"
#include "CondTools/Ecal/interface/EcalTimeOffsetXMLTranslator.h"
#include "CondTools/Ecal/interface/EcalCondHeader.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include "CalibCalorimetry/EcalTiming/interface/EcalCrystalTimingCalibration.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DQM/EcalCommon/interface/Numbers.h"

#include <fstream>

EcalTimeCalibrationValidator::EcalTimeCalibrationValidator(const edm::ParameterSet& ps) :
  inputFiles_ (ps.getParameter<std::vector<std::string> >("InputFileNames")),
  outputTreeFileName_ (ps.getParameter<std::string>("OutputFileName")),
  calibConstantFileName_ (ps.getParameter<std::string>("CalibConstantXMLFileName")),
  calibOffsetFileName_ (ps.getParameter<std::string>("CalibOffsetXMLFileName")),
  disableGlobalShift_ (ps.getParameter<bool>("ZeroGlobalOffset")),
  maxEntries_ (ps.getUntrackedParameter<int>("MaxTreeEntriesToProcess",-1)),
  startingEntry_ (ps.getUntrackedParameter<int>("StartingTreeEntry",0)),
  inRuns_ (ps.getParameter<std::string>("RunIncludeExclude"))
{
  // Tree construction
  myInputTree_ = new TChain ("EcalTimeAnalysis") ;
  std::vector<std::string>::const_iterator file_itr;
  for(file_itr=inputFiles_.begin(); file_itr!=inputFiles_.end(); file_itr++)
    myInputTree_->Add( (*file_itr).c_str() );

  if(!myInputTree_)
  {
    edm::LogError("EcalTimeCalibrationValidator") << "Couldn't find tree EcalTimeAnalysis";
    produce_ = false;
    return;
  }

  myOutputTree_ = 0;
  outputTreeFile_ = TFile::Open(outputTreeFileName_.c_str(),"recreate");
  outputTreeFile_->cd();
  myOutputTree_ = new TTree("EcalTimeAnalysis","EcalTimeAnalysis");
  if(!myOutputTree_)
  {
    edm::LogError("EcalTimeCalibrationValidator") << "Couldn't make output tree";
    produce_ = false;
    return;
  }

  genIncludeExcludeVectors(inRuns_,runIncludeVector,runExcludeVector);

  produce_ = true;
}

EcalTimeCalibrationValidator::~EcalTimeCalibrationValidator()
{
}

void
EcalTimeCalibrationValidator::analyze(edm::Event const& evt, edm::EventSetup const& es)
{
  if(!produce_)
    return;

  set(es);

  // prepare output
  EcalTimeTreeContent ttreeMembersOutput;

  // Set branch addresses for input tree
  setBranchAddresses(myInputTree_,ttreeMembersInput_);
  // Set branches for output tree
  setBranches(myOutputTree_,ttreeMembersOutput);

  //es.get<EcalTimeCalibConstantsRcd>().get(itime_);
  EcalCondHeader calibFileHeader;
  EcalTimeCalibConstants calibConstants;
  int ret = EcalTimeCalibConstantsXMLTranslator::readXML(calibConstantFileName_,calibFileHeader,calibConstants);
  if(ret)
  {
    edm::LogError("EcalTimeCalibrationValidator") << "Problem reading calibration XML file.  Quitting.";
    return;
  }
  const EcalTimeCalibConstantMap itimeMap = calibConstants;

  EcalCondHeader offsetFileHeader;
  EcalTimeOffsetConstant offsetConstant;
  int retOffset = EcalTimeOffsetXMLTranslator::readXML(calibOffsetFileName_,offsetFileHeader,offsetConstant);
  if(retOffset)
  {
    edm::LogError("EcalTimeCalibrationValidator") << "Problem reading offset  XML file.  Quitting.";
    return;
  }
  if(disableGlobalShift_) {
    recalibratedOffsetEB = 0;
    recalibratedOffsetEE = 0;
  } 
  else {
    recalibratedOffsetEB = offsetConstant.getEBValue();
    recalibratedOffsetEE = offsetConstant.getEEValue();
  }
  
  // Loop over the TTree
  int nEntries = myInputTree_->GetEntries();
  edm::LogInfo("EcalTimeCalibrationValidator") << "Begin loop over TTree";
  //inputTreeFile_->cd();
  
  // Check starting entry
  if(startingEntry_ < 0 || startingEntry_ > nEntries)
  {
    edm::LogError("EcalTimeCalibrationValidator") << "Starting entry number: " << startingEntry_
      << " too large or too small. Quitting.";
    return;
  }

  for(int entry = startingEntry_; entry < nEntries; ++entry)
  {
    if(maxEntries_ >= 1 && entry > startingEntry_+maxEntries_) break;

    if(entry % 10000 == 0)
      edm::LogInfo("EcalTimeCalibrationValidator") << "Processing tree entry: " << entry;

    myInputTree_->GetEntry(entry);

    if(!includeEvent(ttreeMembersInput_.runId,runIncludeVector,runExcludeVector))
      continue;

    ttreeMembersOutput = ttreeMembersInput_;
    // loop over all crys, apply time shifts
    for(int bCluster=0; bCluster < ttreeMembersInput_.nClusters; bCluster++)
    {
      bool isEB = true;
      if(ttreeMembersInput_.xtalInBCIEta[bCluster][0] == -999999) isEB = false;
      for(int cryInBC=0; cryInBC < ttreeMembersInput_.nXtalsInCluster[bCluster]; cryInBC++)
      {
        int hashedIndex = ttreeMembersInput_.xtalInBCHashedIndex[bCluster][cryInBC];
        uint32_t rawId = 0;
        DetId det = 0;
        if(isEB)
        {
          EBDetId detid = EBDetId::unhashIndex(hashedIndex);
          if(detid==EBDetId() || !EBDetId::validHashIndex(hashedIndex)) // make sure DetId is valid
            continue;
          else
          {
            rawId = detid.rawId();
            det = detid;
          }
        }
        else
        {
          EEDetId detid = EEDetId::unhashIndex(hashedIndex);
          if(detid==EEDetId() || !EEDetId::validHashIndex(hashedIndex)) // make sure DetId is valid
            continue;
          else
          {
            rawId = detid.rawId();
            det = detid;
          }
        }
        float origTime = ttreeMembersOutput.xtalInBCTime[bCluster][cryInBC];
        // get orig time calibration coefficient
        const EcalTimeCalibConstantMap & origTimeMap = origTimeCalibConstHandle->getMap();
        EcalTimeCalibConstant itimeconstOrig = 0;
        EcalTimeCalibConstantMap::const_iterator itimeItrOrig = origTimeMap.find(det);
        if( itimeItrOrig!=origTimeMap.end() ) {
          itimeconstOrig = (*itimeItrOrig);
        } else {
          //edm::LogError("EcalTimeCalibrationValidator") << "No time calib const found for xtal "
          //  << rawId
          //  << "in database! something wrong with EcalTimeCalibConstants in the database?";
        }
        // get orig time offset
        
        if(disableGlobalShift_) {
          originalOffsetEB = 0;
          originalOffsetEE = 0;
        } 
        else {
          originalOffsetEB = origTimeOffsetConstHandle->getEBValue();
          originalOffsetEE = origTimeOffsetConstHandle->getEEValue();
        }
        // get the raw time
        if (isEB) {
          origTime-=(itimeconstOrig + originalOffsetEB);
        }
        else {
          origTime-=(itimeconstOrig + originalOffsetEE);
        }

        // get new time calibration coefficient
        EcalTimeCalibConstantMap::const_iterator itimeItr = itimeMap.find(rawId);
        EcalTimeCalibConstant itimeconst = 0;
        if( itimeItr!=itimeMap.end() ) {
          itimeconst = (*itimeItr);
        } else {
          edm::LogError("EcalTimeCalibrationValidator") << "No time calib const found for xtal "
            << rawId
            << "! something wrong with EcalTimeCalibConstants in the XML file?";
        }
        if (isEB) {
          ttreeMembersOutput.xtalInBCTime[bCluster][cryInBC]= origTime + itimeconst + recalibratedOffsetEB;
        }
        else {
          ttreeMembersOutput.xtalInBCTime[bCluster][cryInBC]= origTime + itimeconst + recalibratedOffsetEE;
        }
      }
    }
    myOutputTree_->Fill();
    if(entry % 1000 == 0)
    {
      myOutputTree_->FlushBaskets();
    }
  }
  edm::LogInfo("EcalTimeCalibrationValidator") << "Original Offset EB: "
    << originalOffsetEB << " Orig Offset EE: " 
    << originalOffsetEE << " Recal. Offset EB: " 
    << recalibratedOffsetEB << " Recal. Offset EE: " 
    << recalibratedOffsetEE;
  outputTreeFile_->cd();
  myOutputTree_->Write();
  outputTreeFile_->Close();
}

void EcalTimeCalibrationValidator::set(edm::EventSetup const& eventSetup)
{
  eventSetup.get<EcalTimeCalibConstantsRcd>().get(origTimeCalibConstHandle);
  eventSetup.get<EcalTimeOffsetConstantRcd>().get(origTimeOffsetConstHandle);
}

void EcalTimeCalibrationValidator::beginRun(edm::EventSetup const& eventSetup)
{
}

void EcalTimeCalibrationValidator::beginJob()
{
}

void EcalTimeCalibrationValidator::endJob()
{
  //outputTreeFile_->cd();
  //myOutputTree_->Write();
  //outputTreeFile_->Close();

}

bool EcalTimeCalibrationValidator::includeEvent(double eventParameter,
    std::vector<std::vector<double> > includeVector,
    std::vector<std::vector<double> > excludeVector)
{
  bool keepEvent = false;
  if(includeVector.size()==0) keepEvent = true;
  for(uint i=0; i!=includeVector.size();++i){
    if(includeVector[i].size()==1 && eventParameter==includeVector[i][0])
      keepEvent=true;
    else if(includeVector[i].size()==2 && (eventParameter>=includeVector[i][0] && eventParameter<=includeVector[i][1]))
      keepEvent=true;
  }
  if(!keepEvent) // if it's not in our include list, skip it
    return false;

  keepEvent = true;
  for(uint i=0; i!=excludeVector.size();++i){
    if(excludeVector[i].size()==1 && eventParameter==excludeVector[i][0])
      keepEvent=false;
    else if(excludeVector[i].size()==2 && (eventParameter>=excludeVector[i][0] && eventParameter<=excludeVector[i][1]))
      keepEvent=false;
  }

  return keepEvent; // if someone includes and excludes, exclusion will overrule

}

//
std::vector<std::string> EcalTimeCalibrationValidator::split(std::string msg, std::string separator)
{
  boost::char_separator<char> sep(separator.c_str());
  boost::tokenizer<boost::char_separator<char> > tok(msg, sep );
  std::vector<std::string> token ;
  for ( boost::tokenizer<boost::char_separator<char> >::const_iterator i = tok.begin(); i != tok.end(); ++i ) {
    token.push_back(std::string(*i)) ;
  }
  return token ;
}

//
void EcalTimeCalibrationValidator::genIncludeExcludeVectors(std::string optionString,
    std::vector<std::vector<double> >& includeVector,
    std::vector<std::vector<double> >& excludeVector)
{
  std::vector<std::string> rangeStringVector;
  std::vector<double> rangeIntVector;

  if(optionString != "-1"){
    std::vector<std::string> stringVector = split(optionString,",") ;

    for (uint i=0 ; i<stringVector.size() ; i++) {
      bool exclude = false;

      if(stringVector[i].at(0)=='x'){
        exclude = true;
        stringVector[i].erase(0,1);
      }
      rangeStringVector = split(stringVector[i],"-") ;

      rangeIntVector.clear();
      for(uint j=0; j<rangeStringVector.size();j++) {
        rangeIntVector.push_back(atof(rangeStringVector[j].c_str()));
      }
      if(exclude) excludeVector.push_back(rangeIntVector);
      else includeVector.push_back(rangeIntVector);

    }
  }
}
