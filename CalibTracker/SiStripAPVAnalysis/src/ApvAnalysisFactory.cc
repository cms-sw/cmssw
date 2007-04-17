#include "CalibTracker/SiStripAPVAnalysis/interface/ApvAnalysisFactory.h"
//#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"

using namespace std;
ApvAnalysisFactory::ApvAnalysisFactory(string theAlgorithmType, int theNumCMstripsInGroup, int theMaskCalcFlag, float theMaskNoiseCut,
		     float theMaskDeadCut,
		     float theMaskTruncCut,
		     float theCutToAvoidSignal,
		     int  theEventInitNumber,
		     int theEventIterNumber){

  theAlgorithmType_ =  theAlgorithmType;
  theNumCMstripsInGroup_ = theNumCMstripsInGroup;
  theMaskCalcFlag_ =  theMaskCalcFlag;
  theMaskNoiseCut_ =  theMaskNoiseCut;
  theMaskDeadCut_ =   theMaskDeadCut;
  theMaskTruncCut_ =  theMaskTruncCut;
  theCutToAvoidSignal_ =   theCutToAvoidSignal;
  theEventInitNumber_ =   theEventInitNumber;
  theEventIterNumber_ =  theEventIterNumber;

}

ApvAnalysisFactory::ApvAnalysisFactory(const edm::ParameterSet& pset){

  
  theAlgorithmType_ = pset.getParameter<string>("CalculatorAlgorithm");
  theNumCMstripsInGroup_ = pset.getParameter<int>("NumCMstripsInGroup");
  theMaskCalcFlag_ = pset.getParameter<int>("MaskCalculationFlag");
  
      theMaskNoiseCut_ = pset.getParameter<double>("MaskNoiseCut");
  theMaskDeadCut_ =  pset.getParameter<double>("MaskDeadCut");
  theMaskTruncCut_  = pset.getParameter<double>("MaskTruncationCut");
  theCutToAvoidSignal_ = pset.getParameter<double>("CutToAvoidSignal");
  
  theEventInitNumber_ =  pset.getParameter<int>("NumberOfEventsForInit");
  theEventIterNumber_ = pset.getParameter<int>("NumberOfEventsForIteration");
  apvMap_.clear();

 
}
//----------------------------------
ApvAnalysisFactory::~ApvAnalysisFactory(){
  ApvAnalysisFactory::ApvAnalysisMap::iterator it = apvMap_.begin();
  for(;it!=apvMap_.end();it++)
    {
      vector<ApvAnalysis*>::iterator myApv = (*it).second.begin();
      for(;myApv!=(*it).second.end();myApv++)
	deleteApv(*myApv);
    }
  apvMap_.clear();
}

//----------------------------------

bool ApvAnalysisFactory::instantiateApvs(uint32_t detId, int numberOfApvs){

  ApvAnalysisFactory::ApvAnalysisMap::iterator CPos = apvMap_.find(detId);
  if(CPos != apvMap_.end()) { 
    cout << " APVs for Detector Id " << detId << " already created !!!" << endl;;
    return false;
  }
  vector< ApvAnalysis* > temp;
  for(int i=0;i<numberOfApvs;i++)
    {
      ApvAnalysis* apvTmp = new ApvAnalysis(theEventIterNumber_);
      constructAuxiliaryApvClasses(apvTmp); 
      temp.push_back(apvTmp);
    }
  apvMap_.insert(pair< uint32_t, vector< ApvAnalysis* > >(detId, temp));
   return true;  
}

void ApvAnalysisFactory::constructAuxiliaryApvClasses (ApvAnalysis* theAPV)
{
  //----------------------------------------------------------------
  // Create the ped/noise/CMN calculators, zero suppressors etc.
  // (Is called by addDetUnitAndConstructApvs()).
  //
  // N.B. Don't call this twice for the same APV!
  //-----------------------------------------------------------------
  // cout<<"VirtualApvAnalysisFactory::constructAuxiliaryApvClasses"<<endl;
  TkPedestalCalculator*   thePedestal =0;
  TkNoiseCalculator*      theNoise=0;
  TkApvMask*              theMask=0;
  TkCommonModeCalculator* theCM=0;

  TkCommonMode*   theCommonMode = new TkCommonMode();
  TkCommonModeTopology* theTopology = new TkCommonModeTopology(128, theNumCMstripsInGroup_);
  theCommonMode->setTopology(theTopology);

  // Create desired algorithms.
  if (theAlgorithmType_ == "TT6")
    {
      theMask = new TT6ApvMask(theMaskCalcFlag_,theMaskNoiseCut_,theMaskDeadCut_,theMaskTruncCut_); 
      theNoise = new TT6NoiseCalculator(theEventInitNumber_, theEventIterNumber_, theCutToAvoidSignal_); 
      thePedestal = new TT6PedestalCalculator(theEventInitNumber_, theEventIterNumber_, theCutToAvoidSignal_);
      theCM = new TT6CommonModeCalculator (theNoise, theMask, theCutToAvoidSignal_);
    }
  if(theCommonMode)
    theCM->setCM(theCommonMode);
  if(thePedestal)
    theAPV->setPedestalCalculator(*thePedestal);
  if(theNoise)
    theAPV->setNoiseCalculator(*theNoise);
  if(theMask)
    theAPV->setMask(*theMask);
  if(theCM)
    theAPV->setCommonModeCalculator(*theCM);





}

void ApvAnalysisFactory::update(uint32_t detId, const edm::DetSet<SiStripRawDigi>& in)
{
  map<uint32_t, vector<ApvAnalysis*> >::const_iterator apvAnalysisIt = apvMap_.find(detId);
  if(apvAnalysisIt != apvMap_.end())
    {
      int i=0;
       for(vector<ApvAnalysis*>::const_iterator apvIt = (apvAnalysisIt->second).begin(); apvIt != (apvAnalysisIt->second).end(); apvIt++)
	 {
	   edm::DetSet<SiStripRawDigi> tmpRawDigi;
	   //it is missing the detId ...
	   tmpRawDigi.data.reserve(128);
	   int startStrip = 128*i;
	   int stopStrip = startStrip + 128;
	   
	   for(int istrip = startStrip; istrip < stopStrip;istrip++)
	     {
               if (in.data.size() <= istrip) tmpRawDigi.data.push_back(0);
	       else tmpRawDigi.data.push_back(in.data[istrip]); //maybe dangerous
	     }

	   (*apvIt)->newEvent();
	   (*apvIt)->updateCalibration(tmpRawDigi);
	   i++;
	 }
    }
  
}



void ApvAnalysisFactory::getPedestal(uint32_t detId, int apvNumber, ApvAnalysis::PedestalType& peds)
{
  //Get the pedestal for a given apv
  peds.clear();
  map<uint32_t, vector<ApvAnalysis*> >::const_iterator apvAnalysisIt = apvMap_.find(detId);
  if(apvAnalysisIt != apvMap_.end())
    {
      vector<ApvAnalysis*> myApvs = apvAnalysisIt->second;
      peds = myApvs[apvNumber]->pedestalCalculator().pedestal();
    }  
}

void ApvAnalysisFactory::getPedestal(uint32_t detId, ApvAnalysis::PedestalType& peds)
{
  //Get the pedestal for a given apv
  peds.clear();
  map<uint32_t, vector<ApvAnalysis*> >::const_iterator apvAnalysisIt = apvMap_.find(detId);
  if(apvAnalysisIt != apvMap_.end())
    {
      vector<ApvAnalysis* > theApvs = apvAnalysisIt->second;
      for(vector<ApvAnalysis*>::const_iterator it = theApvs.begin(); it != theApvs.end();it++)
	{
	  ApvAnalysis::PedestalType tmp = (*it)->pedestalCalculator().pedestal();
	  for(ApvAnalysis::PedestalType::const_iterator pit =tmp.begin(); pit!=tmp.end(); pit++) 
	    peds.push_back(*pit);
	}
    }
}
float ApvAnalysisFactory::getStripPedestal(uint32_t detId, int stripNumber)
{
  //Get the pedestal for a given apv
  ApvAnalysis::PedestalType temp;
  int apvNumb = int(stripNumber / 128.); 
  int stripN = (stripNumber - apvNumb*128);
  
  getPedestal(detId, apvNumb, temp);
  return temp[stripN];

}
 void ApvAnalysisFactory::getNoise(uint32_t detId, int apvNumber, ApvAnalysis::PedestalType& noise)
{
  //Get the pedestal for a given apv
  noise.clear();
  map<uint32_t, vector<ApvAnalysis*> >::const_iterator apvAnalysisIt = apvMap_.find(detId);
  if(apvAnalysisIt != apvMap_.end())
    {
      vector<ApvAnalysis* > theApvs = apvAnalysisIt->second;

      noise = theApvs[apvNumber]->noiseCalculator().noise();
    }
}

float ApvAnalysisFactory::getStripNoise(uint32_t detId, int stripNumber)
{
  //Get the pedestal for a given apv
  ApvAnalysis::PedestalType temp;
  int apvNumb = int(stripNumber / 128.); 
  int stripN = (stripNumber - apvNumb*128);
  
  getNoise(detId, apvNumb, temp);
  return temp[stripN];

}

void ApvAnalysisFactory::getNoise(uint32_t detId, ApvAnalysis::PedestalType& peds)
{
  //Get the pedestal for a given apv
  peds.clear();
  map<uint32_t, vector<ApvAnalysis* > >::const_iterator theApvs_map =  apvMap_.find(detId);
  if(theApvs_map != apvMap_.end())
    {
      vector<ApvAnalysis*>::const_iterator theApvs = (theApvs_map->second).begin();
      for(; theApvs !=  (theApvs_map->second).end();theApvs++)
	{
	  ApvAnalysis::PedestalType tmp = (*theApvs)->noiseCalculator().noise();
	  for(ApvAnalysis::PedestalType::const_iterator pit =tmp.begin(); pit!=tmp.end(); pit++) 
	    peds.push_back(*pit);
	}
    }
}


 void ApvAnalysisFactory::getRawNoise(uint32_t detId, int apvNumber, ApvAnalysis::PedestalType& noise)
{
  //Get the pedestal for a given apv
  noise.clear();
  map<uint32_t, vector<ApvAnalysis*> >::const_iterator apvAnalysisIt = apvMap_.find(detId);
  if(apvAnalysisIt != apvMap_.end())
    {
      vector<ApvAnalysis* > theApvs = apvAnalysisIt->second;

      noise = theApvs[apvNumber]->pedestalCalculator().rawNoise();
    }
}

float ApvAnalysisFactory::getStripRawNoise(uint32_t detId, int stripNumber)
{
  //Get the pedestal for a given apv
  ApvAnalysis::PedestalType temp;
  int apvNumb = int(stripNumber / 128.); 
  int stripN = (stripNumber - apvNumb*128);
  
  getRawNoise(detId, apvNumb, temp);
  return temp[stripN];

}

void ApvAnalysisFactory::getRawNoise(uint32_t detId, ApvAnalysis::PedestalType& peds)
{
  //Get the pedestal for a given apv
  peds.clear();
  map<uint32_t, vector<ApvAnalysis* > >::const_iterator theApvs_map =  apvMap_.find(detId);
  if(theApvs_map != apvMap_.end())
    {
      vector<ApvAnalysis*>::const_iterator theApvs = (theApvs_map->second).begin();
      for(; theApvs !=  (theApvs_map->second).end();theApvs++)
	{
	  ApvAnalysis::PedestalType tmp = (*theApvs)->pedestalCalculator().rawNoise();
	  for(ApvAnalysis::PedestalType::const_iterator pit =tmp.begin(); pit!=tmp.end(); pit++) 
	    peds.push_back(*pit);
	}
    }
}



vector<float> ApvAnalysisFactory::getCommonMode(uint32_t detId, int apvNumber)
{
  vector<float> tmp;
  tmp.clear();
  map<uint32_t, vector<ApvAnalysis* > >::const_iterator theApvs_map =  apvMap_.find(detId);
  if(theApvs_map != apvMap_.end())
    {
      vector<ApvAnalysis* > theApvs = theApvs_map->second;
      
      tmp = theApvs[apvNumber]->commonModeCalculator().commonMode()->returnAsVector();
    }
  return tmp;
}
void ApvAnalysisFactory::getCommonMode(uint32_t detId,ApvAnalysis::PedestalType& tmp)
{

  map<uint32_t, vector<ApvAnalysis*> >::const_iterator apvAnalysisIt = apvMap_.find(detId);
  if(apvAnalysisIt != apvMap_.end())
    {
 
      vector<ApvAnalysis* > theApvs = apvAnalysisIt->second;
      for(int i=0; i< theApvs.size(); i++)
	{
	  //To be fixed. We return only the first one in the vector.
	  vector<float> tmp_cm = theApvs[i]->commonModeCalculator().commonMode()->returnAsVector();
	  for(int it = 0; it<tmp_cm.size(); it++)
	    tmp.push_back(tmp_cm[it]);
	}
    }
}

void ApvAnalysisFactory::getMask(uint32_t det_id, TkApvMask::MaskType& tmp)
{
  
  map<uint32_t, vector<ApvAnalysis*> >::const_iterator apvAnalysisIt = apvMap_.find(det_id);
  if(apvAnalysisIt != apvMap_.end())
    {
      
      vector<ApvAnalysis* > theApvs = apvAnalysisIt->second;
      for(int i=0; i< theApvs.size(); i++)
	{
	  TkApvMask::MaskType theMaskType = (theApvs[i]->mask()).mask();
	  //cout <<"theMaskType size "<<theMaskType.size()<<endl;
	      
	  for(int ii=0;ii<theMaskType.size();ii++)
	    {
	      tmp.push_back(theMaskType[ii]);
	      //cout <<"The Value "<<theMaskType[ii]<<" "<<ii<<endl;
	    }
	}
    }
}
bool ApvAnalysisFactory::isUpdating(uint32_t detId)
{
  bool updating = true;
  map<uint32_t, vector<ApvAnalysis*> >::const_iterator apvAnalysisIt = apvMap_.find(detId);
  if(apvAnalysisIt != apvMap_.end())
    {
      for(vector<ApvAnalysis*>::const_iterator apvIt = (apvAnalysisIt->second).begin(); apvIt != (apvAnalysisIt->second).end(); apvIt++)  
	{	
	  if(!( (*apvIt)->pedestalCalculator().status()->isUpdating() ))
	    updating = false;
	}
    }
  return updating;

}

void ApvAnalysisFactory::deleteApv(ApvAnalysis* apv){
  delete &(apv->pedestalCalculator());
  delete &(apv->noiseCalculator());
  delete &(apv->mask());
  delete &(apv->commonModeCalculator().commonMode()->topology());
  delete (apv->commonModeCalculator().commonMode());
  delete &(apv->commonModeCalculator());
  delete apv;

}
