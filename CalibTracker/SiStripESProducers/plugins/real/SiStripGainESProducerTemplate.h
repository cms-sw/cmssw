#ifndef CalibTracker_SiStripESProducers_SiStripGainESProducerTemplate_h
#define CalibTracker_SiStripESProducers_SiStripGainESProducerTemplate_h

// system include files
#include <memory>
#include <utility>

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"
#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"
#include "CalibTracker/Records/interface/SiStripDependentRecords.h"
//
// class declaration
//
template<typename TDependentRecord, typename TInputRecord>
class SiStripGainESProducerTemplate : public edm::ESProducer {
 public:
  SiStripGainESProducerTemplate(const edm::ParameterSet&);
  ~SiStripGainESProducerTemplate() override{};
  
  std::unique_ptr<SiStripGain> produce(const TDependentRecord&);

 private:

  SiStripGain* SiStripGainNormalizationFunction(const TDependentRecord& iRecord);
  double getNFactor(const int apvGainIndex);

  std::vector<edm::ParameterSet> apvGainLabels_;
  std::vector<std::pair<std::string, std::string> > apvgain_;
  std::vector<double> norm_;
  bool automaticMode_;
  bool  printdebug_;
  SiStripGain * gain_;
  std::vector<edm::ESHandle<SiStripApvGain> > pDD;

  void fillApvGain( const SiStripGainRcd & a, const std::pair<std::string, std::string> & recordLabelPair, std::vector<edm::ESHandle<SiStripApvGain> >& pDD );
};

template<typename TDependentRecord, typename TInputRecord>
SiStripGainESProducerTemplate<TDependentRecord,TInputRecord>::SiStripGainESProducerTemplate(const edm::ParameterSet& iConfig)
{
  setWhatProduced(this);

  automaticMode_ = iConfig.getParameter<bool>("AutomaticNormalization");
  printdebug_ = iConfig.getUntrackedParameter<bool>("printDebug", false);
  apvGainLabels_ = iConfig.getParameter<std::vector<edm::ParameterSet> >("APVGain");

  // Fill the vector of apv labels
  std::vector<edm::ParameterSet>::const_iterator gainPSetIt = apvGainLabels_.begin();
  for( ; gainPSetIt != apvGainLabels_.end(); ++gainPSetIt ) {
    apvgain_.push_back( std::make_pair(gainPSetIt->getParameter<std::string>("Record"), gainPSetIt->getUntrackedParameter<std::string>("Label", "")) );
    norm_.push_back(gainPSetIt->getUntrackedParameter<double>("NormalizationFactor", 1.));
  }
  bool badNorm = false;
  std::vector<double>::const_iterator it = norm_.begin();
  for( ; it != norm_.end(); ++it ) {
    if( *it <= 0 ) badNorm = true;
  }

  if(!automaticMode_ && badNorm ){
    edm::LogError("SiStripGainESProducer") << "[SiStripGainESProducer] - ERROR: negative or zero Normalization factor provided. Assuming 1 for such factor" << std::endl;
    norm_ = std::vector<double>(norm_.size(), 1.);
  }
}

template<typename TDependentRecord, typename TInputRecord>
std::unique_ptr<SiStripGain> SiStripGainESProducerTemplate<TDependentRecord,TInputRecord>::produce(const TDependentRecord& iRecord)
{
  std::unique_ptr<SiStripGain> ptr(SiStripGainNormalizationFunction(iRecord));
  return ptr;
}

template<typename TDependentRecord, typename TInputRecord>
void SiStripGainESProducerTemplate<TDependentRecord, TInputRecord>::fillApvGain( const SiStripGainRcd & a, const std::pair<std::string, std::string> & recordLabelPair, std::vector<edm::ESHandle<SiStripApvGain> >& pDD )
{
  // Put in an empty ApvGain and fill it
  pDD.push_back(edm::ESHandle<SiStripApvGain>());
  std::string recordName( recordLabelPair.first );
  std::string labelName( recordLabelPair.second );
  if( recordName == "SiStripApvGainRcd" ) a.getRecord<SiStripApvGainRcd>().get( labelName, pDD.back() );
  else if( recordName == "SiStripApvGain2Rcd" ) a.getRecord<SiStripApvGain2Rcd>().get( labelName, pDD.back() );
  else if( recordName == "SiStripApvGain3Rcd" ) a.getRecord<SiStripApvGain3Rcd>().get( labelName, pDD.back() );
  else edm::LogError("SiStripGainESProducer::SiStripGainNormalizationFunction") << "ERROR: unrecognized record name " << recordName << std::endl
										<< "please specify one of: SiStripApvGainRcd, SiStripApvGain2Rcd, SiStripApvGain3Rcd" << std::endl;
}

template<typename TDependentRecord, typename TInputRecord>
SiStripGain* SiStripGainESProducerTemplate<TDependentRecord,TInputRecord>::SiStripGainNormalizationFunction(const TDependentRecord& iRecord)
{
  // First clean up the pDD vector otherwise it will contain old handles referring to products no more in the es
  pDD.clear();

  if(typeid(TDependentRecord)==typeid(SiStripGainRcd) && typeid(TInputRecord)==typeid(SiStripApvGainRcd)){

    const SiStripGainRcd& a = dynamic_cast<const SiStripGainRcd&>(iRecord);

    fillApvGain( a, apvgain_[0], pDD );
    // Create a new gain object and insert the ApvGain
    SiStripGain * gain = new SiStripGain( *(pDD[0].product()), getNFactor(0), apvgain_[0] );

    if( apvgain_.size() > 1 ) {
      for( unsigned int i=1; i<apvgain_.size(); ++i ) {
        fillApvGain( a, apvgain_[i], pDD );
        // Add the new ApvGain to the gain object
        gain->multiply(*(pDD[i].product()), getNFactor(i), apvgain_[i]);
      }
    }
    return gain;

  }else if(typeid(TDependentRecord)==typeid(SiStripGainSimRcd) && typeid(TInputRecord)==typeid(SiStripApvGainSimRcd)){

    const SiStripGainSimRcd& a = dynamic_cast<const SiStripGainSimRcd&>(iRecord);

    pDD.push_back(edm::ESHandle<SiStripApvGain>());
    a.getRecord<SiStripApvGainSimRcd>().get(apvgain_[0].second, pDD[0]);
    SiStripGain * gain = new SiStripGain( *(pDD[0].product()), getNFactor(0), apvgain_[0] );

    if( apvgain_.size() > 1 ) {
      for( unsigned int i=1; i<apvgain_.size(); ++i ) {
        pDD.push_back(edm::ESHandle<SiStripApvGain>());
        a.getRecord<SiStripApvGainSimRcd>().get(apvgain_[i].second, pDD[i]);
        gain->multiply(*(pDD[i].product()), getNFactor(i), apvgain_[i]);
      }
    }
    return gain;
  }
    
  edm::LogError("SiStripGainESProducer") << "[SiStripGainNormalizationFunction] - ERROR: asking for a pair of records different from <SiStripGainRcd,SiStripApvGainRcd> and <SiStripGainSimRcd,SiStripApvGainSimRcd>" << std::endl;
  return new SiStripGain();
}

template<typename TDependentRecord, typename TInputRecord>
double  SiStripGainESProducerTemplate<TDependentRecord,TInputRecord>::getNFactor(const int apvGainIndex){
  double NFactor=0.;

  if(automaticMode_ || printdebug_ ){

    std::vector<uint32_t> DetIds;
    pDD[apvGainIndex]->getDetIds(DetIds);

    double SumOfGains=0.;
    int NGains=0;
    
    for(std::vector<uint32_t>::const_iterator detit=DetIds.begin(); detit!=DetIds.end(); detit++){
      
      SiStripApvGain::Range detRange = pDD[apvGainIndex]->getRange(*detit);
      
      int iComp=0;
       
      for(std::vector<float>::const_iterator apvit=detRange.first; apvit!=detRange.second; apvit++){
	 
	SumOfGains+=(*apvit);
	NGains++;
	if (printdebug_)
	  edm::LogInfo("SiStripGainESProducer::produce()")<< "detid/component: " << *detit <<"/"<<iComp<< "   gain factor " <<*apvit ;
	iComp++;
      }      
    }
    
    if(automaticMode_){
      if(SumOfGains>0 && NGains>0){
	NFactor=SumOfGains/NGains;
      }
      else{
	edm::LogError("SiStripGainESProducer::produce() - ERROR: empty set of gain values received. Cannot compute normalization factor. Assuming 1 for such factor") << std::endl;
	NFactor=1.;
      }
    }
  }
  
  if(!automaticMode_){
    NFactor=norm_[apvGainIndex];
  }

  if (printdebug_)  edm::LogInfo("SiStripGainESProducer")<< " putting A SiStrip Gain object in eventSetup with normalization factor " << NFactor ;
  return NFactor;
}  
#endif
