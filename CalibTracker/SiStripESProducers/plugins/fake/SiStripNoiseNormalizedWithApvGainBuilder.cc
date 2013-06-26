#include "CalibTracker/SiStripESProducers/plugins/fake/SiStripNoiseNormalizedWithApvGainBuilder.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>

SiStripNoiseNormalizedWithApvGainBuilder::SiStripNoiseNormalizedWithApvGainBuilder( const edm::ParameterSet& iConfig ):
  fp_(iConfig.getUntrackedParameter<edm::FileInPath>("file",edm::FileInPath("CalibTracker/SiStripCommon/data/SiStripDetInfo.dat"))),
  printdebug_(iConfig.getUntrackedParameter<uint32_t>("printDebug",1)),
  pset_(iConfig),
  electronsPerADC_(0.),
  minimumPosValue_(0.),
  stripLengthMode_(true),
  printDebug_(0)
{}

void SiStripNoiseNormalizedWithApvGainBuilder::analyze(const edm::Event& evt, const edm::EventSetup& iSetup)
{
  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  iSetup.get<IdealGeometryRecord>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();
 
  // Read the gain from the given tag
  edm::ESHandle<SiStripApvGain> inputApvGain;
  iSetup.get<SiStripApvGainRcd>().get( inputApvGain );
  std::vector<uint32_t> inputDetIds;
  inputApvGain->getDetIds(inputDetIds);

  // Prepare the new object
  SiStripNoises* obj = new SiStripNoises();

  SiStripDetInfoFileReader reader(fp_.fullPath());



  stripLengthMode_ = pset_.getParameter<bool>("StripLengthMode");
  
  //parameters for random noise generation. not used if Strip length mode is chosen
  std::map<int, std::vector<double> > meanNoise;
  fillParameters(meanNoise, "MeanNoise");
  std::map<int, std::vector<double> > sigmaNoise;
  fillParameters(sigmaNoise, "SigmaNoise");
  minimumPosValue_ = pset_.getParameter<double>("MinPositiveNoise");

  //parameters for strip length proportional noise generation. not used if random mode is chosen
  std::map<int, std::vector<double> > noiseStripLengthLinearSlope;
  fillParameters(noiseStripLengthLinearSlope, "NoiseStripLengthSlope");
  std::map<int, std::vector<double> > noiseStripLengthLinearQuote;
  fillParameters(noiseStripLengthLinearQuote, "NoiseStripLengthQuote");
  electronsPerADC_ = pset_.getParameter<double>("electronPerAdc");        

  printDebug_ = pset_.getUntrackedParameter<uint32_t>("printDebug", 5);

  unsigned int count = 0;
  const std::map<uint32_t, SiStripDetInfoFileReader::DetInfo > DetInfos = reader.getAllData();
  for(std::map<uint32_t, SiStripDetInfoFileReader::DetInfo >::const_iterator it = DetInfos.begin(); it != DetInfos.end(); it++) {

    // Find if this DetId is in the input tag and if so how many are the Apvs for which it contains information
    SiStripApvGain::Range inputRange(inputApvGain->getRange(it->first));

    //Generate Noises for det detid
    SiStripNoises::InputVector theSiStripVector;
    float noise = 0.;
    uint32_t detId = it->first;
    std::pair<int, int> sl = subDetAndLayer(detId, tTopo);
    unsigned short nApvs = it->second.nApvs;

    if(stripLengthMode_) {
      // Use strip length
      double linearSlope = noiseStripLengthLinearSlope[sl.first][sl.second];
      double linearQuote = noiseStripLengthLinearQuote[sl.first][sl.second];
      double stripLength = it->second.stripLength;
      for( unsigned short j=0; j<nApvs; ++j ) {

        double gain = inputApvGain->getApvGain(j, inputRange);

        for( unsigned short stripId = 0; stripId < 128; ++stripId ) {
          noise = ( ( linearSlope*stripLength + linearQuote) / electronsPerADC_ ) * gain;
          if( count<printDebug_ ) printLog(detId, stripId+128*j, noise);
          obj->setData(noise, theSiStripVector);
        }
      }
    }
    else {
      // Use random generator
      double meanN = meanNoise[sl.first][sl.second];
      double sigmaN = sigmaNoise[sl.first][sl.second];
      for( unsigned short j=0; j<nApvs; ++j ) {

        double gain = inputApvGain->getApvGain(j, inputRange);

        for( unsigned short stripId = 0; stripId < 128; ++stripId ) {
          noise = ( CLHEP::RandGauss::shoot(meanN, sigmaN) ) * gain;
          if( noise<=minimumPosValue_ ) noise = minimumPosValue_;
          if( count<printDebug_ ) printLog(detId, stripId+128*j, noise);
          obj->setData(noise, theSiStripVector);
        }
      }
    }
    ++count;
    
    if ( ! obj->put(it->first,theSiStripVector) ) {
      edm::LogError("SiStripNoisesFakeESSource::produce ")<<" detid already exists"<<std::endl;
    }
  }
  
  //End now write data in DB
  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  
  if( mydbservice.isAvailable() ){
    if( mydbservice->isNewTagRequest("SiStripNoisesRcd") ){
      mydbservice->createNewIOV<SiStripNoises>(obj,mydbservice->beginOfTime(),mydbservice->endOfTime(),"SiStripNoisesRcd");
    }
    else {
      mydbservice->appendSinceTime<SiStripNoises>(obj,mydbservice->currentTime(),"SiStripNoisesRcd");
    }
  }
  else {
    edm::LogError("SiStripNoiseNormalizedWithApvGainBuilder")<<"Service is unavailable"<<std::endl;
  }
}

std::pair<int, int> SiStripNoiseNormalizedWithApvGainBuilder::subDetAndLayer(const uint32_t detId, const TrackerTopology* tTopo) const
{
  int layerId = 0;

  StripSubdetector subid(detId);
  int subId = subid.subdetId();

  if( subId == int(StripSubdetector::TIB)) {
    layerId = tTopo->tibLayer(detId) - 1;
  }
  else if(subId == int(StripSubdetector::TOB)) {
    layerId = tTopo->tobLayer(detId) - 1;
  }
  else if(subId == int(StripSubdetector::TID)) {
    layerId = tTopo->tidRing(detId) - 1;
  }
  if(subId == int(StripSubdetector::TEC)) {
    layerId = tTopo->tecRing(detId) - 1;
  }
  return std::make_pair(subId, layerId);
}

void SiStripNoiseNormalizedWithApvGainBuilder::fillParameters(std::map<int, std::vector<double> > & mapToFill, const std::string & parameterName) const
{
  int layersTIB = 4;
  int ringsTID = 3;
  int layersTOB = 6;
  int ringsTEC = 7;

  fillSubDetParameter( mapToFill, pset_.getParameter<std::vector<double> >(parameterName+"TIB"), int(StripSubdetector::TIB), layersTIB );
  fillSubDetParameter( mapToFill, pset_.getParameter<std::vector<double> >(parameterName+"TID"), int(StripSubdetector::TID), ringsTID );
  fillSubDetParameter( mapToFill, pset_.getParameter<std::vector<double> >(parameterName+"TOB"), int(StripSubdetector::TOB), layersTOB );
  fillSubDetParameter( mapToFill, pset_.getParameter<std::vector<double> >(parameterName+"TEC"), int(StripSubdetector::TEC), ringsTEC );
}

void SiStripNoiseNormalizedWithApvGainBuilder::fillSubDetParameter(std::map<int, std::vector<double> > & mapToFill, const std::vector<double> & v, const int subDet, const unsigned short layers) const
{
  if( v.size() == layers ) {
    mapToFill.insert(std::make_pair( subDet, v ));
  }
  else if( v.size() == 1 ) {
    std::vector<double> parV(layers, v[0]);
    mapToFill.insert(std::make_pair( subDet, parV ));
  }
  else {
    throw cms::Exception("Configuration") << "ERROR: number of parameters for subDet " << subDet << " are " << v.size() << ". They must be either 1 or " << layers << std::endl;
  }
}
