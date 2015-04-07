#include "CalibTracker/SiStripESProducers/interface/SiStripNoisesGenerator.h"
#include <boost/cstdint.hpp>
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandGauss.h"

SiStripNoisesGenerator::SiStripNoisesGenerator(const edm::ParameterSet& iConfig,const edm::ActivityRegistry& aReg):
  SiStripDepCondObjBuilderBase<SiStripNoises,TrackerTopology>::SiStripDepCondObjBuilderBase(iConfig),
  electronsPerADC_(0.),
   minimumPosValue_(0.),
   stripLengthMode_(true),
   printDebug_(0)
{
  edm::LogInfo("SiStripNoisesGenerator") <<  "[SiStripNoisesGenerator::SiStripNoisesGenerator]";
}

SiStripNoisesGenerator::~SiStripNoisesGenerator()
{ 
  edm::LogInfo("SiStripNoisesGenerator") <<  "[SiStripNoisesGenerator::~SiStripNoisesGenerator]";
}

SiStripNoises* SiStripNoisesGenerator::createObject(const TrackerTopology* tTopo)
{    
  SiStripNoises* obj = new SiStripNoises();

  stripLengthMode_ = _pset.getParameter<bool>("StripLengthMode");
  
  //parameters for random noise generation. not used if Strip length mode is chosen
  std::map<int, std::vector<double> > meanNoise;
  fillParameters(meanNoise, "MeanNoise");
  std::map<int, std::vector<double> > sigmaNoise;
  fillParameters(sigmaNoise, "SigmaNoise");
  minimumPosValue_ = _pset.getParameter<double>("MinPositiveNoise");

  //parameters for strip length proportional noise generation. not used if random mode is chosen
  std::map<int, std::vector<double> > noiseStripLengthLinearSlope;
  fillParameters(noiseStripLengthLinearSlope, "NoiseStripLengthSlope");
  std::map<int, std::vector<double> > noiseStripLengthLinearQuote;
  fillParameters(noiseStripLengthLinearQuote, "NoiseStripLengthQuote");
  electronsPerADC_ = _pset.getParameter<double>("electronPerAdc");        

  printDebug_ = _pset.getUntrackedParameter<uint32_t>("printDebug", 5);

  uint32_t count = 0;
  edm::FileInPath fp_ = _pset.getParameter<edm::FileInPath>("file");
  SiStripDetInfoFileReader reader(fp_.fullPath());
  const std::map<uint32_t, SiStripDetInfoFileReader::DetInfo > DetInfos  = reader.getAllData();
  
  for(std::map<uint32_t, SiStripDetInfoFileReader::DetInfo >::const_iterator it = DetInfos.begin(); it != DetInfos.end(); ++it) {

    //Generate Noises for det detid
    SiStripNoises::InputVector theSiStripVector;
    float noise = 0.;
    uint32_t detId = it->first;
    std::pair<int, int> sl = subDetAndLayer(detId,tTopo);
    unsigned short nApvs = it->second.nApvs;


    if(stripLengthMode_) {
      // Use strip length
      double linearSlope = noiseStripLengthLinearSlope[sl.first][sl.second];
      double linearQuote = noiseStripLengthLinearQuote[sl.first][sl.second];
      double stripLength = it->second.stripLength;
      for( unsigned short j=0; j<128*nApvs; ++j ) {
	noise = ( linearSlope*stripLength + linearQuote) / electronsPerADC_;
        if( count<printDebug_ ) printLog(detId, j, noise);
        obj->setData(noise, theSiStripVector);
      }
    }
    else {
      // Use random generator
      double meanN = meanNoise[sl.first][sl.second];
      double sigmaN = sigmaNoise[sl.first][sl.second];
      for( unsigned short j=0; j<128*nApvs; ++j ) {
        noise = CLHEP::RandGauss::shoot(meanN, sigmaN);
        if( noise<=minimumPosValue_ ) noise = minimumPosValue_;
        if( count<printDebug_ ) printLog(detId, j, noise);
        obj->setData(noise, theSiStripVector);
      }
    }
    ++count;
    
    if ( ! obj->put(it->first,theSiStripVector) ) {
      edm::LogError("SiStripNoisesFakeESSource::produce ")<<" detid already exists"<<std::endl;
    }
  }
  return obj;
}

std::pair<int, int> SiStripNoisesGenerator::subDetAndLayer( const uint32_t detId, const TrackerTopology* tTopo ) const
{
  int layerId = 0;

  const DetId detectorId=DetId(detId);
  const int subDet = detectorId.subdetId();

  if( subDet == int(StripSubdetector::TIB)) {
    layerId = tTopo->tibLayer(detectorId) - 1;
  }
  else if(subDet == int(StripSubdetector::TOB)) {
    layerId = tTopo->tobLayer(detectorId) - 1;
  }
  else if(subDet == int(StripSubdetector::TID)) {
    layerId = tTopo->tidRing(detectorId) - 1;
  }
  if(subDet == int(StripSubdetector::TEC)) {
    layerId = tTopo->tecRing(detectorId) - - 1;
  }
  return std::make_pair(subDet, layerId);
}

void SiStripNoisesGenerator::fillParameters(std::map<int, std::vector<double> > & mapToFill, const std::string & parameterName) const
{
  int layersTIB = 4;
  int ringsTID = 3;
  int layersTOB = 6;
  int ringsTEC = 7;

  fillSubDetParameter( mapToFill, _pset.getParameter<std::vector<double> >(parameterName+"TIB"), int(StripSubdetector::TIB), layersTIB );
  fillSubDetParameter( mapToFill, _pset.getParameter<std::vector<double> >(parameterName+"TID"), int(StripSubdetector::TID), ringsTID );
  fillSubDetParameter( mapToFill, _pset.getParameter<std::vector<double> >(parameterName+"TOB"), int(StripSubdetector::TOB), layersTOB );
  fillSubDetParameter( mapToFill, _pset.getParameter<std::vector<double> >(parameterName+"TEC"), int(StripSubdetector::TEC), ringsTEC );
}

void SiStripNoisesGenerator::fillSubDetParameter(std::map<int, std::vector<double> > & mapToFill, const std::vector<double> & v, const int subDet, const unsigned short layers) const
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
