#include "CalibTracker/SiStripESProducers/interface/SiStripNoisesGenerator.h"
#include <boost/cstdint.hpp>
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"

#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandGauss.h"

SiStripNoisesGenerator::SiStripNoisesGenerator(const edm::ParameterSet& iConfig,const edm::ActivityRegistry& aReg):
  SiStripCondObjBuilderBase<SiStripNoises>::SiStripCondObjBuilderBase(iConfig),
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

void SiStripNoisesGenerator::createObject()
{    
  obj_ = new SiStripNoises();

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
    std::pair<int, int> sl = subDetAndLayer(detId);
    unsigned short nApvs = it->second.nApvs;


    if(stripLengthMode_) {
      // Use strip length
      double linearSlope = noiseStripLengthLinearSlope[sl.first][sl.second];
      double linearQuote = noiseStripLengthLinearQuote[sl.first][sl.second];
      double stripLength = it->second.stripLength;
      for( unsigned short j=0; j<128*nApvs; ++j ) {
	noise = ( linearSlope*stripLength + linearQuote) / electronsPerADC_;
        if( count<printDebug_ ) printLog(detId, j, noise);
        obj_->setData(noise, theSiStripVector);
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
        obj_->setData(noise, theSiStripVector);
      }
    }
    ++count;
    
    if ( ! obj_->put(it->first,theSiStripVector) ) {
      edm::LogError("SiStripNoisesFakeESSource::produce ")<<" detid already exists"<<std::endl;
    }
  }
}

std::pair<int, int> SiStripNoisesGenerator::subDetAndLayer( const uint32_t detId ) const
{
  int layerId = 0;

  StripSubdetector subid(detId);
  int subId = subid.subdetId();

  if( subId == int(StripSubdetector::TIB)) {
    TIBDetId theTIBDetId(detId);
    layerId = theTIBDetId.layer() - 1;
  }
  else if(subId == int(StripSubdetector::TOB)) {
    TOBDetId theTOBDetId(detId);
    layerId = theTOBDetId.layer() - 1;
  }
  else if(subId == int(StripSubdetector::TID)) {
    TIDDetId theTIDDetId(detId);
    layerId = theTIDDetId.ring() - 1;
  }
  if(subId == int(StripSubdetector::TEC)) {
    TECDetId theTECDetId = TECDetId(detId); 
    layerId = theTECDetId.ring() - 1;
  }
  return std::make_pair(subId, layerId);
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
