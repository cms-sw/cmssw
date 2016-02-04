#ifndef CalibCalorimetry_EcalTrivialCondModules_ESTrivialConditionRetriever_H
#define CalibCalorimetry_EcalTrivialCondModules_ESTrivialConditionRetriever_H
// system include files
#include <string>
#include <vector>

// user include files
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"

#include "CondFormats/ESObjects/interface/ESPedestals.h"
#include "CondFormats/DataRecord/interface/ESPedestalsRcd.h"

#include "CondFormats/ESObjects/interface/ESStripGroupId.h"
#include "CondFormats/ESObjects/interface/ESWeightStripGroups.h"
#include "CondFormats/DataRecord/interface/ESWeightStripGroupsRcd.h"

#include "CondFormats/ESObjects/interface/ESWeight.h"
#include "CondFormats/ESObjects/interface/ESWeightSet.h"
#include "CondFormats/ESObjects/interface/ESTBWeights.h"
#include "CondFormats/DataRecord/interface/ESTBWeightsRcd.h"

#include "CondFormats/ESObjects/interface/ESIntercalibConstants.h"
#include "CondFormats/DataRecord/interface/ESIntercalibConstantsRcd.h"

#include "CondFormats/ESObjects/interface/ESADCToGeVConstant.h"
#include "CondFormats/DataRecord/interface/ESADCToGeVConstantRcd.h"

#include "CondFormats/ESObjects/interface/ESChannelStatus.h"
#include "CondFormats/DataRecord/interface/ESChannelStatusRcd.h"

#include "FWCore/Framework/interface/IOVSyncValue.h"

// forward declarations

namespace edm{
  class ParameterSet;
}

class ESTrivialConditionRetriever : public edm::ESProducer, 
                                      public edm::EventSetupRecordIntervalFinder
{

public:
  ESTrivialConditionRetriever(const edm::ParameterSet&  pset);
  virtual ~ESTrivialConditionRetriever();

  // ---------- member functions ---------------------------
  virtual std::auto_ptr<ESPedestals> produceESPedestals( const ESPedestalsRcd& );
  virtual std::auto_ptr<ESWeightStripGroups> produceESWeightStripGroups( const ESWeightStripGroupsRcd& );
  virtual std::auto_ptr<ESIntercalibConstants> produceESIntercalibConstants( const ESIntercalibConstantsRcd& );

  //  virtual std::auto_ptr<ESIntercalibErrors> produceESIntercalibErrors( const ESIntercalibErrorsRcd& );
  //  virtual std::auto_ptr<ESIntercalibErrors>  getIntercalibErrorsFromConfiguration ( const ESIntercalibErrorsRcd& ) ;

  virtual std::auto_ptr<ESADCToGeVConstant> produceESADCToGeVConstant( const ESADCToGeVConstantRcd& );
  virtual std::auto_ptr<ESTBWeights> produceESTBWeights( const ESTBWeightsRcd& );
  //  virtual std::auto_ptr<ESIntercalibConstants>  getIntercalibConstantsFromConfiguration ( const ESIntercalibConstantsRcd& ) ;

  virtual std::auto_ptr<ESChannelStatus> produceESChannelStatus( const ESChannelStatusRcd& );
  virtual std::auto_ptr<ESChannelStatus> getChannelStatusFromConfiguration( const ESChannelStatusRcd& );

protected:
  //overriding from ContextRecordIntervalFinder
  virtual void setIntervalFor( const edm::eventsetup::EventSetupRecordKey&,
                               const edm::IOVSyncValue& ,
                               edm::ValidityInterval& ) ;
private:
  ESTrivialConditionRetriever( const ESTrivialConditionRetriever& ); // stop default
  const  ESTrivialConditionRetriever& operator=( const ESTrivialConditionRetriever& ); // stop default

  void getWeightsFromConfiguration(const edm::ParameterSet& ps);

  // data members
  double adcToGeVLowConstant_;      // ADC -> GeV scale low
  double adcToGeVHighConstant_;      // ADC -> GeV scale high

  double intercalibConstantMean_;  // mean of intercalib constant. default: 1.0
  double intercalibConstantSigma_; // sigma of intercalib constant
                                  // Gaussian used to generate intercalib constants for
                                  // each channel. no smearing if sigma=0.0 (default)
  // double intercalibErrorMean_;  // mean of intercalib constant error

  double ESpedMean_;              // pedestal mean pedestal at gain 12
  double ESpedRMS_;               // pedestal rms at gain 12

  ESWeightSet amplWeights_;  // weights to compute amplitudes low

  std::string amplWeightsFile_;
  std::string intercalibConstantsFile_ ;
  std::string channelStatusFile_ ;

  bool getWeightsFromFile_;
  bool producedESPedestals_;
  bool producedESWeights_;
  bool producedESIntercalibConstants_;
  bool producedESADCToGeVConstant_;
  bool producedESChannelStatus_;

  int    verbose_; // verbosity

};
#endif
