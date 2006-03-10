//
// $Id: EcalTrivialConditionRetriever.h,v 1.1 2006/03/02 17:03:43 rahatlou Exp $
// Created: 2 Mar 2006
//          Shahram Rahatlou, University of Rome & INFN
//
#ifndef CalibCalorimetry_EcalPlugins_EcalTrivialConditionRetriever_H
#define CalibCalorimetry_EcalPlugins_EcalTrivialConditionRetriever_H
// system include files
#include <string>
#include <vector>

// user include files
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"

#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"

#include "CondFormats/EcalObjects/interface/EcalXtalGroupId.h"
#include "CondFormats/EcalObjects/interface/EcalWeightXtalGroups.h"
#include "CondFormats/DataRecord/interface/EcalWeightXtalGroupsRcd.h"

#include "CondFormats/EcalObjects/interface/EcalWeight.h"
#include "CondFormats/EcalObjects/interface/EcalWeightSet.h"
#include "CondFormats/EcalObjects/interface/EcalTBWeights.h"
#include "CondFormats/DataRecord/interface/EcalTBWeightsRcd.h"

#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"

#include "CondFormats/EcalObjects/interface/EcalMGPAGainRatio.h"
#include "CondFormats/EcalObjects/interface/EcalGainRatios.h"
#include "CondFormats/DataRecord/interface/EcalGainRatiosRcd.h"

#include "CondFormats/EcalObjects/interface/EcalADCToGeVConstant.h"
#include "CondFormats/DataRecord/interface/EcalADCToGeVConstantRcd.h"

#include "CondFormats/EcalObjects/interface/EcalWeightRecAlgoWeights.h"
#include "CondFormats/DataRecord/interface/EcalWeightRecAlgoWeightsRcd.h"

#include "CondCore/IOVService/interface/IOV.h"

// forward declarations

namespace edm{
  class ParameterSet;
};

class EcalTrivialConditionRetriever : public edm::ESProducer, 
                                      public edm::EventSetupRecordIntervalFinder
{

public:
  EcalTrivialConditionRetriever(const edm::ParameterSet&  pset);
  virtual ~EcalTrivialConditionRetriever();

  // ---------- member functions ---------------------------
  virtual std::auto_ptr<EcalPedestals> produceEcalPedestals( const EcalPedestalsRcd& );
  virtual std::auto_ptr<EcalWeightXtalGroups> produceEcalWeightXtalGroups( const EcalWeightXtalGroupsRcd& );
  virtual std::auto_ptr<EcalIntercalibConstants> produceEcalIntercalibConstants( const EcalIntercalibConstantsRcd& );
  virtual std::auto_ptr<EcalGainRatios> produceEcalGainRatios( const EcalGainRatiosRcd& );
  virtual std::auto_ptr<EcalADCToGeVConstant> produceEcalADCToGeVConstant( const EcalADCToGeVConstantRcd& );
  virtual std::auto_ptr<EcalTBWeights> produceEcalTBWeights( const EcalTBWeightsRcd& );


protected:
  //overriding from ContextRecordIntervalFinder
  virtual void setIntervalFor( const edm::eventsetup::EventSetupRecordKey&,
                               const edm::IOVSyncValue& ,
                               edm::ValidityInterval& ) ;
private:
  EcalTrivialConditionRetriever( const EcalTrivialConditionRetriever& ); // stop default
  const  EcalTrivialConditionRetriever& operator=( const EcalTrivialConditionRetriever& ); // stop default

  // data members
  double adcToGeVEBConstant_;      // ADC -> GeV scale for barrel

  double intercalibConstantMean_;  // mean of intercalib constant. default: 1.0
  double intercalibConstantSigma_; // sigma of intercalib constant
                                  // Gaussian used to generate intercalib constants for
                                  // each channel. no smearing if sigma=0.0 (default)

  double pedMeanX12_;              // pedestal mean pedestal at gain 12
  double pedRMSX12_;               // pedestal rms at gain 12
  double pedMeanX6_;               // pedestal mean pedestal at gain 6
  double pedRMSX6_;                // pedestal rms at gain 6
  double pedMeanX1_;               // pedestal mean pedestal at gain 1
  double pedRMSX1_;                // pedestal rms at gain 1

  double gainRatio12over6_;        // ratio of MGPA gain12 / gain6
  double gainRatio6over1_;         // ratio of MGPA gain6 / gain1

  std::vector<EcalWeight> amplWeights_;  // weights to compute amplitudes after ped subtraction
  std::vector<EcalWeight> pedWeights_;  // weights to compute amplitudes w/o ped subtraction
  std::vector<EcalWeight> jittWeights_;  // weights to compute jitter


  int    verbose_; // verbosity



};
#endif
