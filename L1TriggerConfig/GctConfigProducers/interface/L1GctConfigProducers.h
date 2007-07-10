// -*- C++ -*-
//
// Package:    GctConfigProducers
// Class:      L1GctConfigProducers
// 
/**\class L1GctConfigProducers L1GctConfigProducers.h L1Trigger/L1GctConfigProducers/interface/L1GctConfigProducers.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Gregory Heath
//         Created:  Thu Mar  1 15:10:47 CET 2007
// $Id: L1GctConfigProducers.h,v 1.2 2007/04/26 11:22:52 heath Exp $
//
//


// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

#include<vector>

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/L1TObjects/interface/L1GctJetEtCalibrationFunction.h"
#include "CondFormats/L1TObjects/interface/L1GctJetFinderParams.h"
#include "CondFormats/L1TObjects/interface/L1CaloEtScale.h"


class L1GctJetCalibFunRcd;
class L1GctJetFinderParamsRcd;
class L1JetEtScaleRcd;

//
// class declaration
//

class L1GctConfigProducers : public edm::ESProducer {
   public:
      L1GctConfigProducers(const edm::ParameterSet&);
      ~L1GctConfigProducers();

      typedef boost::shared_ptr<L1GctJetEtCalibrationFunction> CalibFunReturnType;
      typedef boost::shared_ptr<L1GctJetFinderParams>          JfParamsReturnType;

      CalibFunReturnType produceCalibFun(const L1GctJetCalibFunRcd&);
      JfParamsReturnType produceJfParams(const L1GctJetFinderParamsRcd&);

      /// Add a dependency on the JetEtScale
      void doWhenChanged(const L1JetEtScaleRcd& jetScaleRcd);
   private:
      // ----------member data ---------------------------

  // PARAMETERS TO BE STORED IN THE JetFinderParameters
  /// seed thresholds and eta boundary
  unsigned m_CenJetSeed;
  unsigned m_FwdJetSeed;
  unsigned m_TauJetSeed;
  unsigned m_EtaBoundry;

  // PARAMETERS TO BE STORED IN THE CalibrationFunction
  /// scale and threshold parameters
  double m_htScaleLSB;
  double m_threshold;

  /// the calibration function - converts jet Et to linear 
  std::vector< std::vector<double> > m_jetCalibFunc;
  std::vector< std::vector<double> > m_tauCalibFunc;

  /// the jet Et scale to be used
  L1CaloEtScale m_jetScale; 

  /// type of correction function to apply
  L1GctJetEtCalibrationFunction::CorrectionFunctionType m_corrFunType; 

  /// member functions to set up the ORCA-style calibrations (if needed)
  void setOrcaStyleParams();
  void setOrcaStyleParamsForBin(std::vector<double>& paramsForBin);

};

