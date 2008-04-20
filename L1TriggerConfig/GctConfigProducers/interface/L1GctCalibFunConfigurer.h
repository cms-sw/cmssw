#ifndef L1GCTCALIBFUNCONFIGURER_H_
#define L1GCTCALIBFUNCONFIGURER_H_
// -*- C++ -*-
//
// Package:    GctConfigProducers
// Class:      L1GctCalibFunConfigurer
// 
/**\class L1GctCalibFunConfigurer L1GctCalibFunConfigurer.h L1Trigger/L1GctConfigProducers/interface/L1GctCalibFunConfigurer.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Gregory Heath
//
//


// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

#include<vector>

// user include files

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/L1TObjects/interface/L1GctJetEtCalibrationFunction.h"

//
// class declaration
//

class L1GctCalibFunConfigurer {
   public:
      L1GctCalibFunConfigurer(const edm::ParameterSet&);
      ~L1GctCalibFunConfigurer();

      typedef boost::shared_ptr<L1GctJetEtCalibrationFunction> CalibFunReturnType;

      CalibFunReturnType produceCalibFun();

   private:
      // ----------member data ---------------------------

  // PARAMETERS TO BE STORED IN THE CalibrationFunction
  /// scale and threshold parameters
  double m_htScaleLSB;
  double m_threshold;

  /// the calibration function - converts jet Et to linear 
  std::vector< std::vector<double> > m_jetCalibFunc;
  std::vector< std::vector<double> > m_tauCalibFunc;

  /// type of correction function to apply
  L1GctJetEtCalibrationFunction::CorrectionFunctionType m_corrFunType; 

  /// member functions to set up the ORCA-style calibrations (if needed)
  void setOrcaStyleParams();
  void setOrcaStyleParamsForBin(std::vector<double>& paramsForBin);

  /// member functions to set up the piecewise cubic calibrations (if needed)
  void setPiecewiseCubicParams();
  void setPiecewiseCubicParamsForBin(std::vector<double>& paramsForBin);

};

#endif

