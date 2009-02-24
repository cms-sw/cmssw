#ifndef L1GCTJFPARAMSCONFIGURER_H_
#define L1GCTJFPARAMSCONFIGURER_H_
// -*- C++ -*-
//
// Package:    GctConfigProducers
// Class:      L1GctJfParamsConfigurer
// 
/**\class L1GctJfParamsConfigurer L1GctJfParamsConfigurer.h L1Trigger/L1GctConfigProducers/interface/L1GctJfParamsConfigurer.h

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

#include "CondFormats/L1TObjects/interface/L1GctJetFinderParams.h"

class L1CaloGeometry;

//
// class declaration
//

class L1GctJfParamsConfigurer {
 public:
  L1GctJfParamsConfigurer(const edm::ParameterSet&);
  ~L1GctJfParamsConfigurer();
  
  typedef boost::shared_ptr<L1GctJetFinderParams>          JfParamsReturnType;
  
  JfParamsReturnType produceJfParams(const L1CaloGeometry* geom);
  
 private:  // methods
  
  /// member functions to set up the ORCA-style calibrations (if needed)
  void setOrcaStyleParams();
  void setOrcaStyleParamsForBin(std::vector<double>& paramsForBin);

  /// member functions to set up the piecewise cubic calibrations (if needed)
  void setPiecewiseCubicParams();
  void setPiecewiseCubicParamsForBin(std::vector<double>& paramsForBin);

  std::vector<double> etToEnergyConversion( const L1CaloGeometry* geom) const;

 private:   // data

  // parameters to be stored in L1GctJetFinderParams
  double m_rgnEtLsb;
  double m_htLsb;
  double m_CenJetSeed;
  double m_FwdJetSeed;
  double m_TauJetSeed;
  double m_tauIsoThresh;
  double m_htJetThresh;
  double m_mhtJetThresh;
  unsigned m_EtaBoundry;
  unsigned m_corrFunType;
  bool m_convertToEnergy;

  std::vector< std::vector<double> > m_jetCalibFunc;
  std::vector< std::vector<double> > m_tauCalibFunc;

};

#endif


