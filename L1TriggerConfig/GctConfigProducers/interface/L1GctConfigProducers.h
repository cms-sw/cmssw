#ifndef L1GCTCONFIGPRODUCERS_H_
#define L1GCTCONFIGPRODUCERS_H_
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
//
//


// system include files
#include <memory>

#include<vector>

// user include files

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"

class L1CaloGeometry;

class L1GctJetFinderParams;
class L1GctChannelMask;

class L1GctJetFinderParamsRcd;
class L1GctChannelMaskRcd;


//
// class declaration
//

class L1GctConfigProducers : public edm::ESProducer {
 public:
  L1GctConfigProducers(const edm::ParameterSet&);
  ~L1GctConfigProducers() override;

  using JfParamsReturnType = std::unique_ptr<L1GctJetFinderParams>;
  using ChanMaskReturnType = std::unique_ptr<L1GctChannelMask>;

  JfParamsReturnType produceJfParams(const L1GctJetFinderParamsRcd&);
  ChanMaskReturnType produceChanMask(const L1GctChannelMaskRcd&);

  std::vector<double> etToEnergyConversion(const L1CaloGeometry* geom) const;
 
 private:
  // ----------member data ---------------------------
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

  unsigned m_metEtaMask;
  unsigned m_tetEtaMask;
  unsigned m_mhtEtaMask;
  unsigned m_thtEtaMask;

};

#endif


