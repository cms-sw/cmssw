/**
 * Interface to CDF Midpoint Cone Algorithm from fastjet package
 * F.Ratnikov, UMd, June 22, 2007
 * Redesigned on Aug. 1, 2007 by F.R.
 * $Id: CMSIterativeConeAlgorithm.cc,v 1.8 2007/07/20 18:46:38 fedor Exp $
 **/

#include <string>

#include "fastjet/JetDefinition.hh"
#include "fastjet/CDFMidPointPlugin.hh"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoJets/JetAlgorithms/interface/CDFMidpointAlgorithmWrapper.h"

CDFMidpointAlgorithmWrapper::CDFMidpointAlgorithmWrapper(const edm::ParameterSet& fConfig)
  : FastJetBaseWrapper (fConfig),
    mPlugin (0)
{}

CDFMidpointAlgorithmWrapper::~CDFMidpointAlgorithmWrapper () {
  delete mPlugin;
}

void CDFMidpointAlgorithmWrapper::makeJetDefinition (const edm::ParameterSet& fConfig) {
  //configuring algorithm 
  mPlugin = new fastjet::CDFMidPointPlugin (fConfig.getParameter<double>("seedThreshold"),
					    fConfig.getParameter<double>("coneRadius"),
					    fConfig.getParameter<double>("coneAreaFraction"),
					    fConfig.getParameter<int>("maxPairSize"),
					    fConfig.getParameter<int>("maxIterations"),
					    fConfig.getParameter<double>("overlapThreshold"));
  delete mJetDefinition;
  mJetDefinition = new fastjet::JetDefinition (mPlugin);
}
