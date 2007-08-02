
/**
 * Interface to Seedless Infrared Safe Cone algorithm (http://projects.hepforge.org/siscone)
 * F.Ratnikov, UMd, June 22, 2007
 * Redesigned on Aug. 1, 2007 by F.R.
 * $Id: SISConeAlgorithmWrapper.cc,v 1.2 2007/08/02 17:42:58 fedor Exp $
 **/

#include <string>

#include "fastjet/JetDefinition.hh"
#include "fastjet/SISConePlugin.hh"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoJets/JetAlgorithms/interface/SISConeAlgorithmWrapper.h"

SISConeAlgorithmWrapper::SISConeAlgorithmWrapper(const edm::ParameterSet& fConfig)
  : FastJetBaseWrapper (fConfig),
    mPlugin (0)
{
  //configuring algorithm 
  std::string splitMergeScale (fConfig.getParameter<std::string>("splitMergeScale"));
  fastjet::SISConePlugin::SplitMergeScale scale = fastjet::SISConePlugin::SM_pttilde;
  if (splitMergeScale == "pt") scale = fastjet::SISConePlugin::SM_pt;
  else if (splitMergeScale == "Et") scale = fastjet::SISConePlugin::SM_Et;
  else if (splitMergeScale == "mt") scale = fastjet::SISConePlugin::SM_mt;
  else if (splitMergeScale != "pttilde")  edm::LogError("SISConeJetDefinition") 
    << "SISConeAlgorithmWrapper::SISConeAlgorithmWrapper-> Unknown scale " << splitMergeScale
    << ". Known scales are: pt Et mt pttilde";

  mPlugin = new fastjet::SISConePlugin (fConfig.getParameter<double>("coneRadius"), 
					fConfig.getParameter<double>("coneOverlapThreshold"),
					fConfig.getParameter<int>("maxPasses"),
					fConfig.getParameter<double>("protojetPtMin"),
					fConfig.getParameter<bool>("caching"),
					scale);
  mJetDefinition = new fastjet::JetDefinition (mPlugin);
}

SISConeAlgorithmWrapper::~SISConeAlgorithmWrapper () {
  delete mPlugin;
}

