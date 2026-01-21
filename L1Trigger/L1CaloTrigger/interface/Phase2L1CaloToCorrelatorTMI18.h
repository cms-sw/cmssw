#ifndef Phase2L1CaloToCorrelatorTMI18_H
#define Phase2L1CaloToCorrelatorTMI18_H


// system include files
#include <memory>
#include <unistd.h>


#include <iostream>
#include <fstream>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
//#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include <memory>
#include <math.h>
#include <vector>
#include <list>
#include <TLorentzVector.h>

#include "FWCore/Framework/interface/ESHandle.h"
//#include "DataFormats/L1TCalorimeterPhase2/interface/DigitizedClusterCorrelatorTMI18.h"
//#include "DataFormats/L1TCalorimeterPhase2/interface/DigitizedPFClusterCorrelatorTMI18.h"
#include "DataFormats/L1TCalorimeterPhase2/interface/GCTEmDigiCluster.h"
#include "DataFormats/L1TCalorimeterPhase2/interface/GCTHadDigiCluster.h"
#include "DataFormats/L1TCalorimeterPhase2/interface/DigitizedCaloToCorrelatorTMI18.h"

#ifdef __MAKECINT__
// #pragma extra_include "TLorentzVector.h";
#pragma link C++ class std::vector<TLorentzVector>;
#endif

//
// class declaration
//
using std::vector;

#endif
