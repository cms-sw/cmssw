#ifndef RecoTauTag_TauTagTools_TauMVADBConfiguration_H
#define RecoTauTag_TauTagTools_TauMVADBConfiguration_H

/*
 * TauMVADBConfiguration.h
 *
 * Serves as a single point of configuration for the conditions
 * database identifiers etc for the Tau MVA package.
 *
 * Intended to facilitate easy branching of the tau code across
 * releases w/ different DB constraints
 *
 * Author: Evan K. Friis, UC Davis (friis@physics.ucdavis.edu)
 *
 */

// for CMSSW > 3_0
// requires 
//        V05-03-00 CondFormats/DataRecord
//        V00-02-00 CondCore/BTauPlugins


#include "CondFormats/DataRecord/interface/TauTagMVAComputerRcd.h"
typedef TauTagMVAComputerRcd TauMVAFrameworkDBRcd;

// for 2_2 < CMSSW < 3_0
//#include "CondFormats/DataRecord/interface/BTauGenericMVAJetTagComputerRcd.h"
//typedef BTauGenericMVAJetTagComputerRcd TauMVAFrameworkDBRcd ;

#endif
