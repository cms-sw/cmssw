/*
 * AlgoMuonBase.cc
 *
 *  Created on: Mar 1, 2019
 *      Author: Karol Bunkowski kbunkow@cern.ch
 */

#include "L1Trigger/L1TMuonOverlapPhase1/interface/AlgoMuonBase.h"

AlgoMuonBase::AlgoMuonBase(const ProcConfigurationBase* config)
    : firedLayerBitsInBx(config->getBxToProcess(), boost::dynamic_bitset<>(config->nLayers())) {}

AlgoMuonBase::~AlgoMuonBase() {}
