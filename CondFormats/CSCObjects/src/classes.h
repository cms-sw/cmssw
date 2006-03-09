#include "CondFormats/CSCObjects/interface/CSCPedestals.h"
template std::map< int, std::vector<CSCPedestals::Item> >::iterator;
template std::map< int, std::vector<CSCPedestals::Item> >::const_iterator;
template std::vector< CSCPedestals::Item >::iterator;
template std::vector< CSCPedestals::Item >::const_iterator;

#include "CondFormats/CSCObjects/interface/CSCGains.h"
template std::map< int, std::vector< CSCGains::Item> >::iterator;
template std::map< int, std::vector< CSCGains::Item> >::const_iterator;
template std::vector< CSCGains::Item >::iterator;
template std::vector< CSCGains::Item >::const_iterator;

#include "CondFormats/CSCObjects/interface/CSCNoiseMatrix.h"
template std::map< int, std::vector< CSCNoiseMatrix::Item> >::iterator;
template std::map< int, std::vector< CSCNoiseMatrix::Item> >::const_iterator;
template std::vector< CSCNoiseMatrix::Item >::iterator;
template std::vector< CSCNoiseMatrix::Item >::const_iterator;

#include "CondFormats/CSCObjects/interface/CSCcrosstalk.h"
template std::map< int, std::vector< CSCcrosstalk::Item> >::iterator;
template std::map< int, std::vector< CSCcrosstalk::Item> >::const_iterator;
template std::vector< CSCcrosstalk::Item >::iterator;
template std::vector< CSCcrosstalk::Item >::const_iterator;

#include "CondFormats/CSCObjects/interface/CSCIdentifier.h"
template std::vector< CSCIdentifier::Item >::iterator;
template std::vector< CSCIdentifier::Item >::const_iterator;

#include "CondFormats/CSCObjects/interface/CSCReadoutMapping.h"
template std::vector< CSCReadoutMapping::CSCLabel >::iterator;
template std::vector< CSCReadoutMapping::CSCLabel >::const_iterator;

#include "CondFormats/CSCObjects/interface/CSCTriggerMapping.h"
template std::vector< CSCTriggerMapping::CSCTriggerConnection >::iterator;
template std::vector< CSCTriggerMapping::CSCTriggerConnection >::const_iterator;
