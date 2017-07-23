/** \file
 *  \based on CSCDigiToRaw
 *  \author J. Lee - UoS
 */

#include <algorithm>
#include "boost/dynamic_bitset.hpp"
#include "boost/foreach.hpp"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/CRC16.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"

#include "EventFilter/GEMRawToDigi/src/GEMRawToDigi.h"

using namespace edm;
using namespace std;

void GEMRawToDigi::readFedBuffers(const GEMDigiCollection& gemDigi,
				  const GEMPadDigiCollection& gemPadDigi,
				  const GEMPadDigiClusterCollection& gemPadDigiCluster,
				  const GEMCoPadDigiCollection& gemCoPadDigi,			
				  FEDRawDataCollection& fed_buffers,
				  const GEMChamberMap* theMapping, 
				  edm::Event & e)
{
  beginEvent(mapping);
  read(gemDigi);
  read(gemPadDigi);
  read(gemPadDigiCluster);
  read(gemCoPadDigi);
}

void GEMRawToDigi::read(const GEMDigiCollection& digis)
{
