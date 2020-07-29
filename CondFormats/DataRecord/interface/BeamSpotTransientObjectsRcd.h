#ifndef DataRecord_BeamSpotTransientObjectsRcd_h
#define DataRecord_BeamSpotTransientObjectsRcd_h
// -*- C++ -*-
//
// Package:     DataRecord
// Class  :     BeamSpotTransientObjectsRcd
//
/**\class BeamSpotTransientObjectsRcd BeamSpotTransientObjectsRcd.h CondFormats/DataRecord/interface/BeamSpotTransientObjectsRcd.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:
// Created:     Tue Mar  6 19:34:33 CST 2007
// $Id$
//
#include "FWCore/Utilities/interface/mplVector.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/BeamSpotOnlineHLTObjectsRcd.h"
#include "CondFormats/DataRecord/interface/BeamSpotOnlineLegacyObjectsRcd.h"
#include "CondFormats/DataRecord/interface/BeamSpotObjectsRcd.h"

class BeamSpotTransientObjectsRcd
    : public edm::eventsetup::DependentRecordImplementation<
          BeamSpotTransientObjectsRcd,
<<<<<<< HEAD
          edm::mpl::Vector<BeamSpotOnlineHLTObjectsRcd, BeamSpotOnlineLegacyObjectsRcd, BeamSpotObjectsRcd> > {};
=======
          boost::mpl::vector<BeamSpotOnlineHLTObjectsRcd, BeamSpotOnlineLegacyObjectsRcd, BeamSpotObjectsRcd> > {};
>>>>>>> applying code style patch

#endif
