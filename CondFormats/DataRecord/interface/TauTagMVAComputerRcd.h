#ifndef DataRecord_TauTagMVAComputerRcd_h
#define DataRecord_TauTagMVAComputerRcd_h
// -*- C++ -*-
//
// Package:     DataRecord
// Class  :     TauTagMVAComputerRcd
// 
/**\class TauTagMVAComputerRcd TauTagMVAComputerRcd.h CondFormats/DataRecord/interface/TauTagMVAComputerRcd.h

 Description:  Record to persist MVAComputerContainer objects used in tau MVA discrimination

*/
//
// Author:      Evan K. Friis, friis@physics.ucdavis.edu
// Created:     Wed Nov 12 10:52:48 PST 2008
// 
//

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"

class TauTagMVAComputerRcd : public edm::eventsetup::EventSetupRecordImplementation<TauTagMVAComputerRcd> {};

#endif
