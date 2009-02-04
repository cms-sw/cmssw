#ifndef CondTools_L1Trigger_L1CondDBIOVWriter_h
#define CondTools_L1Trigger_L1CondDBIOVWriter_h
// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1CondDBIOVWriter
// 
/**\class L1CondDBIOVWriter L1CondDBIOVWriter.h CondTools/L1Trigger/interface/L1CondDBIOVWriter.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Werner Sun
//         Created:  Sun Mar  2 20:10:36 CET 2008
// $Id$
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondTools/L1Trigger/interface/DataWriter.h"
#include "CondTools/L1Trigger/interface/DataReader.h"

// forward declarations

class L1CondDBIOVWriter : public edm::EDAnalyzer {
   public:
      explicit L1CondDBIOVWriter(const edm::ParameterSet&);
      ~L1CondDBIOVWriter();


   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // ----------member data ---------------------------
      l1t::DataWriter m_writer ;
      l1t::DataReader m_reader ;
      std::string m_keyTag ;

      // Map of records to tags
      std::map<std::string, std::string > m_recordToTagMap ;
};


#endif
