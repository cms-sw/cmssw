// -*- C++ -*-
//
// Package:    L1TMicroGMTLUTDumper
// Class:      L1TMicroGMTLUTDumper
//
/**\class L1TMicroGMTLUTDumper L1TMicroGMTLUTDumper.cc L1Trigger/L1TGlobalMuon/plugins/L1TMicroGMTLUTDumper.cc

 Description: Takes txt-file input and produces barrel- / overlap- / forward TF muons

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Joschka Philip Lingemann,40 3-B01,+41227671598,
//         Created:  Thu Oct  3 10:12:30 CEST 2013
// $Id$
//
//


// system include files
#include <memory>
#include <fstream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "L1Trigger/L1TMuon/interface/MicroGMTRankPtQualLUT.h"
#include "L1Trigger/L1TMuon/interface/MicroGMTMatchQualLUT.h"


#include <iostream>
//
// class declaration
//
using namespace l1t;

class L1TMicroGMTLUTDumper : public edm::EDAnalyzer {
   public:
      explicit L1TMicroGMTLUTDumper(const edm::ParameterSet&);
      ~L1TMicroGMTLUTDumper();
      virtual void analyze(const edm::Event&, const edm::EventSetup&);

   private:
      void dumpLut(MicroGMTLUT*, const std::string&);

      // ----------member data ---------------------------
      std::string m_foldername;
      MicroGMTRankPtQualLUT m_rankLUT;

      MicroGMTMatchQualLUT m_boPosMatchQualLUT;
      MicroGMTMatchQualLUT m_boNegMatchQualLUT;
      MicroGMTMatchQualLUT m_foPosMatchQualLUT;
      MicroGMTMatchQualLUT m_foNegMatchQualLUT;
      MicroGMTMatchQualLUT m_brlSingleMatchQualLUT;
      MicroGMTMatchQualLUT m_ovlPosSingleMatchQualLUT;
      MicroGMTMatchQualLUT m_ovlNegSingleMatchQualLUT;
      MicroGMTMatchQualLUT m_fwdPosSingleMatchQualLUT;
      MicroGMTMatchQualLUT m_fwdNegSingleMatchQualLUT;
};

//
// constants, enums and typedefs
//


//
// static data member definitions
//

//
// constructors and destructor
//
L1TMicroGMTLUTDumper::L1TMicroGMTLUTDumper(const edm::ParameterSet& iConfig) :
    m_rankLUT(iConfig),
    m_boPosMatchQualLUT(iConfig, "BOPos", cancel_t::omtf_bmtf_pos),
    m_boNegMatchQualLUT(iConfig, "BONeg", cancel_t::omtf_bmtf_neg),
    m_foPosMatchQualLUT(iConfig, "FOPos", cancel_t::omtf_emtf_pos),
    m_foNegMatchQualLUT(iConfig, "FONeg", cancel_t::omtf_emtf_neg),
    m_brlSingleMatchQualLUT(iConfig, "BrlSingle", cancel_t::bmtf_bmtf),
    m_ovlPosSingleMatchQualLUT(iConfig, "OvlPosSingle", cancel_t::omtf_omtf_pos),
    m_ovlNegSingleMatchQualLUT(iConfig, "OvlNegSingle", cancel_t::omtf_omtf_neg),
    m_fwdPosSingleMatchQualLUT(iConfig, "FwdPosSingle", cancel_t::emtf_emtf_pos),
    m_fwdNegSingleMatchQualLUT(iConfig, "FwdNegSingle", cancel_t::emtf_emtf_neg)
{
  //register your products

  //now do what ever other initialization is needed
  m_foldername = iConfig.getParameter<std::string> ("out_directory");


}


L1TMicroGMTLUTDumper::~L1TMicroGMTLUTDumper()
{
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}


//
// member functions
//
void
L1TMicroGMTLUTDumper::dumpLut(MicroGMTLUT* lut, const std::string& oName) {
  std::ofstream fStream(m_foldername+oName);
  lut->save(fStream);
  fStream.close();
}



// ------------ method called to produce the data  ------------
void
L1TMicroGMTLUTDumper::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  dumpLut(&m_rankLUT, std::string("/rank_lut.json"));
  dumpLut(&m_boPosMatchQualLUT, std::string("/boPosMatchQualLUT.json"));
  dumpLut(&m_boNegMatchQualLUT, std::string("/boNegMatchQualLUT.json"));
  dumpLut(&m_foPosMatchQualLUT, std::string("/foPosMatchQualLUT.json"));
  dumpLut(&m_foNegMatchQualLUT, std::string("/foNegMatchQualLUT.json"));
  dumpLut(&m_brlSingleMatchQualLUT, std::string("/brlSingleMatchQualLUT.json"));
  dumpLut(&m_ovlPosSingleMatchQualLUT, std::string("/ovlPosSingleMatchQualLUT.json"));
  dumpLut(&m_ovlNegSingleMatchQualLUT, std::string("/ovlNegSingleMatchQualLUT.json"));
  dumpLut(&m_fwdPosSingleMatchQualLUT, std::string("/fwdPosSingleMatchQualLUT.json"));
  dumpLut(&m_fwdNegSingleMatchQualLUT, std::string("/fwdNegSingleMatchQualLUT.json"));

}


//define this as a plug-in
DEFINE_FWK_MODULE(L1TMicroGMTLUTDumper);
