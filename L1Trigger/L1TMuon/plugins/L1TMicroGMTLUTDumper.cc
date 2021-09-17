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
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "L1Trigger/L1TMuon/interface/MicroGMTRankPtQualLUT.h"
#include "L1Trigger/L1TMuon/interface/MicroGMTAbsoluteIsolationCheckLUT.h"
#include "L1Trigger/L1TMuon/interface/MicroGMTRelativeIsolationCheckLUT.h"
#include "L1Trigger/L1TMuon/interface/MicroGMTCaloIndexSelectionLUT.h"
#include "L1Trigger/L1TMuon/interface/MicroGMTExtrapolationLUT.h"
#include "L1Trigger/L1TMuon/interface/MicroGMTMatchQualLUT.h"
#include "L1Trigger/L1TMuon/interface/MicroGMTLUTFactories.h"

#include "CondFormats/L1TObjects/interface/L1TMuonGlobalParams.h"
#include "CondFormats/DataRecord/interface/L1TMuonGlobalParamsRcd.h"
#include "L1Trigger/L1TMuon/interface/L1TMuonGlobalParamsHelper.h"

#include <iostream>
//
// class declaration
//
using namespace l1t;

class L1TMicroGMTLUTDumper : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  explicit L1TMicroGMTLUTDumper(const edm::ParameterSet&);
  ~L1TMicroGMTLUTDumper() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void endRun(edm::Run const&, edm::EventSetup const&) override;

  void dumpLut(MicroGMTLUT*, const std::string&);

  // ----------member data ---------------------------
  std::unique_ptr<L1TMuonGlobalParamsHelper> microGMTParamsHelper;
  std::string m_foldername;

  std::shared_ptr<MicroGMTRankPtQualLUT> m_rankLUT;

  std::shared_ptr<MicroGMTAbsoluteIsolationCheckLUT> m_absIsoCheckMemLUT;
  std::shared_ptr<MicroGMTRelativeIsolationCheckLUT> m_relIsoCheckMemLUT;

  std::shared_ptr<MicroGMTCaloIndexSelectionLUT> m_idxSelMemPhiLUT;
  std::shared_ptr<MicroGMTCaloIndexSelectionLUT> m_idxSelMemEtaLUT;

  std::shared_ptr<MicroGMTExtrapolationLUT> m_bPhiExtrapolationLUT;
  std::shared_ptr<MicroGMTExtrapolationLUT> m_oPhiExtrapolationLUT;
  std::shared_ptr<MicroGMTExtrapolationLUT> m_fPhiExtrapolationLUT;
  std::shared_ptr<MicroGMTExtrapolationLUT> m_bEtaExtrapolationLUT;
  std::shared_ptr<MicroGMTExtrapolationLUT> m_oEtaExtrapolationLUT;
  std::shared_ptr<MicroGMTExtrapolationLUT> m_fEtaExtrapolationLUT;

  std::shared_ptr<MicroGMTMatchQualLUT> m_boPosMatchQualLUT;
  std::shared_ptr<MicroGMTMatchQualLUT> m_boNegMatchQualLUT;
  std::shared_ptr<MicroGMTMatchQualLUT> m_foPosMatchQualLUT;
  std::shared_ptr<MicroGMTMatchQualLUT> m_foNegMatchQualLUT;
  //std::shared_ptr<MicroGMTMatchQualLUT> m_brlSingleMatchQualLUT;
  std::shared_ptr<MicroGMTMatchQualLUT> m_ovlPosSingleMatchQualLUT;
  std::shared_ptr<MicroGMTMatchQualLUT> m_ovlNegSingleMatchQualLUT;
  std::shared_ptr<MicroGMTMatchQualLUT> m_fwdPosSingleMatchQualLUT;
  std::shared_ptr<MicroGMTMatchQualLUT> m_fwdNegSingleMatchQualLUT;
  edm::ESGetToken<L1TMuonGlobalParams, L1TMuonGlobalParamsRcd> m_microGMTParamsToken;
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
L1TMicroGMTLUTDumper::L1TMicroGMTLUTDumper(const edm::ParameterSet& iConfig) {
  //now do what ever other initialization is needed
  m_foldername = iConfig.getParameter<std::string>("out_directory");
  m_microGMTParamsToken = esConsumes<L1TMuonGlobalParams, L1TMuonGlobalParamsRcd, edm::Transition::BeginRun>();

  microGMTParamsHelper = std::make_unique<L1TMuonGlobalParamsHelper>();
}

L1TMicroGMTLUTDumper::~L1TMicroGMTLUTDumper() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//
void L1TMicroGMTLUTDumper::dumpLut(MicroGMTLUT* lut, const std::string& oName) {
  std::ofstream fStream(m_foldername + oName);
  lut->save(fStream);
  fStream.close();
}

// ------------ method called to produce the data  ------------
void L1TMicroGMTLUTDumper::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  dumpLut(m_rankLUT.get(), std::string("/SortRank.txt"));
  dumpLut(m_absIsoCheckMemLUT.get(), std::string("/AbsIsoCheckMem.txt"));
  dumpLut(m_relIsoCheckMemLUT.get(), std::string("/RelIsoCheckMem.txt"));
  dumpLut(m_idxSelMemPhiLUT.get(), std::string("/IdxSelMemPhi.txt"));
  dumpLut(m_idxSelMemEtaLUT.get(), std::string("/IdxSelMemEta.txt"));
  dumpLut(m_bPhiExtrapolationLUT.get(), std::string("/BPhiExtrapolation.txt"));
  dumpLut(m_oPhiExtrapolationLUT.get(), std::string("/OPhiExtrapolation.txt"));
  dumpLut(m_fPhiExtrapolationLUT.get(), std::string("/EPhiExtrapolation.txt"));
  dumpLut(m_bEtaExtrapolationLUT.get(), std::string("/BEtaExtrapolation.txt"));
  dumpLut(m_oEtaExtrapolationLUT.get(), std::string("/OEtaExtrapolation.txt"));
  dumpLut(m_fEtaExtrapolationLUT.get(), std::string("/EEtaExtrapolation.txt"));
  dumpLut(m_boPosMatchQualLUT.get(), std::string("/BOPosMatchQual.txt"));
  dumpLut(m_boNegMatchQualLUT.get(), std::string("/BONegMatchQual.txt"));
  dumpLut(m_foPosMatchQualLUT.get(), std::string("/EOPosMatchQual.txt"));
  dumpLut(m_foNegMatchQualLUT.get(), std::string("/EONegMatchQual.txt"));
  //dumpLut(m_brlSingleMatchQualLUT.get(), std::string("/BmtfSingleMatchQual.txt"));
  dumpLut(m_ovlPosSingleMatchQualLUT.get(), std::string("/OmtfPosSingleMatchQual.txt"));
  dumpLut(m_ovlNegSingleMatchQualLUT.get(), std::string("/OmtfNegSingleMatchQual.txt"));
  dumpLut(m_fwdPosSingleMatchQualLUT.get(), std::string("/EmtfPosSingleMatchQual.txt"));
  dumpLut(m_fwdNegSingleMatchQualLUT.get(), std::string("/EmtfNegSingleMatchQual.txt"));
}

// ------------ method called when starting to processes a run  ------------
void L1TMicroGMTLUTDumper::beginRun(edm::Run const& run, edm::EventSetup const& iSetup) {
  edm::ESHandle<L1TMuonGlobalParams> microGMTParamsHandle = iSetup.getHandle(m_microGMTParamsToken);

  microGMTParamsHelper = std::make_unique<L1TMuonGlobalParamsHelper>(*microGMTParamsHandle.product());
  if (!microGMTParamsHelper) {
    edm::LogError("L1TMicroGMTLUTDumper") << "Could not retrieve parameters from Event Setup" << std::endl;
  }

  int fwVersion = microGMTParamsHelper->fwVersion();
  m_rankLUT = MicroGMTRankPtQualLUTFactory::create(microGMTParamsHelper->sortRankLUT(), fwVersion);

  m_absIsoCheckMemLUT =
      MicroGMTAbsoluteIsolationCheckLUTFactory::create(microGMTParamsHelper->absIsoCheckMemLUT(), fwVersion);
  m_relIsoCheckMemLUT =
      MicroGMTRelativeIsolationCheckLUTFactory::create(microGMTParamsHelper->relIsoCheckMemLUT(), fwVersion);
  m_idxSelMemPhiLUT = MicroGMTCaloIndexSelectionLUTFactory::create(
      microGMTParamsHelper->idxSelMemPhiLUT(), l1t::MicroGMTConfiguration::PHI, fwVersion);
  m_idxSelMemEtaLUT = MicroGMTCaloIndexSelectionLUTFactory::create(
      microGMTParamsHelper->idxSelMemEtaLUT(), l1t::MicroGMTConfiguration::ETA, fwVersion);

  m_bPhiExtrapolationLUT = MicroGMTExtrapolationLUTFactory::create(
      microGMTParamsHelper->bPhiExtrapolationLUT(), l1t::MicroGMTConfiguration::PHI_OUT, fwVersion);
  m_oPhiExtrapolationLUT = MicroGMTExtrapolationLUTFactory::create(
      microGMTParamsHelper->oPhiExtrapolationLUT(), l1t::MicroGMTConfiguration::PHI_OUT, fwVersion);
  m_fPhiExtrapolationLUT = MicroGMTExtrapolationLUTFactory::create(
      microGMTParamsHelper->fPhiExtrapolationLUT(), l1t::MicroGMTConfiguration::PHI_OUT, fwVersion);
  m_bEtaExtrapolationLUT = MicroGMTExtrapolationLUTFactory::create(
      microGMTParamsHelper->bEtaExtrapolationLUT(), l1t::MicroGMTConfiguration::ETA_OUT, fwVersion);
  m_oEtaExtrapolationLUT = MicroGMTExtrapolationLUTFactory::create(
      microGMTParamsHelper->oEtaExtrapolationLUT(), l1t::MicroGMTConfiguration::ETA_OUT, fwVersion);
  m_fEtaExtrapolationLUT = MicroGMTExtrapolationLUTFactory::create(
      microGMTParamsHelper->fEtaExtrapolationLUT(), l1t::MicroGMTConfiguration::ETA_OUT, fwVersion);

  m_boPosMatchQualLUT = l1t::MicroGMTMatchQualLUTFactory::create(
      microGMTParamsHelper->bOPosMatchQualLUT(), cancel_t::omtf_bmtf_pos, fwVersion);
  m_boNegMatchQualLUT = l1t::MicroGMTMatchQualLUTFactory::create(
      microGMTParamsHelper->bONegMatchQualLUT(), cancel_t::omtf_bmtf_neg, fwVersion);
  m_foPosMatchQualLUT = l1t::MicroGMTMatchQualLUTFactory::create(
      microGMTParamsHelper->fOPosMatchQualLUT(), cancel_t::omtf_emtf_pos, fwVersion);
  m_foNegMatchQualLUT = l1t::MicroGMTMatchQualLUTFactory::create(
      microGMTParamsHelper->fONegMatchQualLUT(), cancel_t::omtf_emtf_neg, fwVersion);
  //m_brlSingleMatchQualLUT = l1t::MicroGMTMatchQualLUTFactory::create(microGMTParamsHelper->brlSingleMatchQualLUT(), cancel_t::bmtf_bmtf, fwVersion);
  m_ovlPosSingleMatchQualLUT = l1t::MicroGMTMatchQualLUTFactory::create(
      microGMTParamsHelper->ovlPosSingleMatchQualLUT(), cancel_t::omtf_omtf_pos, fwVersion);
  m_ovlNegSingleMatchQualLUT = l1t::MicroGMTMatchQualLUTFactory::create(
      microGMTParamsHelper->ovlNegSingleMatchQualLUT(), cancel_t::omtf_omtf_neg, fwVersion);
  m_fwdPosSingleMatchQualLUT = l1t::MicroGMTMatchQualLUTFactory::create(
      microGMTParamsHelper->fwdPosSingleMatchQualLUT(), cancel_t::emtf_emtf_pos, fwVersion);
  m_fwdNegSingleMatchQualLUT = l1t::MicroGMTMatchQualLUTFactory::create(
      microGMTParamsHelper->fwdNegSingleMatchQualLUT(), cancel_t::emtf_emtf_neg, fwVersion);
}

// ------------ method called when ending to processes a run  ------------
void L1TMicroGMTLUTDumper::endRun(edm::Run const&, edm::EventSetup const&) {}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TMicroGMTLUTDumper);
