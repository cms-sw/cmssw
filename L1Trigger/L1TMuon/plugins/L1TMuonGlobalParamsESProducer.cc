// -*- C++ -*-
//
// Package:    L1Trigger/L1TMuonGlobalParamsESProducer
// Class:      L1TMuonGlobalParamsESProducer
// 
/**\class L1TMuonGlobalParamsESProducer L1TMuonGlobalParamsESProducer.h L1Trigger/L1TMuonGlobalParamsESProducer/plugins/L1TMuonGlobalParamsESProducer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Thomas Reis
//         Created:  Mon, 21 Sep 2015 13:28:49 GMT
//
//


// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESProducts.h"

#include "CondFormats/L1TObjects/interface/L1TMuonGlobalParams.h"
#include "CondFormats/DataRecord/interface/L1TMuonGlobalParamsRcd.h"
#include "L1Trigger/L1TMuon/interface/L1TMuonGlobalParamsHelper.h"
#include "L1Trigger/L1TMuon/interface/MicroGMTLUTFactories.h"

//
// class declaration
//

class L1TMuonGlobalParamsESProducer : public edm::ESProducer {
   public:
      L1TMuonGlobalParamsESProducer(const edm::ParameterSet&);
      ~L1TMuonGlobalParamsESProducer();

      typedef boost::shared_ptr<L1TMuonGlobalParams> ReturnType;

      ReturnType produce(const L1TMuonGlobalParamsRcd&);
   private:
      L1TMuonGlobalParams m_params;
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
L1TMuonGlobalParamsESProducer::L1TMuonGlobalParamsESProducer(const edm::ParameterSet& iConfig)
{
   //the following line is needed to tell the framework what
   // data is being produced
   setWhatProduced(this);

   L1TMuonGlobalParamsHelper m_params_helper;

   // Firmware version
   unsigned fwVersion = iConfig.getParameter<unsigned>("fwVersion");
   m_params_helper.setFwVersion(fwVersion);

   // uGMT disabled inputs
   bool disableCaloInputs = iConfig.getParameter<bool>("caloInputsDisable");
   if (disableCaloInputs) {
      m_params_helper.setCaloInputsToDisable(std::bitset<28>(0xFFFFFFF));
   } else {
      m_params_helper.setCaloInputsToDisable(std::bitset<28>());
   }

   std::vector<unsigned> bmtfInputsToDisable = iConfig.getParameter<std::vector<unsigned> >("bmtfInputsToDisable");
   std::bitset<12> bmtfDisables;
   for (size_t i = 0; i < bmtfInputsToDisable.size(); ++i) {
     bmtfDisables.set(i, bmtfInputsToDisable[i] > 0);
   }
   m_params_helper.setBmtfInputsToDisable(bmtfDisables);

   std::vector<unsigned> omtfInputsToDisable = iConfig.getParameter<std::vector<unsigned> >("omtfInputsToDisable");
   std::bitset<6> omtfpDisables;
   std::bitset<6> omtfnDisables;
   for (size_t i = 0; i < omtfInputsToDisable.size(); ++i) {
     if (i < 6) {
       omtfpDisables.set(i, omtfInputsToDisable[i] > 0);
     } else {
       omtfnDisables.set(i-6, omtfInputsToDisable[i] > 0);
     }
   }
   m_params_helper.setOmtfpInputsToDisable(omtfpDisables);
   m_params_helper.setOmtfnInputsToDisable(omtfnDisables);

   std::vector<unsigned> emtfInputsToDisable = iConfig.getParameter<std::vector<unsigned> >("emtfInputsToDisable");
   std::bitset<6> emtfpDisables;
   std::bitset<6> emtfnDisables;
   for (size_t i = 0; i < emtfInputsToDisable.size(); ++i) {
     if (i < 6) {
       emtfpDisables.set(i, emtfInputsToDisable[i] > 0);
     } else {
       emtfnDisables.set(i-6, emtfInputsToDisable[i] > 0);
     }
   }
   m_params_helper.setEmtfpInputsToDisable(emtfpDisables);
   m_params_helper.setEmtfnInputsToDisable(emtfnDisables);

   // masked inputs
   bool caloInputsMasked = iConfig.getParameter<bool>("caloInputsMasked");
   if (caloInputsMasked) {
      m_params_helper.setMaskedCaloInputs(std::bitset<28>(0xFFFFFFF));
   } else {
      m_params_helper.setMaskedCaloInputs(std::bitset<28>());
   }

   std::vector<unsigned> maskedBmtfInputs = iConfig.getParameter<std::vector<unsigned> >("maskedBmtfInputs");
   std::bitset<12> bmtfMasked;
   for (size_t i = 0; i < maskedBmtfInputs.size(); ++i) {
     bmtfMasked.set(i, maskedBmtfInputs[i] > 0);
   }
   m_params_helper.setMaskedBmtfInputs(bmtfMasked);

   std::vector<unsigned> maskedOmtfInputs = iConfig.getParameter<std::vector<unsigned> >("maskedOmtfInputs");
   std::bitset<6> omtfpMasked;
   std::bitset<6> omtfnMasked;
   for (size_t i = 0; i < maskedOmtfInputs.size(); ++i) {
     if (i < 6) {
       omtfpMasked.set(i, maskedOmtfInputs[i] > 0);
     } else {
       omtfnMasked.set(i-6, maskedOmtfInputs[i] > 0);
     }
   }
   m_params_helper.setMaskedOmtfpInputs(omtfpMasked);
   m_params_helper.setMaskedOmtfnInputs(omtfnMasked);

   std::vector<unsigned> maskedEmtfInputs = iConfig.getParameter<std::vector<unsigned> >("maskedEmtfInputs");
   std::bitset<6> emtfpMasked;
   std::bitset<6> emtfnMasked;
   for (size_t i = 0; i < maskedEmtfInputs.size(); ++i) {
     if (i < 6) {
       emtfpMasked.set(i, maskedEmtfInputs[i] > 0);
     } else {
       emtfnMasked.set(i-6, maskedEmtfInputs[i] > 0);
     }
   }
   m_params_helper.setMaskedEmtfpInputs(emtfpMasked);
   m_params_helper.setMaskedEmtfnInputs(emtfnMasked);

   // LUTs
   m_params_helper.setFwdPosSingleMatchQualLUTMaxDR(iConfig.getParameter<double>("FwdPosSingleMatchQualLUTMaxDR"),
                                                    iConfig.getParameter<double>("FwdPosSingleMatchQualLUTfEta"),
                                                    iConfig.getParameter<double>("FwdPosSingleMatchQualLUTfPhi"));
   m_params_helper.setFwdNegSingleMatchQualLUTMaxDR(iConfig.getParameter<double>("FwdNegSingleMatchQualLUTMaxDR"),
                                                    iConfig.getParameter<double>("FwdNegSingleMatchQualLUTfEta"),
                                                    iConfig.getParameter<double>("FwdNegSingleMatchQualLUTfPhi"));
   m_params_helper.setOvlPosSingleMatchQualLUTMaxDR(iConfig.getParameter<double>("OvlPosSingleMatchQualLUTMaxDR"),
                                                    iConfig.getParameter<double>("OvlPosSingleMatchQualLUTfEta"),
                                                    iConfig.getParameter<double>("OvlPosSingleMatchQualLUTfEtaCoarse"),
                                                    iConfig.getParameter<double>("OvlPosSingleMatchQualLUTfPhi"));
   m_params_helper.setOvlNegSingleMatchQualLUTMaxDR(iConfig.getParameter<double>("OvlNegSingleMatchQualLUTMaxDR"),
                                                    iConfig.getParameter<double>("OvlNegSingleMatchQualLUTfEta"),
                                                    iConfig.getParameter<double>("OvlNegSingleMatchQualLUTfEtaCoarse"),
                                                    iConfig.getParameter<double>("OvlNegSingleMatchQualLUTfPhi"));
   m_params_helper.setBOPosMatchQualLUTMaxDR(iConfig.getParameter<double>("BOPosMatchQualLUTMaxDR"),
                                             iConfig.getParameter<double>("BOPosMatchQualLUTfEta"),
                                             iConfig.getParameter<double>("BOPosMatchQualLUTfEtaCoarse"),
                                             iConfig.getParameter<double>("BOPosMatchQualLUTfPhi"));
   m_params_helper.setBONegMatchQualLUTMaxDR(iConfig.getParameter<double>("BONegMatchQualLUTMaxDR"),
                                             iConfig.getParameter<double>("BONegMatchQualLUTfEta"),
                                             iConfig.getParameter<double>("BONegMatchQualLUTfEtaCoarse"),
                                             iConfig.getParameter<double>("BONegMatchQualLUTfPhi"));
   m_params_helper.setFOPosMatchQualLUTMaxDR(iConfig.getParameter<double>("FOPosMatchQualLUTMaxDR"),
                                             iConfig.getParameter<double>("FOPosMatchQualLUTfEta"),
                                             iConfig.getParameter<double>("FOPosMatchQualLUTfEtaCoarse"),
                                             iConfig.getParameter<double>("FOPosMatchQualLUTfPhi"));
   m_params_helper.setFONegMatchQualLUTMaxDR(iConfig.getParameter<double>("FONegMatchQualLUTMaxDR"),
                                             iConfig.getParameter<double>("FONegMatchQualLUTfEta"),
                                             iConfig.getParameter<double>("FONegMatchQualLUTfEtaCoarse"),
                                             iConfig.getParameter<double>("FONegMatchQualLUTfPhi"));

   unsigned sortRankLUTPtFactor = iConfig.getParameter<unsigned>("SortRankLUTPtFactor");
   unsigned sortRankLUTQualFactor = iConfig.getParameter<unsigned>("SortRankLUTQualFactor");
   m_params_helper.setSortRankLUTFactors(sortRankLUTPtFactor, sortRankLUTQualFactor);

   auto absIsoCheckMemLUT = l1t::MicroGMTAbsoluteIsolationCheckLUTFactory::create (iConfig.getParameter<std::string>("AbsIsoCheckMemLUTPath"), fwVersion);
   auto relIsoCheckMemLUT = l1t::MicroGMTRelativeIsolationCheckLUTFactory::create (iConfig.getParameter<std::string>("RelIsoCheckMemLUTPath"), fwVersion);
   auto idxSelMemPhiLUT = l1t::MicroGMTCaloIndexSelectionLUTFactory::create (iConfig.getParameter<std::string>("IdxSelMemPhiLUTPath"), l1t::MicroGMTConfiguration::PHI, fwVersion);
   auto idxSelMemEtaLUT = l1t::MicroGMTCaloIndexSelectionLUTFactory::create (iConfig.getParameter<std::string>("IdxSelMemEtaLUTPath"), l1t::MicroGMTConfiguration::ETA, fwVersion);
   auto fwdPosSingleMatchQualLUT = l1t::MicroGMTMatchQualLUTFactory::create (iConfig.getParameter<std::string>("FwdPosSingleMatchQualLUTPath"),
                                                                             iConfig.getParameter<double>("FwdPosSingleMatchQualLUTMaxDR"),
                                                                             iConfig.getParameter<double>("FwdPosSingleMatchQualLUTfEta"),
                                                                             iConfig.getParameter<double>("FwdPosSingleMatchQualLUTfEta"), // set the coarse eta factor = fine eta factor
                                                                             iConfig.getParameter<double>("FwdPosSingleMatchQualLUTfPhi"),
                                                                             l1t::cancel_t::emtf_emtf_pos, fwVersion);
   auto fwdNegSingleMatchQualLUT = l1t::MicroGMTMatchQualLUTFactory::create (iConfig.getParameter<std::string>("FwdNegSingleMatchQualLUTPath"),
                                                                             iConfig.getParameter<double>("FwdNegSingleMatchQualLUTMaxDR"),
                                                                             iConfig.getParameter<double>("FwdNegSingleMatchQualLUTfEta"),
                                                                             iConfig.getParameter<double>("FwdNegSingleMatchQualLUTfEta"), // set the coarse eta factor = fine eta factor
                                                                             iConfig.getParameter<double>("FwdNegSingleMatchQualLUTfPhi"),
                                                                             l1t::cancel_t::emtf_emtf_neg, fwVersion);
   auto ovlPosSingleMatchQualLUT = l1t::MicroGMTMatchQualLUTFactory::create (iConfig.getParameter<std::string>("OvlPosSingleMatchQualLUTPath"),
                                                                             iConfig.getParameter<double>("OvlPosSingleMatchQualLUTMaxDR"),
                                                                             iConfig.getParameter<double>("OvlPosSingleMatchQualLUTfEta"),
                                                                             iConfig.getParameter<double>("OvlPosSingleMatchQualLUTfEtaCoarse"),
                                                                             iConfig.getParameter<double>("OvlPosSingleMatchQualLUTfPhi"),
                                                                             l1t::cancel_t::omtf_omtf_pos, fwVersion);
   auto ovlNegSingleMatchQualLUT = l1t::MicroGMTMatchQualLUTFactory::create (iConfig.getParameter<std::string>("OvlNegSingleMatchQualLUTPath"),
                                                                             iConfig.getParameter<double>("OvlNegSingleMatchQualLUTMaxDR"),
                                                                             iConfig.getParameter<double>("OvlNegSingleMatchQualLUTfEta"),
                                                                             iConfig.getParameter<double>("OvlNegSingleMatchQualLUTfEtaCoarse"),
                                                                             iConfig.getParameter<double>("OvlNegSingleMatchQualLUTfPhi"),
                                                                             l1t::cancel_t::omtf_omtf_neg, fwVersion);
   auto bOPosMatchQualLUT = l1t::MicroGMTMatchQualLUTFactory::create (iConfig.getParameter<std::string>("BOPosMatchQualLUTPath"),
                                                                      iConfig.getParameter<double>("BOPosMatchQualLUTMaxDR"),
                                                                      iConfig.getParameter<double>("BOPosMatchQualLUTfEta"),
                                                                      iConfig.getParameter<double>("BOPosMatchQualLUTfEtaCoarse"),
                                                                      iConfig.getParameter<double>("BOPosMatchQualLUTfPhi"),
                                                                      l1t::cancel_t::omtf_bmtf_pos, fwVersion);
   auto bONegMatchQualLUT = l1t::MicroGMTMatchQualLUTFactory::create (iConfig.getParameter<std::string>("BONegMatchQualLUTPath"),
                                                                      iConfig.getParameter<double>("BONegMatchQualLUTMaxDR"),
                                                                      iConfig.getParameter<double>("BONegMatchQualLUTfEta"),
                                                                      iConfig.getParameter<double>("BONegMatchQualLUTfEtaCoarse"),
                                                                      iConfig.getParameter<double>("BONegMatchQualLUTfPhi"),
                                                                      l1t::cancel_t::omtf_bmtf_neg, fwVersion);
   auto fOPosMatchQualLUT = l1t::MicroGMTMatchQualLUTFactory::create (iConfig.getParameter<std::string>("FOPosMatchQualLUTPath"),
                                                                      iConfig.getParameter<double>("FOPosMatchQualLUTMaxDR"),
                                                                      iConfig.getParameter<double>("FOPosMatchQualLUTfEta"),
                                                                      iConfig.getParameter<double>("FOPosMatchQualLUTfEtaCoarse"),
                                                                      iConfig.getParameter<double>("FOPosMatchQualLUTfPhi"),
                                                                      l1t::cancel_t::omtf_emtf_pos, fwVersion);
   auto fONegMatchQualLUT = l1t::MicroGMTMatchQualLUTFactory::create (iConfig.getParameter<std::string>("FONegMatchQualLUTPath"),
                                                                      iConfig.getParameter<double>("FONegMatchQualLUTMaxDR"),
                                                                      iConfig.getParameter<double>("FONegMatchQualLUTfEta"),
                                                                      iConfig.getParameter<double>("FONegMatchQualLUTfEtaCoarse"),
                                                                      iConfig.getParameter<double>("FONegMatchQualLUTfPhi"),
                                                                      l1t::cancel_t::omtf_emtf_neg, fwVersion);
   auto bPhiExtrapolationLUT = l1t::MicroGMTExtrapolationLUTFactory::create (iConfig.getParameter<std::string>("BPhiExtrapolationLUTPath"), l1t::MicroGMTConfiguration::PHI_OUT, fwVersion);
   auto oPhiExtrapolationLUT = l1t::MicroGMTExtrapolationLUTFactory::create (iConfig.getParameter<std::string>("OPhiExtrapolationLUTPath"), l1t::MicroGMTConfiguration::PHI_OUT, fwVersion);
   auto fPhiExtrapolationLUT = l1t::MicroGMTExtrapolationLUTFactory::create (iConfig.getParameter<std::string>("FPhiExtrapolationLUTPath"), l1t::MicroGMTConfiguration::PHI_OUT, fwVersion);
   auto bEtaExtrapolationLUT = l1t::MicroGMTExtrapolationLUTFactory::create (iConfig.getParameter<std::string>("BEtaExtrapolationLUTPath"), l1t::MicroGMTConfiguration::ETA_OUT, fwVersion);
   auto oEtaExtrapolationLUT = l1t::MicroGMTExtrapolationLUTFactory::create (iConfig.getParameter<std::string>("OEtaExtrapolationLUTPath"), l1t::MicroGMTConfiguration::ETA_OUT, fwVersion);
   auto fEtaExtrapolationLUT = l1t::MicroGMTExtrapolationLUTFactory::create (iConfig.getParameter<std::string>("FEtaExtrapolationLUTPath"), l1t::MicroGMTConfiguration::ETA_OUT, fwVersion);
   auto rankPtQualityLUT = l1t::MicroGMTRankPtQualLUTFactory::create (iConfig.getParameter<std::string>("SortRankLUTPath"), fwVersion, sortRankLUTPtFactor, sortRankLUTQualFactor);
   m_params_helper.setAbsIsoCheckMemLUT(*absIsoCheckMemLUT);
   m_params_helper.setRelIsoCheckMemLUT(*relIsoCheckMemLUT);
   m_params_helper.setIdxSelMemPhiLUT(*idxSelMemPhiLUT);
   m_params_helper.setIdxSelMemEtaLUT(*idxSelMemEtaLUT);
   m_params_helper.setFwdPosSingleMatchQualLUT(*fwdPosSingleMatchQualLUT);
   m_params_helper.setFwdNegSingleMatchQualLUT(*fwdNegSingleMatchQualLUT);
   m_params_helper.setOvlPosSingleMatchQualLUT(*ovlPosSingleMatchQualLUT);
   m_params_helper.setOvlNegSingleMatchQualLUT(*ovlNegSingleMatchQualLUT);
   m_params_helper.setBOPosMatchQualLUT(*bOPosMatchQualLUT);
   m_params_helper.setBONegMatchQualLUT(*bONegMatchQualLUT);
   m_params_helper.setFOPosMatchQualLUT(*fOPosMatchQualLUT);
   m_params_helper.setFONegMatchQualLUT(*fONegMatchQualLUT);
   m_params_helper.setBPhiExtrapolationLUT(*bPhiExtrapolationLUT);
   m_params_helper.setOPhiExtrapolationLUT(*oPhiExtrapolationLUT);
   m_params_helper.setFPhiExtrapolationLUT(*fPhiExtrapolationLUT);
   m_params_helper.setBEtaExtrapolationLUT(*bEtaExtrapolationLUT);
   m_params_helper.setOEtaExtrapolationLUT(*oEtaExtrapolationLUT);
   m_params_helper.setFEtaExtrapolationLUT(*fEtaExtrapolationLUT);
   m_params_helper.setSortRankLUT(*rankPtQualityLUT);

   // LUT paths
   m_params_helper.setAbsIsoCheckMemLUTPath        (iConfig.getParameter<std::string>("AbsIsoCheckMemLUTPath"));
   m_params_helper.setRelIsoCheckMemLUTPath        (iConfig.getParameter<std::string>("RelIsoCheckMemLUTPath"));
   m_params_helper.setIdxSelMemPhiLUTPath          (iConfig.getParameter<std::string>("IdxSelMemPhiLUTPath"));
   m_params_helper.setIdxSelMemEtaLUTPath          (iConfig.getParameter<std::string>("IdxSelMemEtaLUTPath"));
   m_params_helper.setFwdPosSingleMatchQualLUTPath (iConfig.getParameter<std::string>("FwdPosSingleMatchQualLUTPath"));
   m_params_helper.setFwdNegSingleMatchQualLUTPath (iConfig.getParameter<std::string>("FwdNegSingleMatchQualLUTPath"));
   m_params_helper.setOvlPosSingleMatchQualLUTPath (iConfig.getParameter<std::string>("OvlPosSingleMatchQualLUTPath"));
   m_params_helper.setOvlNegSingleMatchQualLUTPath (iConfig.getParameter<std::string>("OvlNegSingleMatchQualLUTPath"));
   m_params_helper.setBOPosMatchQualLUTPath        (iConfig.getParameter<std::string>("BOPosMatchQualLUTPath"));
   m_params_helper.setBONegMatchQualLUTPath        (iConfig.getParameter<std::string>("BONegMatchQualLUTPath"));
   m_params_helper.setFOPosMatchQualLUTPath        (iConfig.getParameter<std::string>("FOPosMatchQualLUTPath"));
   m_params_helper.setFONegMatchQualLUTPath        (iConfig.getParameter<std::string>("FONegMatchQualLUTPath"));
   m_params_helper.setBPhiExtrapolationLUTPath     (iConfig.getParameter<std::string>("BPhiExtrapolationLUTPath"));
   m_params_helper.setOPhiExtrapolationLUTPath     (iConfig.getParameter<std::string>("OPhiExtrapolationLUTPath"));
   m_params_helper.setFPhiExtrapolationLUTPath     (iConfig.getParameter<std::string>("FPhiExtrapolationLUTPath"));
   m_params_helper.setBEtaExtrapolationLUTPath     (iConfig.getParameter<std::string>("BEtaExtrapolationLUTPath"));
   m_params_helper.setOEtaExtrapolationLUTPath     (iConfig.getParameter<std::string>("OEtaExtrapolationLUTPath"));
   m_params_helper.setFEtaExtrapolationLUTPath     (iConfig.getParameter<std::string>("FEtaExtrapolationLUTPath"));
   m_params_helper.setSortRankLUTPath              (iConfig.getParameter<std::string>("SortRankLUTPath"));

   // temp hack to avoid ALCA/DB signoff:
   m_params = cast_to_L1TMuonGlobalParams((L1TMuonGlobalParams_PUBLIC)m_params_helper);
}


L1TMuonGlobalParamsESProducer::~L1TMuonGlobalParamsESProducer()
{
}


//
// member functions
//

// ------------ method called to produce the data  ------------
L1TMuonGlobalParamsESProducer::ReturnType
L1TMuonGlobalParamsESProducer::produce(const L1TMuonGlobalParamsRcd& iRecord)
{
   using namespace edm::es;
   boost::shared_ptr<L1TMuonGlobalParams> pMicroGMTParams;

   pMicroGMTParams = boost::shared_ptr<L1TMuonGlobalParams>(new L1TMuonGlobalParams(m_params));
   return pMicroGMTParams;
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(L1TMuonGlobalParamsESProducer);
