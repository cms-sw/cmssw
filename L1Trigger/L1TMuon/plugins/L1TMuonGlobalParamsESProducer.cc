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

   // Firmware version
   unsigned fwVersion = iConfig.getParameter<unsigned>("fwVersion");

   m_params.setFwVersion(fwVersion);

   int bxMin = iConfig.getParameter<int>("bxMin");
   int bxMax = iConfig.getParameter<int>("bxMax");
   if (bxMin > bxMax) {
      m_params.setBxMin(bxMax);
      m_params.setBxMax(bxMin);
   } else {
      m_params.setBxMin(bxMin);
      m_params.setBxMax(bxMax);
   }

   //m_params.setBrlSingleMatchQualLUTMaxDR(iConfig.getParameter<double>("BrlSingleMatchQualLUTMaxDR"));
   m_params.setFwdPosSingleMatchQualLUTMaxDR(iConfig.getParameter<double>("FwdPosSingleMatchQualLUTMaxDR"));
   m_params.setFwdNegSingleMatchQualLUTMaxDR(iConfig.getParameter<double>("FwdNegSingleMatchQualLUTMaxDR"));
   m_params.setOvlPosSingleMatchQualLUTMaxDR(iConfig.getParameter<double>("OvlPosSingleMatchQualLUTMaxDR"));
   m_params.setOvlNegSingleMatchQualLUTMaxDR(iConfig.getParameter<double>("OvlNegSingleMatchQualLUTMaxDR"));
   m_params.setBOPosMatchQualLUTMaxDR(iConfig.getParameter<double>("BOPosMatchQualLUTMaxDR"), iConfig.getParameter<double>("BOPosMatchQualLUTMaxDREtaFine"));
   m_params.setBONegMatchQualLUTMaxDR(iConfig.getParameter<double>("BONegMatchQualLUTMaxDR"), iConfig.getParameter<double>("BONegMatchQualLUTMaxDREtaFine"));
   m_params.setFOPosMatchQualLUTMaxDR(iConfig.getParameter<double>("FOPosMatchQualLUTMaxDR"));
   m_params.setFONegMatchQualLUTMaxDR(iConfig.getParameter<double>("FONegMatchQualLUTMaxDR"));

   unsigned sortRankLUTPtFactor = iConfig.getParameter<unsigned>("SortRankLUTPtFactor");
   unsigned sortRankLUTQualFactor = iConfig.getParameter<unsigned>("SortRankLUTQualFactor");
   m_params.setSortRankLUTFactors(sortRankLUTPtFactor, sortRankLUTQualFactor);

   auto absIsoCheckMemLUT = l1t::MicroGMTAbsoluteIsolationCheckLUTFactory::create (iConfig.getParameter<std::string>("AbsIsoCheckMemLUTPath"), fwVersion);
   auto relIsoCheckMemLUT = l1t::MicroGMTRelativeIsolationCheckLUTFactory::create (iConfig.getParameter<std::string>("RelIsoCheckMemLUTPath"), fwVersion);
   auto idxSelMemPhiLUT = l1t::MicroGMTCaloIndexSelectionLUTFactory::create (iConfig.getParameter<std::string>("IdxSelMemPhiLUTPath"), l1t::MicroGMTConfiguration::PHI, fwVersion);
   auto idxSelMemEtaLUT = l1t::MicroGMTCaloIndexSelectionLUTFactory::create (iConfig.getParameter<std::string>("IdxSelMemEtaLUTPath"), l1t::MicroGMTConfiguration::ETA, fwVersion);
   //auto brlSingleMatchQualLUT = l1t::MicroGMTMatchQualLUTFactory::create (iConfig.getParameter<std::string>("BrlSingleMatchQualLUTPath"), iConfig.getParameter<double>("BrlSingleMatchQualLUTMaxDR"), l1t::cancel_t::bmtf_bmtf, fwVersion);
   auto fwdPosSingleMatchQualLUT = l1t::MicroGMTMatchQualLUTFactory::create (iConfig.getParameter<std::string>("FwdPosSingleMatchQualLUTPath"), iConfig.getParameter<double>("FwdPosSingleMatchQualLUTMaxDR"), l1t::cancel_t::emtf_emtf_pos, fwVersion);
   auto fwdNegSingleMatchQualLUT = l1t::MicroGMTMatchQualLUTFactory::create (iConfig.getParameter<std::string>("FwdNegSingleMatchQualLUTPath"), iConfig.getParameter<double>("FwdNegSingleMatchQualLUTMaxDR"), l1t::cancel_t::emtf_emtf_neg, fwVersion);
   auto ovlPosSingleMatchQualLUT = l1t::MicroGMTMatchQualLUTFactory::create (iConfig.getParameter<std::string>("OvlPosSingleMatchQualLUTPath"), iConfig.getParameter<double>("OvlPosSingleMatchQualLUTMaxDR"), l1t::cancel_t::omtf_omtf_pos, fwVersion);
   auto ovlNegSingleMatchQualLUT = l1t::MicroGMTMatchQualLUTFactory::create (iConfig.getParameter<std::string>("OvlNegSingleMatchQualLUTPath"), iConfig.getParameter<double>("OvlNegSingleMatchQualLUTMaxDR"), l1t::cancel_t::omtf_omtf_neg, fwVersion);
   auto bOPosMatchQualLUT = l1t::MicroGMTMatchQualLUTFactory::create (iConfig.getParameter<std::string>("BOPosMatchQualLUTPath"), iConfig.getParameter<double>("BOPosMatchQualLUTMaxDR"), l1t::cancel_t::omtf_bmtf_pos, fwVersion);
   auto bONegMatchQualLUT = l1t::MicroGMTMatchQualLUTFactory::create (iConfig.getParameter<std::string>("BONegMatchQualLUTPath"), iConfig.getParameter<double>("BONegMatchQualLUTMaxDR"), l1t::cancel_t::omtf_bmtf_neg, fwVersion);
   auto fOPosMatchQualLUT = l1t::MicroGMTMatchQualLUTFactory::create (iConfig.getParameter<std::string>("FOPosMatchQualLUTPath"), iConfig.getParameter<double>("FOPosMatchQualLUTMaxDR"), l1t::cancel_t::omtf_emtf_pos, fwVersion);
   auto fONegMatchQualLUT = l1t::MicroGMTMatchQualLUTFactory::create (iConfig.getParameter<std::string>("FONegMatchQualLUTPath"), iConfig.getParameter<double>("FONegMatchQualLUTMaxDR"), l1t::cancel_t::omtf_emtf_neg, fwVersion);
   auto bPhiExtrapolationLUT = l1t::MicroGMTExtrapolationLUTFactory::create (iConfig.getParameter<std::string>("BPhiExtrapolationLUTPath"), l1t::MicroGMTConfiguration::PHI_OUT, fwVersion);
   auto oPhiExtrapolationLUT = l1t::MicroGMTExtrapolationLUTFactory::create (iConfig.getParameter<std::string>("OPhiExtrapolationLUTPath"), l1t::MicroGMTConfiguration::PHI_OUT, fwVersion);
   auto fPhiExtrapolationLUT = l1t::MicroGMTExtrapolationLUTFactory::create (iConfig.getParameter<std::string>("FPhiExtrapolationLUTPath"), l1t::MicroGMTConfiguration::PHI_OUT, fwVersion);
   auto bEtaExtrapolationLUT = l1t::MicroGMTExtrapolationLUTFactory::create (iConfig.getParameter<std::string>("BEtaExtrapolationLUTPath"), l1t::MicroGMTConfiguration::ETA_OUT, fwVersion);
   auto oEtaExtrapolationLUT = l1t::MicroGMTExtrapolationLUTFactory::create (iConfig.getParameter<std::string>("OEtaExtrapolationLUTPath"), l1t::MicroGMTConfiguration::ETA_OUT, fwVersion);
   auto fEtaExtrapolationLUT = l1t::MicroGMTExtrapolationLUTFactory::create (iConfig.getParameter<std::string>("FEtaExtrapolationLUTPath"), l1t::MicroGMTConfiguration::ETA_OUT, fwVersion);
   auto rankPtQualityLUT = l1t::MicroGMTRankPtQualLUTFactory::create (iConfig.getParameter<std::string>("SortRankLUTPath"), fwVersion, sortRankLUTPtFactor, sortRankLUTQualFactor);
   m_params.setAbsIsoCheckMemLUT(*absIsoCheckMemLUT);
   m_params.setRelIsoCheckMemLUT(*relIsoCheckMemLUT);
   m_params.setIdxSelMemPhiLUT(*idxSelMemPhiLUT);
   m_params.setIdxSelMemEtaLUT(*idxSelMemEtaLUT);
   //m_params.setBrlSingleMatchQualLUT(*brlSingleMatchQualLUT);
   m_params.setFwdPosSingleMatchQualLUT(*fwdPosSingleMatchQualLUT);
   m_params.setFwdNegSingleMatchQualLUT(*fwdNegSingleMatchQualLUT);
   m_params.setOvlPosSingleMatchQualLUT(*ovlPosSingleMatchQualLUT);
   m_params.setOvlNegSingleMatchQualLUT(*ovlNegSingleMatchQualLUT);
   m_params.setBOPosMatchQualLUT(*bOPosMatchQualLUT);
   m_params.setBONegMatchQualLUT(*bONegMatchQualLUT);
   m_params.setFOPosMatchQualLUT(*fOPosMatchQualLUT);
   m_params.setFONegMatchQualLUT(*fONegMatchQualLUT);
   m_params.setBPhiExtrapolationLUT(*bPhiExtrapolationLUT);
   m_params.setOPhiExtrapolationLUT(*oPhiExtrapolationLUT);
   m_params.setFPhiExtrapolationLUT(*fPhiExtrapolationLUT);
   m_params.setBEtaExtrapolationLUT(*bEtaExtrapolationLUT);
   m_params.setOEtaExtrapolationLUT(*oEtaExtrapolationLUT);
   m_params.setFEtaExtrapolationLUT(*fEtaExtrapolationLUT);
   m_params.setSortRankLUT(*rankPtQualityLUT);

   // LUT paths
   m_params.setAbsIsoCheckMemLUTPath        (iConfig.getParameter<std::string>("AbsIsoCheckMemLUTPath"));
   m_params.setRelIsoCheckMemLUTPath        (iConfig.getParameter<std::string>("RelIsoCheckMemLUTPath"));
   m_params.setIdxSelMemPhiLUTPath          (iConfig.getParameter<std::string>("IdxSelMemPhiLUTPath"));
   m_params.setIdxSelMemEtaLUTPath          (iConfig.getParameter<std::string>("IdxSelMemEtaLUTPath"));
   //m_params.setBrlSingleMatchQualLUTPath    (iConfig.getParameter<std::string>("BrlSingleMatchQualLUTPath"));
   m_params.setFwdPosSingleMatchQualLUTPath (iConfig.getParameter<std::string>("FwdPosSingleMatchQualLUTPath"));
   m_params.setFwdNegSingleMatchQualLUTPath (iConfig.getParameter<std::string>("FwdNegSingleMatchQualLUTPath"));
   m_params.setOvlPosSingleMatchQualLUTPath (iConfig.getParameter<std::string>("OvlPosSingleMatchQualLUTPath"));
   m_params.setOvlNegSingleMatchQualLUTPath (iConfig.getParameter<std::string>("OvlNegSingleMatchQualLUTPath"));
   m_params.setBOPosMatchQualLUTPath        (iConfig.getParameter<std::string>("BOPosMatchQualLUTPath"));
   m_params.setBONegMatchQualLUTPath        (iConfig.getParameter<std::string>("BONegMatchQualLUTPath"));
   m_params.setFOPosMatchQualLUTPath        (iConfig.getParameter<std::string>("FOPosMatchQualLUTPath"));
   m_params.setFONegMatchQualLUTPath        (iConfig.getParameter<std::string>("FONegMatchQualLUTPath"));
   m_params.setBPhiExtrapolationLUTPath     (iConfig.getParameter<std::string>("BPhiExtrapolationLUTPath"));
   m_params.setOPhiExtrapolationLUTPath     (iConfig.getParameter<std::string>("OPhiExtrapolationLUTPath"));
   m_params.setFPhiExtrapolationLUTPath     (iConfig.getParameter<std::string>("FPhiExtrapolationLUTPath"));
   m_params.setBEtaExtrapolationLUTPath     (iConfig.getParameter<std::string>("BEtaExtrapolationLUTPath"));
   m_params.setOEtaExtrapolationLUTPath     (iConfig.getParameter<std::string>("OEtaExtrapolationLUTPath"));
   m_params.setFEtaExtrapolationLUTPath     (iConfig.getParameter<std::string>("FEtaExtrapolationLUTPath"));
   m_params.setSortRankLUTPath              (iConfig.getParameter<std::string>("SortRankLUTPath"));
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
