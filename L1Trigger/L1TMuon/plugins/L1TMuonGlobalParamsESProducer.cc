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

   auto absIsoCheckMemLUT = l1t::MicroGMTAbsoluteIsolationCheckLUTFactory::create (iConfig.getParameter<std::string>("AbsIsoCheckMemLUTPath"), fwVersion);
   auto relIsoCheckMemLUT = l1t::MicroGMTRelativeIsolationCheckLUTFactory::create (iConfig.getParameter<std::string>("RelIsoCheckMemLUTPath"), fwVersion);
   auto idxSelMemPhiLUT = l1t::MicroGMTCaloIndexSelectionLUTFactory::create (iConfig.getParameter<std::string>("IdxSelMemPhiLUTPath"), 1, fwVersion);
   auto idxSelMemEtaLUT = l1t::MicroGMTCaloIndexSelectionLUTFactory::create (iConfig.getParameter<std::string>("IdxSelMemEtaLUTPath"), 0, fwVersion);
   auto brlSingleMatchQualLUT = l1t::MicroGMTMatchQualLUTFactory::create (iConfig.getParameter<std::string>("BrlSingleMatchQualLUTPath"), l1t::cancel_t::bmtf_bmtf, fwVersion);
   auto fwdPosSingleMatchQualLUT = l1t::MicroGMTMatchQualLUTFactory::create (iConfig.getParameter<std::string>("FwdPosSingleMatchQualLUTPath"), l1t::cancel_t::emtf_emtf_pos, fwVersion);
   auto fwdNegSingleMatchQualLUT = l1t::MicroGMTMatchQualLUTFactory::create (iConfig.getParameter<std::string>("FwdNegSingleMatchQualLUTPath"), l1t::cancel_t::emtf_emtf_neg, fwVersion);
   auto ovlPosSingleMatchQualLUT = l1t::MicroGMTMatchQualLUTFactory::create (iConfig.getParameter<std::string>("OvlPosSingleMatchQualLUTPath"), l1t::cancel_t::omtf_omtf_pos, fwVersion);
   auto ovlNegSingleMatchQualLUT = l1t::MicroGMTMatchQualLUTFactory::create (iConfig.getParameter<std::string>("OvlNegSingleMatchQualLUTPath"), l1t::cancel_t::omtf_omtf_neg, fwVersion);
   auto bOPosMatchQualLUT = l1t::MicroGMTMatchQualLUTFactory::create (iConfig.getParameter<std::string>("BOPosMatchQualLUTPath"), l1t::cancel_t::omtf_bmtf_pos, fwVersion);
   auto bONegMatchQualLUT = l1t::MicroGMTMatchQualLUTFactory::create (iConfig.getParameter<std::string>("BONegMatchQualLUTPath"), l1t::cancel_t::omtf_bmtf_neg, fwVersion);
   auto fOPosMatchQualLUT = l1t::MicroGMTMatchQualLUTFactory::create (iConfig.getParameter<std::string>("FOPosMatchQualLUTPath"), l1t::cancel_t::omtf_emtf_pos, fwVersion);
   auto fONegMatchQualLUT = l1t::MicroGMTMatchQualLUTFactory::create (iConfig.getParameter<std::string>("FONegMatchQualLUTPath"), l1t::cancel_t::omtf_emtf_neg, fwVersion);
   auto bPhiExtrapolationLUT = l1t::MicroGMTExtrapolationLUTFactory::create (iConfig.getParameter<std::string>("BPhiExtrapolationLUTPath"), fwVersion);
   auto oPhiExtrapolationLUT = l1t::MicroGMTExtrapolationLUTFactory::create (iConfig.getParameter<std::string>("OPhiExtrapolationLUTPath"), fwVersion);
   auto fPhiExtrapolationLUT = l1t::MicroGMTExtrapolationLUTFactory::create (iConfig.getParameter<std::string>("FPhiExtrapolationLUTPath"), fwVersion);
   auto bEtaExtrapolationLUT = l1t::MicroGMTExtrapolationLUTFactory::create (iConfig.getParameter<std::string>("BEtaExtrapolationLUTPath"), fwVersion);
   auto oEtaExtrapolationLUT = l1t::MicroGMTExtrapolationLUTFactory::create (iConfig.getParameter<std::string>("OEtaExtrapolationLUTPath"), fwVersion);
   auto fEtaExtrapolationLUT = l1t::MicroGMTExtrapolationLUTFactory::create (iConfig.getParameter<std::string>("FEtaExtrapolationLUTPath"), fwVersion);
   auto rankPtQualityLUT = l1t::MicroGMTRankPtQualLUTFactory::create (iConfig.getParameter<std::string>("SortRankLUTPath"), fwVersion);
   m_params.setAbsIsoCheckMemLUT(*absIsoCheckMemLUT);
   m_params.setRelIsoCheckMemLUT(*relIsoCheckMemLUT);
   m_params.setIdxSelMemPhiLUT(*idxSelMemPhiLUT);
   m_params.setIdxSelMemEtaLUT(*idxSelMemEtaLUT);
   m_params.setBrlSingleMatchQualLUT(*brlSingleMatchQualLUT);
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
   m_params.setBrlSingleMatchQualLUTPath    (iConfig.getParameter<std::string>("BrlSingleMatchQualLUTPath"));
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
