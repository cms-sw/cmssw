// -*- C++ -*-
//
// Package:    RecoLocalCalo/HcalRecProducers
// Class:      HBHEPhase1Reconstructor
// 
/**\class HBHEPhase1Reconstructor HBHEPhase1Reconstructor.cc RecoLocalCalo/HcalRecProducers/plugins/HBHEPhase1Reconstructor.cc

 Description: Phase 1 reconstruction module for HB/HE

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Igor Volobouev
//         Created:  Tue, 21 Jun 2016 00:56:40 GMT
//
//


// system include files
#include <memory>
#include <utility>
#include <algorithm>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"

#include "CalibFormats/CaloObjects/interface/CaloSamples.h"

#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSeverityLevelComputer.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSeverityLevelComputerRcd.h"

// Base class for Phase 1 HB/HE reco algorithms configuration objects
#include "CondFormats/HcalObjects/interface/AbsHFPhase1AlgoData.h"

// Parser for Phase 1 HB/HE reco algorithms
#include "RecoLocalCalo/HcalRecAlgos/interface/parseHBHEPhase1AlgoDescription.h"

// Some helper functions
namespace {
    // Class Data must inherit from AbsHFPhase1AlgoData
    // and must have a copy constructor. This function
    // returns an object allocated on the heap.
    template <class Data, class Record>
    Data* fetchHBHEPhase1AlgoDataHelper(const edm::EventSetup& es)
    {
        edm::ESHandle<Data> p;
        es.get<Record>().get(p);
        return new Data(*p.product());
    }

    // Factory function for fetching (from EventSetup) objects
    // of the types inheriting from AbsHFPhase1AlgoData
    std::unique_ptr<AbsHFPhase1AlgoData>
    fetchHBHEPhase1AlgoData(const std::string& className, const edm::EventSetup& es)
    {
        AbsHFPhase1AlgoData* data = 0;
        // Compare with possibe class names
        // if (className == "MyHFPhase1AlgoData")
        //     data = fetchHBHEPhase1AlgoDataHelper<MyHFPhase1AlgoData, MyHFPhase1AlgoDataRcd>(es);
        // else if (className == "OtherHFPhase1AlgoData")
        //     ...;
        return std::unique_ptr<AbsHFPhase1AlgoData>(data);
    }

    // The following function should apply the SiPM nonlinearity
    // correction. It may need extra arguments for that. It should
    // return the best estimate of the charge before pedestal subtraction.
    double getRawChargeFromSample(const QIE11DataFrame::Sample& s,
                                  const double decodedCharge,
                                  const HcalCalibrations& calib)
    {
        // FIX THIS!!!
        return decodedCharge;
    }

    double getRawChargeFromSample(const HcalQIESample& s,
                                  const double decodedCharge,
                                  const HcalCalibrations& calib)
    {
        return decodedCharge;
    }

    float getTDCTimeFromSample(const QIE11DataFrame::Sample& s)
    {
        // Conversion from TDC to ns for the QIE11 chip
        static const float qie11_tdc_to_ns = 0.5f;

        // TDC values produced in case the pulse is always above/below
        // the discriminator
        static const int qie11_tdc_code_overshoot = 62;
        static const int qie11_tdc_code_undershoot = 63;

        const int tdc = s.tdc();
        float t = qie11_tdc_to_ns*tdc;
        if (tdc == qie11_tdc_code_overshoot)
            t = HcalSpecialTimes::UNKNOWN_T_OVERSHOOT;
        else if (tdc == qie11_tdc_code_undershoot)
            t = HcalSpecialTimes::UNKNOWN_T_UNDERSHOOT;
        return t;
    }

    float getTDCTimeFromSample(const HcalQIESample&)
    {
        return HcalSpecialTimes::UNKNOWN_T_NOTDC;
    }

    // The first element of the pair indicates presence of optical
    // link errors. The second indicated presence of capid errors.
    std::pair<bool,bool> findHWErrors(const HBHEDataFrame& df,
                                      const unsigned len)
    {
        bool linkErr = false;
        bool capidErr = false;
        if (len)
        {
            int expectedCapid = df[0].capid();
            for (unsigned i=0; i<len; ++i)
            {
                if (df[i].er())
                    linkErr = true;
                if (df[i].capid() != expectedCapid)
                    capidErr = true;
                expectedCapid = (expectedCapid + 1) % 4;
            }
        }
        return std::pair<bool,bool>(linkErr, capidErr);
    }

    std::pair<bool,bool> findHWErrors(const QIE11DataFrame& df,
                                      const unsigned /* len */)
    {
        return std::pair<bool,bool>(df.linkError(), df.capidError());
    }
}


//
// class declaration
//
class HBHEPhase1Reconstructor : public edm::stream::EDProducer<>
{
public:
    explicit HBHEPhase1Reconstructor(const edm::ParameterSet&);
    ~HBHEPhase1Reconstructor();

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
    virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
    virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
    virtual void produce(edm::Event&, const edm::EventSetup&) override;

    // Configuration parameters
    std::string algoConfigClass_;
    bool processQIE8_;
    bool processQIE11_;
    bool saveInfos_;
    bool saveDroppedInfos_;
    bool makeRecHits_;
    bool dropZSmarkedPassed_;
    bool tsFromDB_;
    bool recoParamsFromDB_;

    // Other members
    edm::EDGetTokenT<HBHEDigiCollection> tok_qie8_;
    edm::EDGetTokenT<QIE11DigiCollection> tok_qie11_;
    std::unique_ptr<AbsHBHEPhase1Algo> reco_;
    std::unique_ptr<AbsHFPhase1AlgoData> recoConfig_;
    std::unique_ptr<HcalRecoParams> paramTS_;

    // Status bit setters
    // ... Not available yet ...

    // For the function below, arguments "infoColl" and/or "rechits"
    // are allowed to be null.
    template<class DataFrame, class Collection>
    void processData(const Collection& coll,
                     const HcalDbService& cond,
                     const HcalChannelQuality& qual,
                     const HcalSeverityLevelComputer& severity,
                     const bool isRealData,
                     HBHEChannelInfo* info,
                     HBHEChannelInfoCollection* infoColl,
                     HBHERecHitCollection* rechits);

    // Methods for setting rechit status bits
    void setAsicSpecificBits(const HBHEDataFrame& frame,
                             const HBHEChannelInfo& info, HBHERecHit* rh);
    void setAsicSpecificBits(const QIE11DataFrame& frame,
                             const HBHEChannelInfo& info, HBHERecHit* rh);
    void setCommonStatusBits(const HBHEChannelInfo& info, HBHERecHit* rh);
};

//
// constructors and destructor
//
HBHEPhase1Reconstructor::HBHEPhase1Reconstructor(const edm::ParameterSet& conf)
    : algoConfigClass_(conf.getParameter<std::string>("algoConfigClass")),
      processQIE8_(conf.getParameter<bool>("processQIE8")),
      processQIE11_(conf.getParameter<bool>("processQIE11")),
      saveInfos_(conf.getParameter<bool>("saveInfos")),
      saveDroppedInfos_(conf.getParameter<bool>("saveDroppedInfos")),
      makeRecHits_(conf.getParameter<bool>("makeRecHits")),
      dropZSmarkedPassed_(conf.getParameter<bool>("dropZSmarkedPassed")),
      tsFromDB_(conf.getParameter<bool>("tsFromDB")),
      recoParamsFromDB_(conf.getParameter<bool>("recoParamsFromDB")),
      reco_(parseHBHEPhase1AlgoDescription(conf.getParameter<edm::ParameterSet>("algorithm")))
{
    // Check that the reco algorithm has been successfully configured
    if (!reco_.get())
        throw cms::Exception("HBHEPhase1BadConfig")
            << "Invalid HBHEPhase1Algo algorithm configuration"
            << std::endl;

    if (processQIE8_)
        tok_qie8_ = consumes<HBHEDigiCollection>(
            conf.getParameter<edm::InputTag>("digiLabelQIE8"));

    if (processQIE11_)
        tok_qie11_ = consumes<QIE11DigiCollection>(
            conf.getParameter<edm::InputTag>("digiLabelQIE11"));

    if (saveInfos_)
        produces<HBHEChannelInfoCollection>();

    if (makeRecHits_)
        produces<HBHERecHitCollection>();
}


HBHEPhase1Reconstructor::~HBHEPhase1Reconstructor()
{
   // do anything here that needs to be done at destruction time
   // (e.g. close files, deallocate resources etc.)
}


//
// member functions
//
template<class DFrame, class Collection>
void HBHEPhase1Reconstructor::processData(const Collection& coll,
                                          const HcalDbService& cond,
                                          const HcalChannelQuality& qual,
                                          const HcalSeverityLevelComputer& severity,
                                          const bool isRealData,
                                          HBHEChannelInfo* channelInfo,
                                          HBHEChannelInfoCollection* infos,
                                          HBHERecHitCollection* rechits)
{
    // If "saveDroppedInfos_" flag is set, fill the info with something
    // meaningful even if the database tells us to drop this channel.
    // Note that this flag affects only "infos", the rechits are still
    // not going to be constructed from such channels.
    const bool skipDroppedChannels = !(infos && saveDroppedInfos_);

    // Iterate over the input collection
    for (typename Collection::const_iterator it = coll.begin();
         it != coll.end(); ++it)
    {
        const DFrame& frame(*it);
        const HcalDetId cell(frame.id());
        const HcalRecoParam* param_ts = nullptr;
        if (tsFromDB_ || recoParamsFromDB_)
            param_ts = paramTS_->getValues(cell.rawId());

        // Check if the database tells us to drop this channel
        const HcalChannelStatus* mydigistatus = qual.getValues(cell.rawId());
        const bool taggedBadByDb = severity.dropChannel(mydigistatus->getValue());
        if (taggedBadByDb && skipDroppedChannels)
            continue;

        // Check if the channel is zero suppressed
        bool dropByZS = false;
        if (dropZSmarkedPassed_)
            if (frame.zsMarkAndPass())
                dropByZS = true;
        if (dropByZS && skipDroppedChannels)
            continue;

        // Basic ADC decoding tools
        const HcalCalibrations& calib = cond.getHcalCalibrations(cell);
        const HcalQIECoder* channelCoder = cond.getHcalCoder(cell);
        const HcalQIEShape* shape = cond.getHcalShape(channelCoder);
        HcalCoderDb coder(*channelCoder, *shape);

        // ADC to fC conversion
        CaloSamples cs;
        coder.adc2fC(frame, cs);

        // Prepare to iterate over time slices
        const int nRead = cs.size();
        const int maxTS = std::min(nRead, static_cast<int>(HBHEChannelInfo::MAXSAMPLES));
        const int soi = tsFromDB_ ? param_ts->firstSample() : frame.presamples();
        int soiCapid = 4;

        // Go over time slices and fill the samples
        for (int ts = 0; ts < maxTS; ++ts)
        {
            auto s(frame[ts]);
            const int capid = s.capid();
            const double pedestal = calib.pedestal(capid);
            const double gain = calib.respcorrgain(capid);
            const double rawCharge = getRawChargeFromSample(s, cs[ts], calib);
            const float t = getTDCTimeFromSample(s);
            channelInfo->setSample(ts, s.adc(), rawCharge, pedestal, gain, t);
            if (ts == soi)
                soiCapid = capid;
        }

        // Fill the overall channel info items
        const std::pair<bool,bool> hwerr = findHWErrors(frame, maxTS);
        channelInfo->setChannelInfo(cell, maxTS, soi, soiCapid,
                                    hwerr.first, hwerr.second,
                                    taggedBadByDb || dropByZS);

        // If needed, add the channel info to the output collection
        const bool makeThisRechit = !channelInfo->isDropped();
        if (infos && (saveDroppedInfos_ || makeThisRechit))
            infos->push_back(*channelInfo);

        // Reconstruct the rechit
        if (rechits && makeThisRechit)
        {
            const HcalRecoParam* pptr = nullptr;
            if (recoParamsFromDB_)
                pptr = param_ts;
            HBHERecHit rh = reco_->reconstruct(*channelInfo, pptr, calib, isRealData);
            if (rh.id().rawId())
            {
                setAsicSpecificBits(frame, *channelInfo, &rh);
                setCommonStatusBits(*channelInfo, &rh);
                rechits->push_back(rh);
            }
        }
    }
}

void HBHEPhase1Reconstructor::setCommonStatusBits(
    const HBHEChannelInfo& info, HBHERecHit* rh)
{
    // FIX THIS!!! after status bit inventory
}

void HBHEPhase1Reconstructor::setAsicSpecificBits(
    const HBHEDataFrame& frame, const HBHEChannelInfo& info, HBHERecHit* rh)
{
    // FIX THIS!!! after status bit inventory
}

void HBHEPhase1Reconstructor::setAsicSpecificBits(
    const QIE11DataFrame& frame, const HBHEChannelInfo& info, HBHERecHit* rh)
{
    // FIX THIS!!! after status bit inventory
}

// ------------ method called to produce the data  ------------
void
HBHEPhase1Reconstructor::produce(edm::Event& e, const edm::EventSetup& eventSetup)
{
    using namespace edm;

    // Get the Hcal topology if needed
    ESHandle<HcalTopology> htopo;
    if (tsFromDB_ || recoParamsFromDB_)
    {
        eventSetup.get<HcalRecNumberingRecord>().get(htopo);
        paramTS_->setTopo(htopo.product());
    }

    // Fetch the calibrations
    ESHandle<HcalDbService> conditions;
    eventSetup.get<HcalDbRecord>().get(conditions);

    ESHandle<HcalChannelQuality> p;
    eventSetup.get<HcalChannelQualityRcd>().get("withTopo", p);
 
    ESHandle<HcalSeverityLevelComputer> mycomputer;
    eventSetup.get<HcalSeverityLevelComputerRcd>().get(mycomputer);

    // Find the input data
    unsigned maxOutputSize = 0;
    Handle<HBHEDigiCollection> hbDigis;
    if (processQIE8_)
    {
        e.getByToken(tok_qie8_, hbDigis);
        maxOutputSize += hbDigis->size();
    }

    Handle<QIE11DigiCollection> heDigis;
    if (processQIE11_)
    {
        e.getByToken(tok_qie11_, heDigis);
        maxOutputSize += heDigis->size();
    }

    // Create new output collections
    std::unique_ptr<HBHEChannelInfoCollection> infos;
    if (saveInfos_)
    {
        infos = std::make_unique<HBHEChannelInfoCollection>();
        infos->reserve(maxOutputSize);
    }

    std::unique_ptr<HBHERecHitCollection> out;
    if (makeRecHits_)
    {
        out = std::make_unique<HBHERecHitCollection>();
        out->reserve(maxOutputSize);
    }

    // Process the input collections, filling the output ones
    const bool isData = e.isRealData();
    if (processQIE8_)
    {
        HBHEChannelInfo channelInfo(false);
        processData<HBHEDataFrame>(*hbDigis, *conditions, *p, *mycomputer,
                                   isData, &channelInfo, infos.get(), out.get());
    }

    if (processQIE11_)
    {
        HBHEChannelInfo channelInfo(true);
        processData<QIE11DataFrame>(*heDigis, *conditions, *p, *mycomputer,
                                    isData, &channelInfo, infos.get(), out.get());
    }

    // Add the output collections to the event record
    if (saveInfos_)
        e.put(std::move(infos));
    if (makeRecHits_)
        e.put(std::move(out));
}

// ------------ method called when starting to processes a run  ------------
void
HBHEPhase1Reconstructor::beginRun(edm::Run const& r, edm::EventSetup const& es)
{
    if (tsFromDB_ || recoParamsFromDB_)
    {
        edm::ESHandle<HcalRecoParams> p;
        es.get<HcalRecoParamsRcd>().get(p);
        paramTS_ = std::make_unique<HcalRecoParams>(*p.product());
    }

    if (reco_->isConfigurable())
    {
        recoConfig_ = fetchHBHEPhase1AlgoData(algoConfigClass_, es);
        if (!recoConfig_.get())
            throw cms::Exception("HBHEPhase1BadConfig")
                << "Invalid HFPhase1Reconstructor \"algoConfigClass\" parameter value \""
                << algoConfigClass_ << '"' << std::endl;
        if (!reco_->configure(recoConfig_.get()))
            throw cms::Exception("HBHEPhase1BadConfig")
                << "Failed to configure HBHEPhase1Algo algorithm from EventSetup"
                << std::endl;
    }

    reco_->beginRun(r, es);
}

void
HBHEPhase1Reconstructor::endRun(edm::Run const&, edm::EventSetup const&)
{
    reco_->endRun();
}

#define add_param_set(name) /**/       \
    edm::ParameterSetDescription name; \
    name.setAllowAnything();           \
    desc.add<edm::ParameterSetDescription>(#name, name)

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
HBHEPhase1Reconstructor::fillDescriptions(edm::ConfigurationDescriptions& descriptions)
{
    edm::ParameterSetDescription desc;

    desc.add<edm::InputTag>("digiLabelQIE8");
    desc.add<edm::InputTag>("digiLabelQIE11");
    desc.add<std::string>("algoConfigClass");
    desc.add<bool>("processQIE8");
    desc.add<bool>("processQIE11");
    desc.add<bool>("saveInfos");
    desc.add<bool>("saveDroppedInfos");
    desc.add<bool>("makeRecHits");
    desc.add<bool>("dropZSmarkedPassed");
    desc.add<bool>("tsFromDB");
    desc.add<bool>("recoParamsFromDB");

    add_param_set(algorithm);
    
    descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(HBHEPhase1Reconstructor);
