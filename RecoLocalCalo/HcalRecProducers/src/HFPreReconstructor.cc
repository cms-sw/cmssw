// -*- C++ -*-
//
// Package:    RecoLocalCalo/HcalRecProducers
// Class:      HFPreReconstructor
// 
/**\class HFPreReconstructor HFPreReconstructor.cc RecoLocalCalo/HcalRecProducers/src/HFPreReconstructor.cc

 Description: Phase 1 HF reco with QIE 10 and split-anode readout

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Igor Volobouev
//         Created:  Thu, 05 May 2016 00:17:51 GMT
//
//


// system include files
#include <memory>
#include <utility>
#include <cassert>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"

#include "RecoLocalCalo/HcalRecAlgos/interface/HFPreRecAlgo.h"

//
// class declaration
//

class HFPreReconstructor : public edm::stream::EDProducer<>
{
public:
    explicit HFPreReconstructor(const edm::ParameterSet&);
    ~HFPreReconstructor() override;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
    typedef std::pair<HcalDetId,int> PmtAnodeId;
    typedef std::pair<PmtAnodeId,const HFQIE10Info*> QIE10InfoWithId;

    void beginRun(const edm::Run&, const edm::EventSetup&) override;
    void produce(edm::Event&, const edm::EventSetup&) override;

    // Module configuration parameters
    edm::InputTag inputLabel_;
    int forceSOI_;
    int soiShift_;
    bool dropZSmarkedPassed_;
    bool tsFromDB_;

    // Other members
    HFPreRecAlgo reco_;
    edm::EDGetTokenT<QIE10DigiCollection> tok_hfQIE10_;
    std::vector<HFQIE10Info> qie10Infos_;
    std::vector<QIE10InfoWithId> sortedQIE10Infos_;
    std::unique_ptr<HcalRecoParams> paramTS_;

    // Fill qie10Infos_ from the event data
    void fillInfos(const edm::Event& e, const edm::EventSetup& eventSetup);

    // Fill out sortedQIE10Infos_ from qie10Infos_ and return the PMT count
    unsigned sortDataByPmt();
};

//
// constructors and destructor
//
HFPreReconstructor::HFPreReconstructor(const edm::ParameterSet& conf)
    : inputLabel_(conf.getParameter<edm::InputTag>("digiLabel")),
      forceSOI_(conf.getParameter<int>("forceSOI")),
      soiShift_(conf.getParameter<int>("soiShift")),
      dropZSmarkedPassed_(conf.getParameter<bool>("dropZSmarkedPassed")),
      tsFromDB_(conf.getParameter<bool>("tsFromDB")),
      reco_(conf.getParameter<bool>("sumAllTimeSlices"))
{
    // Describe consumed data
    tok_hfQIE10_ = consumes<QIE10DigiCollection>(inputLabel_);

    // Register the product
    produces<HFPreRecHitCollection>();
}


HFPreReconstructor::~HFPreReconstructor()
{
 
   // do anything here that needs to be done at destruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//
unsigned
HFPreReconstructor::sortDataByPmt()
{
    sortedQIE10Infos_.clear();
    unsigned pmtCount = 0;
    const unsigned sz = qie10Infos_.size();
    if (sz)
    {
        // Perform sorting
        sortedQIE10Infos_.reserve(sz);
        const HFQIE10Info* info = &qie10Infos_[0];
        for (unsigned i=0; i<sz; ++i)
        {
            const HcalDetId id(info[i].id());
            sortedQIE10Infos_.push_back(QIE10InfoWithId(PmtAnodeId(id.baseDetId(), id.depth()), info+i));
        }
        std::sort(sortedQIE10Infos_.begin(), sortedQIE10Infos_.end());

        // Count the PMTs
        HcalDetId previousBaseId(sortedQIE10Infos_[0].first.first);
        pmtCount = 1;
        for (unsigned i=1; i<sz; ++i)
        {
            const HcalDetId baseId(sortedQIE10Infos_[i].first.first);
            if (baseId != previousBaseId)
            {
                previousBaseId = baseId;
                ++pmtCount;
            }
        }
    }
    return pmtCount;
}

void
HFPreReconstructor::fillInfos(const edm::Event& e, const edm::EventSetup& eventSetup)
{
    using namespace edm;

    // Clear the collection we want to fill in this method
    qie10Infos_.clear();

    // Get the Hcal topology if needed
    ESHandle<HcalTopology> htopo;
    if (tsFromDB_)
    {
        eventSetup.get<HcalRecNumberingRecord>().get(htopo);
        paramTS_->setTopo(htopo.product());
    }

    // Get the calibrations
    ESHandle<HcalDbService> conditions;
    eventSetup.get<HcalDbRecord>().get(conditions);

    // Get the input collection
    Handle<QIE10DigiCollection> digi;
    e.getByToken(tok_hfQIE10_, digi);

    const unsigned inputSize = digi->size();
    if (inputSize)
    {
        // Process the digis and fill out the HFQIE10Info vector
        qie10Infos_.reserve(inputSize);

        for (QIE10DigiCollection::const_iterator it = digi->begin();
             it != digi->end(); ++it)
        {
            const QIE10DataFrame& frame(*it);
            const HcalDetId cell(frame.id());

            // Protection against calibration channels which are not
            // in the database but can still come in the QIE10DataFrame
            // in the laser calibs, etc.
            if (cell.subdet() != HcalSubdetector::HcalForward)
                continue;

            // Check zero suppression
            if (dropZSmarkedPassed_)
                if (frame.zsMarkAndPass())
                    continue;

            const HcalCalibrations& calibrations(conditions->getHcalCalibrations(cell));
            const HcalQIECoder* channelCoder = conditions->getHcalCoder(cell);
            const HcalQIEShape* shape = conditions->getHcalShape(channelCoder);
            const HcalCoderDb coder(*channelCoder, *shape);

            int tsToUse = forceSOI_;
            if (tsToUse < 0)
            {
                if (tsFromDB_)
                {
                    const HcalRecoParam* param_ts = paramTS_->getValues(cell.rawId());
                    tsToUse = param_ts->firstSample();
                }
                else
                    // Get the "sample of interest" from the data frame itself
                    tsToUse = frame.presamples();
            }

            // Reconstruct the charge, energy, etc
            const HFQIE10Info& info = reco_.reconstruct(frame, tsToUse+soiShift_, coder, calibrations);
            if (info.id().rawId())
                qie10Infos_.push_back(info);
        }
    }
}

void
HFPreReconstructor::beginRun(const edm::Run& r, const edm::EventSetup& es)
{
    if (tsFromDB_)
    {
        edm::ESHandle<HcalRecoParams> p;
        es.get<HcalRecoParamsRcd>().get(p);
        paramTS_ = std::make_unique<HcalRecoParams>(*p.product());
    }
}

// ------------ method called to produce the data  ------------
void
HFPreReconstructor::produce(edm::Event& e, const edm::EventSetup& eventSetup)
{
    // Process the input data
    fillInfos(e, eventSetup);

    // Create a new output collection
    std::unique_ptr<HFPreRecHitCollection> out(std::make_unique<HFPreRecHitCollection>());

    // Fill the output collection
    const unsigned pmtCount = sortDataByPmt();
    if (pmtCount)
    {
        out->reserve(pmtCount);
        const unsigned sz = sortedQIE10Infos_.size();
        HcalDetId previousBaseId(sortedQIE10Infos_[0].first.first);
        unsigned nFound = 1;
        for (unsigned i=1; i<=sz; ++i)
        {
            bool appendData = i == sz;
            if (i < sz)
            {
                const HcalDetId baseId(sortedQIE10Infos_[i].first.first);
                if (baseId == previousBaseId)
                    ++nFound;
                else
                {
                    appendData = true;
                    previousBaseId = baseId;
                }
            }

            if (appendData)
            {
                // If we have found more than two QIE10 with the same base id,
                // there is a bug somewhere in the dataframe. We can't do
                // anything useful about it here. Once we make sure that
                // everything works as expected, this assertion can be removed.
                assert(nFound <= 2);

                const HFQIE10Info* first = nullptr;
                const HFQIE10Info* second = sortedQIE10Infos_[i-1].second;

                if (nFound >= 2)
                    first = sortedQIE10Infos_[i-2].second;
                else if (sortedQIE10Infos_[i-1].first.second < 3)
                {
                    // Only one QIE10 readout found for this PMT.
                    // Arrange for depth 1 and 2 to be "first".
                    first = second;
                    second = nullptr;
                }

                out->push_back(HFPreRecHit(sortedQIE10Infos_[i-nFound].first.first,
                                           first, second));

                // Reset the QIE find count for this base id
                nFound = 1;
            }
        }

        assert(out->size() == pmtCount);
    }

    // Add the output collection to the event record
    e.put(std::move(out));
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
HFPreReconstructor::fillDescriptions(edm::ConfigurationDescriptions& descriptions)
{
    edm::ParameterSetDescription desc;

    desc.add<edm::InputTag>("digiLabel");
    desc.add<int>("forceSOI", -1);
    desc.add<int>("soiShift", 0);
    desc.add<bool>("dropZSmarkedPassed");
    desc.add<bool>("tsFromDB");
    desc.add<bool>("sumAllTimeSlices");

    descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(HFPreReconstructor);
