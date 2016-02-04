#include <L1Trigger/HardwareValidation/plugins/MuonCandProducerMon.h>

#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/L1TObjects/interface/L1MuTriggerScales.h"
#include "CondFormats/DataRecord/interface/L1MuTriggerScalesRcd.h"
#include "CondFormats/L1TObjects/interface/L1MuTriggerPtScale.h"
#include "CondFormats/DataRecord/interface/L1MuTriggerPtScaleRcd.h"

MuonCandProducerMon::MuonCandProducerMon(const edm::ParameterSet& pset) {

    verbose_ = pset.getUntrackedParameter<int>("VerboseFlag", 0);

    CSCinput_ = pset.getUntrackedParameter<edm::InputTag>("CSCinput",
            (edm::InputTag) ("csctfdigis"));
    DTinput_ = pset.getUntrackedParameter<edm::InputTag>("DTinput",
            (edm::InputTag) ("dttfdigis"));

    cscPtLUT_ = 0;
    m_scalesCacheID = 0;
    m_ptScaleCacheID = 0;

    produces<std::vector<L1MuRegionalCand> >("CSC");
    produces<std::vector<L1MuRegionalCand> >("DT");
}

MuonCandProducerMon::~MuonCandProducerMon() {
}

void MuonCandProducerMon::endJob() {
    if (cscPtLUT_)
        delete cscPtLUT_;
}

void MuonCandProducerMon::produce(edm::Event& iEvent,
        const edm::EventSetup& iSetup) {

    edm::Handle<L1CSCTrackCollection> CSCtracks;
    iEvent.getByLabel(CSCinput_, CSCtracks);

    edm::Handle<L1MuDTTrackContainer> DTtracks;
    iEvent.getByLabel(DTinput_, DTtracks);

    std::auto_ptr<std::vector<L1MuRegionalCand> > csc_product(
            new std::vector<L1MuRegionalCand>);

    std::auto_ptr<std::vector<L1MuRegionalCand> > dt_product(
            new std::vector<L1MuRegionalCand>);

    if (!CSCtracks.isValid()) {

        csc_product->push_back(L1MuRegionalCand());

    } else {

        typedef L1CSCTrackCollection::const_iterator ctcIt;

        for (ctcIt tcit = CSCtracks->begin(); tcit != CSCtracks->end(); tcit++) {

            L1MuRegionalCand cand(tcit->first.getDataWord(), tcit->first.bx());

            // set pt value

            // Update CSCTFTrackBuilder only if the scales have changed.  Use the
            // EventSetup cacheIdentifier to tell when this has happened.
            if (iSetup.get<L1MuTriggerScalesRcd>().cacheIdentifier()
                    != m_scalesCacheID
                    || iSetup.get<L1MuTriggerPtScaleRcd>().cacheIdentifier()
                            != m_ptScaleCacheID) {
                if (cscPtLUT_)
                    delete cscPtLUT_;
                edm::ESHandle<L1MuTriggerScales> scales;
                iSetup.get<L1MuTriggerScalesRcd>().get(scales);
                edm::ESHandle<L1MuTriggerPtScale> ptScale;
                iSetup.get<L1MuTriggerPtScaleRcd>().get(ptScale);
                // Create a dummy pset for CSC Pt LUTs

                edm::ParameterSet ptLUTset;
                ptLUTset.addParameter<bool>("ReadPtLUT", false);
                ptLUTset.addParameter<bool>("isBinary", false);
                ptLUTset.addUntrackedParameter<std::string>("LUTPath", "./");

                cscPtLUT_ = new CSCTFPtLUT(ptLUTset, scales.product(),
                        ptScale.product());
                m_scalesCacheID
                        = iSetup.get<L1MuTriggerScalesRcd>().cacheIdentifier();
                m_ptScaleCacheID
                        = iSetup.get<L1MuTriggerPtScaleRcd>().cacheIdentifier();
            }

            ptadd thePtAddress(tcit->first.ptLUTAddress());
            ptdat thePtData = cscPtLUT_->Pt(thePtAddress);
            const unsigned int rank =
                    (thePtAddress.track_fr ? thePtData.front_rank
                            : thePtData.rear_rank);
            unsigned int quality = 0;
            unsigned int pt = 0;
            csc::L1Track::decodeRank(rank, pt, quality);
            cand.setQualityPacked(quality & 0x3);
            cand.setPtPacked(pt & 0x1f);
            csc_product->push_back(cand);
        }
    }

    if (!DTtracks.isValid()) {

        dt_product->push_back(L1MuRegionalCand());

    } else {

        typedef std::vector<L1MuDTTrackCand>::const_iterator ctcIt;

        std::vector<L1MuDTTrackCand> *dttc = DTtracks->getContainer();

        for (ctcIt it = dttc->begin(); it != dttc->end(); it++) {
            dt_product->push_back(L1MuRegionalCand(*it));
        }
    }

    iEvent.put(csc_product, "CSC");
    iEvent.put(dt_product, "DT");
}

