// -*- C++ -*-
//
// Package:    RecoLocalCalo/HcalRecProducers
// Class:      HFPhase1Reconstructor
// 
/**\class HFPhase1Reconstructor HFPhase1Reconstructor.cc RecoLocalCalo/HcalRecProducers/src/HFPhase1Reconstructor.cc

 Description: Phase 1 HF reco with QIE 10 and split-anode readout

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Igor Volobouev
//         Created:  Thu, 25 May 2016 00:17:51 GMT
//
//

// system include files

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"

#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSeverityLevelComputer.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSeverityLevelComputerRcd.h"

// Noise cleanup algos
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalHF_PETalgorithm.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalHF_S9S1algorithm.h"

// Parser for Phase 1 HF reco algorithms
#include "RecoLocalCalo/HcalRecAlgos/interface/parseHFPhase1AlgoDescription.h"

// Fetcher for reco algorithm data
#include "RecoLocalCalo/HcalRecAlgos/interface/fetchHcalAlgoData.h"

//
// class declaration
//
class HFPhase1Reconstructor : public edm::stream::EDProducer<>
{
public:
    explicit HFPhase1Reconstructor(const edm::ParameterSet&);
    ~HFPhase1Reconstructor();

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
    virtual void beginRun(const edm::Run&, const edm::EventSetup&) override;
    virtual void produce(edm::Event&, const edm::EventSetup&) override;

    // Configuration parameters
    std::string algoConfigClass_;
    bool setNoiseFlags_;
    bool useChannelQualityFromDB_;
    bool checkChannelQualityForDepth3and4_;

    // Other members
    edm::EDGetTokenT<HFPreRecHitCollection> tok_PreRecHit_;
    std::unique_ptr<AbsHFPhase1Algo> reco_;
    std::unique_ptr<AbsHcalAlgoData> recoConfig_;

    // Noise cleanup algos
    std::unique_ptr<HcalHF_S9S1algorithm> hfS9S1_;
    std::unique_ptr<HcalHF_S9S1algorithm> hfS8S1_;
    std::unique_ptr<HcalHF_PETalgorithm>  hfPET_;
};

//
// constructors and destructor
//
HFPhase1Reconstructor::HFPhase1Reconstructor(const edm::ParameterSet& conf)
    : algoConfigClass_(conf.getParameter<std::string>("algoConfigClass")),
      setNoiseFlags_(conf.getParameter<bool>("setNoiseFlags")),
      useChannelQualityFromDB_(conf.getParameter<bool>("useChannelQualityFromDB")),
      checkChannelQualityForDepth3and4_(conf.getParameter<bool>("checkChannelQualityForDepth3and4")),
      reco_(parseHFPhase1AlgoDescription(conf.getParameter<edm::ParameterSet>("algorithm")))
{
    // Check that the reco algorithm has been successfully configured
    if (!reco_.get())
        throw cms::Exception("HFPhase1BadConfig")
            << "Invalid HFPhase1Reconstructor algorithm configuration"
            << std::endl;

    // Configure the noise cleanup algorithms
    if (setNoiseFlags_)
    {
        const edm::ParameterSet& psS9S1 = conf.getParameter<edm::ParameterSet>("S9S1stat");
        hfS9S1_ = std::make_unique<HcalHF_S9S1algorithm>(
            psS9S1.getParameter<std::vector<double> >("short_optimumSlope"),
            psS9S1.getParameter<std::vector<double> >("shortEnergyParams"),
            psS9S1.getParameter<std::vector<double> >("shortETParams"),
            psS9S1.getParameter<std::vector<double> >("long_optimumSlope"),
            psS9S1.getParameter<std::vector<double> >("longEnergyParams"),
            psS9S1.getParameter<std::vector<double> >("longETParams"),
            psS9S1.getParameter<int>("HcalAcceptSeverityLevel"),
            psS9S1.getParameter<bool>("isS8S1") );

        const edm::ParameterSet& psS8S1 = conf.getParameter<edm::ParameterSet>("S8S1stat");
        hfS8S1_ = std::make_unique<HcalHF_S9S1algorithm>(
            psS8S1.getParameter<std::vector<double> >("short_optimumSlope"),
            psS8S1.getParameter<std::vector<double> >("shortEnergyParams"),
            psS8S1.getParameter<std::vector<double> >("shortETParams"),
            psS8S1.getParameter<std::vector<double> >("long_optimumSlope"),
            psS8S1.getParameter<std::vector<double> >("longEnergyParams"),
            psS8S1.getParameter<std::vector<double> >("longETParams"),
            psS8S1.getParameter<int>("HcalAcceptSeverityLevel"),
            psS8S1.getParameter<bool>("isS8S1") );

        const edm::ParameterSet& psPET = conf.getParameter<edm::ParameterSet>("PETstat");
        hfPET_ = std::make_unique<HcalHF_PETalgorithm>(
            psPET.getParameter<std::vector<double> >("short_R"),
            psPET.getParameter<std::vector<double> >("shortEnergyParams"),
            psPET.getParameter<std::vector<double> >("shortETParams"),
            psPET.getParameter<std::vector<double> >("long_R"),
            psPET.getParameter<std::vector<double> >("longEnergyParams"),
            psPET.getParameter<std::vector<double> >("longETParams"),
            psPET.getParameter<int>("HcalAcceptSeverityLevel"),
            psPET.getParameter<std::vector<double> >("short_R_29"),
            psPET.getParameter<std::vector<double> >("long_R_29") );
    }

    // Describe consumed data
    tok_PreRecHit_ = consumes<HFPreRecHitCollection>(
        conf.getParameter<edm::InputTag>("inputLabel"));

    // Register the product
    produces<HFRecHitCollection>();
}


HFPhase1Reconstructor::~HFPhase1Reconstructor()
{
 
   // do anything here that needs to be done at destruction time
   // (e.g. close files, deallocate resources etc.)

}


void
HFPhase1Reconstructor::beginRun(const edm::Run& r, const edm::EventSetup& es)
{
    if (reco_->isConfigurable())
    {
        recoConfig_ = fetchHcalAlgoData(algoConfigClass_, es);
        if (!recoConfig_.get())
            throw cms::Exception("HFPhase1BadConfig")
                << "Invalid HFPhase1Reconstructor \"algoConfigClass\" parameter value \""
                << algoConfigClass_ << '"' << std::endl;
        if (!reco_->configure(recoConfig_.get()))
            throw cms::Exception("HFPhase1BadConfig")
                << "Failed to configure HFPhase1Reconstructor algorithm from EventSetup"
                << std::endl;
    }
}

// ------------ method called to produce the data  ------------
void
HFPhase1Reconstructor::produce(edm::Event& e, const edm::EventSetup& eventSetup)
{
    using namespace edm;

    // Fetch the calibrations
    ESHandle<HcalDbService> conditions;
    eventSetup.get<HcalDbRecord>().get(conditions);

    ESHandle<HcalChannelQuality> p;
    eventSetup.get<HcalChannelQualityRcd>().get("withTopo", p);
    const HcalChannelQuality* myqual = p.product();
 
    ESHandle<HcalSeverityLevelComputer> mycomputer;
    eventSetup.get<HcalSeverityLevelComputerRcd>().get(mycomputer);
    const HcalSeverityLevelComputer* mySeverity = mycomputer.product();

    // Get the input data
    Handle<HFPreRecHitCollection> preRecHits;
    e.getByToken(tok_PreRecHit_, preRecHits);

    // Create a new output collection
    std::unique_ptr<HFRecHitCollection> rec(std::make_unique<HFRecHitCollection>());
    rec->reserve(preRecHits->size());

    // Iterate over the input and fill the output collection
    for (HFPreRecHitCollection::const_iterator it = preRecHits->begin();
         it != preRecHits->end(); ++it)
    {
        // The check whether this PMT is single-anode one should go here.
        // Fix this piece of code if/when mixed-anode readout configurations
        // become available.
        const bool thisIsSingleAnodePMT = false;

        // Check if the anodes were tagged bad in the database
        bool taggedBadByDb[2] = {false, false};
        if (useChannelQualityFromDB_)
        {
            if (checkChannelQualityForDepth3and4_ && !thisIsSingleAnodePMT)
            {
                HcalDetId anodeIds[2];
                anodeIds[0] = it->id();
                anodeIds[1] = anodeIds[0].secondAnodeId();
                for (unsigned i=0; i<2; ++i)
                {
                    const HcalChannelStatus* mydigistatus = myqual->getValues(anodeIds[i].rawId());
                    taggedBadByDb[i] = mySeverity->dropChannel(mydigistatus->getValue());
                }
            }
            else
            {
                const HcalChannelStatus* mydigistatus = myqual->getValues(it->id().rawId());
                const bool b = mySeverity->dropChannel(mydigistatus->getValue());
                taggedBadByDb[0] = b;
                taggedBadByDb[1] = b;
            }
        }

        // Reconstruct the rechit
        const HFRecHit& rh = reco_->reconstruct(
            *it, conditions->getHcalCalibrations(it->id()),
            taggedBadByDb, thisIsSingleAnodePMT);

        // The rechit will have the id of 0 if the algorithm
        // decides that it should be dropped
        if (rh.id().rawId())
            rec->push_back(rh);
    }

    // At this point, the rechits contain energy, timing,
    // as well as proper auxiliary words. We do still need
    // to set certain flags using the noise cleanup algoritms.

    // The following flags require the full set of rechits.
    // These flags need to be set consecutively.
    if (setNoiseFlags_)
    {
        // Step 1:  Set PET flag  (short fibers of |ieta|==29)
        for (HFRecHitCollection::iterator i = rec->begin(); i != rec->end(); ++i)
        {
            int depth=i->id().depth();
            int ieta=i->id().ieta();
            // Short fibers and all channels at |ieta|=29 use PET settings in Algo 3
            if (depth==2 || abs(ieta)==29 ) 
                hfPET_->HFSetFlagFromPET(*i, *rec, myqual, mySeverity);
        }

        // Step 2:  Set S8S1 flag (short fibers or |ieta|==29)
        for (HFRecHitCollection::iterator i = rec->begin(); i != rec->end(); ++i)
        {
            int depth=i->id().depth();
            int ieta=i->id().ieta();
            // Short fibers and all channels at |ieta|=29 use PET settings in Algo 3
            if (depth==2 || abs(ieta)==29 ) 
                hfS8S1_->HFSetFlagFromS9S1(*i, *rec, myqual, mySeverity);
        }

        // Set 3:  Set S9S1 flag (long fibers)
        for (HFRecHitCollection::iterator i = rec->begin(); i != rec->end(); ++i)
        {
            int depth=i->id().depth();
            int ieta=i->id().ieta();
            // Short fibers and all channels at |ieta|=29 use PET settings in Algo 3
            if (depth==1 && abs(ieta)!=29 ) 
                hfS9S1_->HFSetFlagFromS9S1(*i, *rec, myqual, mySeverity);
        }
    }

    // Add the output collection to the event record
    e.put(std::move(rec));
}

#define add_param_set(name) /**/ \
    edm::ParameterSetDescription name; \
    name.setAllowAnything(); \
    desc.add<edm::ParameterSetDescription>(#name, name)

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
HFPhase1Reconstructor::fillDescriptions(edm::ConfigurationDescriptions& descriptions)
{
    edm::ParameterSetDescription desc;

    desc.add<edm::InputTag>("inputLabel");
    desc.add<std::string>("algoConfigClass");
    desc.add<bool>("setNoiseFlags");
    desc.add<bool>("useChannelQualityFromDB");
    desc.add<bool>("checkChannelQualityForDepth3and4");
    desc.add<edm::ParameterSetDescription>("algorithm", fillDescriptionForParseHFPhase1AlgoDescription());

    add_param_set(S9S1stat);
    add_param_set(S8S1stat);
    add_param_set(PETstat);
    
    descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(HFPhase1Reconstructor);
