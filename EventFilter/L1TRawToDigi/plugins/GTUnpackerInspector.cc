// A filter designed to trigger on bad GT data
// Currently defined as not having 5 BXs
// It will also produce either duplicate GT data, or empty GT data
// depending on whether or not corruption has been found

// Original Author: Andrew Loeliger
// Created: Mon, 22 May, 2023

#include <memory>
#include <string>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/L1Trigger/interface/BXVector.h"
#include "DataFormats/L1Trigger/interface/Muon.h"
#include "DataFormats/L1Trigger/interface/MuonShower.h"
#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/Tau.h"

template<class T>
class GTUnpackerInspector : public edm::stream::EDFilter<> {
    public:
        explicit GTUnpackerInspector(const edm::ParameterSet&);
        ~GTUnpackerInspector() override {};

        static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

    private:
        void beginStream(edm::StreamID) override {streamErrors = 0;};
        bool filter(edm::Event&, const edm::EventSetup&) override;
        void endStream() override;
        bool isDataCorrupt(const edm::Handle<BXVector<T>>& objectHandle);

        // member data
        edm::EDGetTokenT<BXVector<T>> bxCollection;
        unsigned nBXsAllowed;
        string outputCollectionName;
        unsigned streamErrors;
};

template<class T>
GTUnpackerInspector<T>::GTUnpackerInspector(const edm::ParameterSet& iConfig):
    bxCollection(consumes<BXVector<T>>(iConfig.getParameter<edm::InputTag>("objectSrc"))),
    nBXsAllowed(iConfig.getParameter<unsigned>("nBXs")),
    outputCollectionName(iConfig.getParameter<string>("outputCollectionName"))

{
    produces<BXVector<T>>(outputCollectionName);
}


template<class T>
bool GTUnpackerInspector<T>::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    edm::Handle<BXVector<T>> objectHandle;
    iEvent.getByToken(bxCollection, objectHandle);

    //make a default empty BX vector that can be returned if necessary
    int firstBX = -1*(nBXsAllowed/2);
    int lastBX = (nBXsAllowed % 2 == 0) ? -1*firstBX + 1 : -1 * firstBX; //place the extra BX number on the positive side by default
    std::unique_ptr<BXVector<T>> outputObject = std::make_unique<BXVector<T>>(nBXsAllowed, lastBX, firstBX);

    //check the event for corruption
    bool corrupt = isDataCorrupt(objectHandle);

    // if we find corrupt data, we warn and proceed with the empty vector
    if(corrupt)
        edm::LogWarning("UnpackerInspectorSubstitution")<<"A GTUnpackerInspector is putting an empty collection into the event.";
    // Otherwise, we just copy the input data and output it
    else
        *outputObject = *objectHandle;

    iEvent.put(std::move(outputObject), outputCollectionName);

    return not corrupt;
}

template<class T>
void GTUnpackerInspector<T>::endStream()
{
    if(streamErrors > 0)
        edm::LogInfo("UnpackerInspectorStreamCorruptedEvents")<<"A GTUnpackerInspector stream caught "<<streamErrors<<" corrupt data events from the GT unpacker";
}

// This is simplistic at the moment, but left modularized as a function in 
// anticipation that it might see more complicated checks at some point in the future.
template<class T>
bool GTUnpackerInspector<T>::isDataCorrupt(const edm::Handle<BXVector<T>>& objectHandle)
{
    bool isCorrupt =  ((objectHandle->getLastBX()-objectHandle->getFirstBX()+1) == (int)nBXsAllowed);
    if (isCorrupt)
    {
        ++streamErrors;
        edm::LogError("UnpackerInspectorFoundCorruptData")<<"A GTUnpackerInspector has detected corrupted data";
    }
    return isCorrupt;
}

template<class T>
void GTUnpackerInspector<T>::fillDescriptions(edm::ConfigurationDescriptions& descriptions)
{
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("objectSrc");
    desc.add<unsigned>("nBXs", 5);
    desc.add<string>("outputCollectionName"); 
    descriptions.addDefault(desc);
}

typedef GTUnpackerInspector<l1t::Muon> GTMuonUnpackerInspector;
typedef GTUnpackerInspector<l1t::MuonShower> GTMuonShowerUnpackerInspector;
typedef GTUnpackerInspector<l1t::EGamma> GTEGammaUnpackerInspector;
typedef GTUnpackerInspector<l1t::EtSum> GTEtSumUnpackerInspector;
typedef GTUnpackerInspector<l1t::Jet> GTJetUnpackerInspector;
typedef GTUnpackerInspector<l1t::Tau> GTTauUnpackerInspector;

DEFINE_FWK_MODULE(GTMuonUnpackerInspector);
DEFINE_FWK_MODULE(GTMuonShowerUnpackerInspector);
DEFINE_FWK_MODULE(GTEGammaUnpackerInspector);
DEFINE_FWK_MODULE(GTEtSumUnpackerInspector);
DEFINE_FWK_MODULE(GTJetUnpackerInspector);
DEFINE_FWK_MODULE(GTTauUnpackerInspector);