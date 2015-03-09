#ifndef HLTJetsCleanedFromLeadingLeptons_h
#define HLTJetsCleanedFromLeadingLeptons_h

#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "HLTrigger/HLTcore/interface/defaultModuleLabel.h"



/**
 * \class HLTJetsCleanedFromLeadingLeptons
 * \author Andrey Popov, inspired by code by Lukasz Kreczko
 * \brief Produces a collection of jets cleaned against leading leptons
 * 
 * Leptons (muons or electrons) are read from results of a previous HLT filter. They are ordered in
 * energy, and the user can configure how many leading leptons are used to clean jets. If the
 * requested number is larger than the total number of leptons, all of them are used.
 * 
 * The plugin loops over the given collection of jets and exclude ones that are close to one of the
 * leading leptons. References to surviving jets are stored in the same format as expected by the
 * HLTJetCollectionsFilter plugin.
 */
template <typename JetType>
class HLTJetsCleanedFromLeadingLeptons: public edm::stream::EDProducer<>
{
public:
    typedef std::vector<JetType> JetCollection;
    typedef edm::Ref<JetCollection> JetRef;
    typedef edm::RefVector<JetCollection> JetRefVector;
    
    typedef std::vector<edm::RefVector<JetCollection, JetType,
     edm::refhelper::FindUsingAdvance<JetCollection, JetType>>> JetCollectionVector;
    //^ This is the type expected by HLTJetCollectionsFilter
    
private:
    /**
     * \class EtaPhiE
     * \brief An auxiliary class to store momentum parametrised in eta, phi, and energy
     * 
     * It is useful to deal with muons and ECAL superclusters on a common basis.
     */
    class EtaPhiE
    {
    public:
        /// Constructor
        EtaPhiE(double eta, double phi, double e);
        
    public:
        /// Returns pseudorapidity
        double eta() const;
        
        /// Returns azimuthal angle
        double phi() const;
        
        /// Returns energy
        double e() const;
        
        /// A comparison operator to sort a collection of objects of this type
        bool operator<(EtaPhiE const &rhs) const;
        
    private:
        /// Pseudorapidity and azimuthal angle
        double etaValue, phiValue;
        
        /// Energy
        double eValue;
    };
    
public:
    /// Constructor
    HLTJetsCleanedFromLeadingLeptons(edm::ParameterSet const &iConfig);
    
public:
    /// Describes configuration of the plugin
    static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);
    
    /// Produces jets cleaned against leptons
    virtual void produce(edm::Event &iEvent, edm::EventSetup const &iSetup);
    
private:
    /// Token to identify a collection of leptons that pass an HLT filter
    edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> leptonToken;
    
    /// Token to access a collection of jets
    edm::EDGetTokenT<std::vector<JetType>> jetToken;
    
    /// A square of the minimal allowed angular separation between a lepton and a jet
    double minDeltaR2;
    
    /**
     * \brief Number of leading leptons against which the jets are cleaned
     * 
     * If the number is larger than the total number of leptons, the jets are cleaned against all
     * leptons.
     */
    unsigned numLeptons;
};



// Implementation

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"

#include "DataFormats/Math/interface/deltaR.h"


template <typename JetType>
HLTJetsCleanedFromLeadingLeptons<JetType>::EtaPhiE::EtaPhiE(double eta, double phi, double e):
    etaValue(eta), phiValue(phi),
    eValue(e)
{}


template <typename JetType>
double HLTJetsCleanedFromLeadingLeptons<JetType>::EtaPhiE::eta() const
{
    return etaValue;
}


template <typename JetType>
double HLTJetsCleanedFromLeadingLeptons<JetType>::EtaPhiE::phi() const
{
    return phiValue;
}


template <typename JetType>
double HLTJetsCleanedFromLeadingLeptons<JetType>::EtaPhiE::e() const
{
    return eValue;
}


template <typename JetType>
bool HLTJetsCleanedFromLeadingLeptons<JetType>::EtaPhiE::operator<(EtaPhiE const &rhs) const
{
    return (eValue < rhs.eValue);
}


template <typename JetType>
HLTJetsCleanedFromLeadingLeptons<JetType>::HLTJetsCleanedFromLeadingLeptons(
 edm::ParameterSet const &iConfig):
    minDeltaR2(std::pow(iConfig.getParameter<double>("minDeltaR"), 2)),
    numLeptons(iConfig.getParameter<unsigned>("numLeptons"))
{
    leptonToken = consumes<trigger::TriggerFilterObjectWithRefs>(
     iConfig.getParameter<edm::InputTag>("leptons"));
    jetToken = consumes<std::vector<JetType>>(iConfig.getParameter<edm::InputTag>("jets"));
    
    produces<JetCollectionVector>();
}


template <typename JetType>
void HLTJetsCleanedFromLeadingLeptons<JetType>::fillDescriptions(
 edm::ConfigurationDescriptions &descriptions)
{
    edm::ParameterSetDescription desc;
    
    desc.add<edm::InputTag>("leptons", edm::InputTag("triggerFilterObjectWithRefs"))->
     setComment("A collection of leptons that pass an HLT filter");
    desc.add<edm::InputTag>("jets", edm::InputTag("jetCollection"))->
     setComment("A collection of jets");
    desc.add<double>("minDeltaR", 0.3)->
     setComment("Minimal allowed angular separation between a jet and a lepton");
    desc.add<unsigned>("numLeptons", 1)->
     setComment("Number of leading leptons against which the jets are cleaned");
    
    descriptions.add(defaultModuleLabel<HLTJetsCleanedFromLeadingLeptons<JetType>>(), desc);
}


template <typename JetType>
void HLTJetsCleanedFromLeadingLeptons<JetType>::produce(edm::Event &iEvent,
 edm::EventSetup const &iSetup)
{
    // Read results of the lepton filter
    edm::Handle<trigger::TriggerFilterObjectWithRefs> filterOutput;
    iEvent.getByToken(leptonToken, filterOutput);
    
    // Momenta of the leptons that passed the filter will be pushed into a vector
    std::vector<EtaPhiE> leptonMomenta;
    
    
    // First, assume these are muons and try to store their momenta
    trigger::VRmuon muons;
    filterOutput->getObjects(trigger::TriggerMuon, muons);
    
    for (auto const &muRef: muons)  // the collection might be empty
        leptonMomenta.emplace_back(muRef->eta(), muRef->phi(), muRef->energy());
    
    
    // Then get the momenta as if these are electrons. Electrons are tricky because they can be
    //stored with three types of trigger objects: TriggerElectron, TriggerPhoton, and
    //TriggerCluster. Try them one by one.
    trigger::VRelectron electrons;
    filterOutput->getObjects(trigger::TriggerElectron, electrons);
    
    for (auto const &eRef: electrons)  // the collection might be empty
    {
        auto const &sc = eRef->superCluster();
        leptonMomenta.emplace_back(sc->eta(), sc->phi(), sc->energy());
    }
    
    trigger::VRphoton photons;
    filterOutput->getObjects(trigger::TriggerPhoton, photons);
    
    for (auto const &eRef: photons)  // the collection might be empty
    {
        auto const &sc = eRef->superCluster();
        leptonMomenta.emplace_back(sc->eta(), sc->phi(), sc->energy());
    }
    
    trigger::VRphoton clusters;
    filterOutput->getObjects(trigger::TriggerCluster, clusters);
    
    for (auto const &eRef: clusters)  // the collection might be empty
    {
        auto const &sc = eRef->superCluster();
        leptonMomenta.emplace_back(sc->eta(), sc->phi(), sc->energy());
    }
    
    
    // Make sure the momenta are sorted
    std::sort(leptonMomenta.rbegin(), leptonMomenta.rend());
    
    
    // Read the source collection of jets
    edm::Handle<JetCollection> jetHandle;
    iEvent.getByToken(jetToken, jetHandle);
    JetCollection const &jets = *jetHandle;
    
    
    // Put references to jets that are not matched to leptons into a dedicated collection
    JetRefVector cleanedJetRefs;
    unsigned const numLeptonsToLoop = std::min<unsigned>(leptonMomenta.size(), numLeptons);
    
    for (unsigned iJet = 0; iJet < jets.size(); ++iJet)
    {
        bool overlap = false;
        
        for (unsigned iLepton = 0; iLepton < numLeptonsToLoop; ++iLepton)
            if (reco::deltaR2(leptonMomenta.at(iLepton), jets.at(iJet)) < minDeltaR2)
            {
                overlap = true;
                break;
            }
        
        if (not overlap)
            cleanedJetRefs.push_back(JetRef(jetHandle, iJet));
    }
    
    
    // Store the collection in the event
    std::auto_ptr<JetCollectionVector> product(new JetCollectionVector);
    //^ Have to use the depricated auto_ptr here because this is what edm::Event::put expects
    product->emplace_back(cleanedJetRefs);
    iEvent.put(product);
}

#endif  // HLTJetsCleanedFromLeadingLeptons_h
