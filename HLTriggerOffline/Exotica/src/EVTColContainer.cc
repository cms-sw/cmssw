#ifndef HLTRIGGEROFFLINE_EXOTICA_EVTCOLCONTAINER_CC
#define HLTRIGGEROFFLINE_EXOTICA_EVTCOLCONTAINER_CC

/** \class EVTColContainer
 *
 *  Class to manage all the object collections in
 *  the Exotica Validation package.
 *
 *  \author  J. Duarte Campderros
 *
 */

#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/PFMETFwd.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include<vector>
#include<map>

/// Container with all the objects needed
/// Notice that I have "more or less" followed
/// the pdgIds of the particles involved.
struct EVTColContainer {
    enum {
        ELEC = 11,
        MUON = 13,
        PFTAU = 15,
        PHOTON = 22,
        PFMET = 39,
        JET = 211,
        _nMAX
    };

    int nOfCollections;
    int nInitialized;
    const reco::GenParticleCollection * genParticles;
    const std::vector<reco::Muon> * muons;
    const std::vector<reco::GsfElectron> * electrons;
    const std::vector<reco::Photon> * photons;
    const std::vector<reco::PFMET> * pfMETs;
    const std::vector<reco::PFTau> * pfTaus;
    const std::vector<reco::PFJet> * jets;
    const edm::TriggerResults   * triggerResults ;
    EVTColContainer():
        nOfCollections(6),
        nInitialized(0),
        genParticles(0),
        muons(0),
        electrons(0),
        photons(0),
        pfMETs(0),
        pfTaus(0),
        jets(0),
        triggerResults(0)
    {
    }
    ///
    bool isAllInit()
    {
        return (nInitialized == nOfCollections);
    }

    bool isCommonInit()
    {
        return false;
    }

    /// Reset: clear all collections
    void reset()
    {
        nInitialized = 0;
        genParticles = 0;
        muons = 0;
        electrons = 0;
        photons = 0;
        pfMETs = 0;
        pfTaus = 0;
        jets = 0;
        triggerResults = 0;
    }

    /// Setter: multiple overloaded function
    void set(const reco::MuonCollection * v)
    {
	muons = v;
        ++nInitialized;
    }
    void set(const reco::GsfElectronCollection * v)
    {
        electrons = v;
        ++nInitialized;
    }
    void set(const reco::PhotonCollection * v)
    {
        photons = v;
        ++nInitialized;
    }
    void set(const reco::PFMETCollection * v)
    {
        pfMETs = v;
        ++nInitialized;
    }
    void set(const reco::PFTauCollection * v)
    {
        pfTaus = v;
        ++nInitialized;
    }
    void set(const reco::PFJetCollection * v)
    {
        jets = v;
        ++nInitialized;
    }

    /// Get size of collections
    const unsigned int getSize(const unsigned int & objtype) const
    {
        unsigned int size = 0;
        if (objtype == EVTColContainer::MUON && muons != 0) {
            size = muons->size();
        } else if (objtype == EVTColContainer::ELEC && electrons != 0) {
            size = electrons->size();
        } else if (objtype == EVTColContainer::PHOTON && photons != 0) {
            size = photons->size();
        } else if (objtype == EVTColContainer::PFMET && pfMETs != 0) {
            size = pfMETs->size();
        } else if (objtype == EVTColContainer::PFTAU && pfTaus != 0) {
            size = pfTaus->size();
        } else if (objtype == EVTColContainer::JET && jets != 0) {
            size = jets->size();
        }

        return size;
    }

    /// Tranform types into strings
    const static std::string getTypeString(const unsigned int & objtype)
    {
        std::string objTypestr;

        if (objtype == EVTColContainer::MUON) {
            objTypestr = "Mu";
        } else if (objtype == EVTColContainer::ELEC) {
            objTypestr = "Ele";
        } else if (objtype == EVTColContainer::PHOTON) {
            objTypestr = "Photon";
        } else if (objtype == EVTColContainer::PFMET) {
            objTypestr = "MET";
        } else if (objtype == EVTColContainer::PFTAU) {
            objTypestr = "PFTau";
        } else if (objtype == EVTColContainer::JET) {
            objTypestr = "Jet";
        } else {
            edm::LogError("ExoticaValidations") << "EVTColContainer::getTypeString, "
                                                << "NOT Implemented error (object type id='" << objtype << "')" << std::endl;;
        }

        return objTypestr;
    }
};
#endif
