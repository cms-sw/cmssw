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
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/METFwd.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/PFMETFwd.h"
#include "DataFormats/METReco/interface/GenMET.h"
#include "DataFormats/METReco/interface/GenMETFwd.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETFwd.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticle.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticleFwd.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
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
        PHOTON  = 22,
        ELEC    = 11,
        MUON    = 13,
        MUTRK   = 130,
        PFTAU   = 15,
        TRACK   = 0,  
        PFMET   = 39,
	PFMHT   = 40,
        MET     = 390000,
        GENMET  = 390001,
        CALOMET = 390002,
        HLTMET  = 390003,
        L1MET   = 390004,
        PFJET   = 211,
        CALOJET = 111, 
	CALOMHT = 400002,
        _nMAX
    };

    int nOfCollections;
    int nInitialized;

    const reco::GenParticleCollection            * genParticles;
    const std::vector<reco::Muon>                * muons;
    const std::vector<reco::Track>               * tracks;
    const std::vector<reco::GsfElectron>         * electrons;
    const std::vector<reco::Photon>              * photons;
    const std::vector<reco::MET>                 * METs;
    const std::vector<reco::PFMET>               * pfMETs;
    const std::vector<reco::PFMET>               * pfMHTs;
    const std::vector<reco::GenMET>              * genMETs;
    const std::vector<reco::CaloMET>             * caloMETs;
    const std::vector<reco::CaloMET>             * caloMHTs;
    const std::vector<l1extra::L1EtMissParticle> * l1METs;
    const std::vector<reco::PFTau>               * pfTaus;
    const std::vector<reco::PFJet>               * pfJets;
    const std::vector<reco::CaloJet>             * caloJets;
    const edm::TriggerResults                    * triggerResults ;
    const reco::BeamSpot                         * bs;

    EVTColContainer():
        nOfCollections(6),
        nInitialized(0),
        genParticles(0),
        muons(0),
        tracks(0),
        electrons(0),
        photons(0),
        METs(0),
        pfMETs(0),
        pfMHTs(0),
        genMETs(0),
        caloMETs(0),
        caloMHTs(0),
        l1METs(0),
        pfTaus(0),
        pfJets(0),
        caloJets(0),
        triggerResults(0),
        bs(0)
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
        tracks = 0;
        electrons = 0;
        photons = 0;
        METs = 0;
        pfMETs = 0;
        pfMHTs = 0;
        genMETs = 0;
        caloMETs = 0;
        caloMHTs = 0;
        l1METs = 0;
        pfTaus = 0;
        pfJets = 0;
        caloJets = 0;
        triggerResults = 0;
        bs = 0;
    }

    /// Setter: multiple overloaded function
    void set(const reco::MuonCollection * v)
    {
	muons = v;
        ++nInitialized;
    }
    void set(const reco::TrackCollection * v)
    {
	tracks = v;
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
    void set(const reco::METCollection * v)
    {
        METs = v;
        ++nInitialized;
    }
    void set(const reco::PFMETCollection * v)
    {
        pfMETs = v;
        ++nInitialized;
    }
    void setPFMHT(const reco::PFMETCollection * v)
    {
        pfMHTs = v;
        ++nInitialized;
    }
    void set(const reco::GenMETCollection * v)
    {
        genMETs = v;
        ++nInitialized;
    }
    void set(const reco::CaloMETCollection * v)
    {
        caloMETs = v;
        ++nInitialized;
    }
    void setCaloMHT(const reco::CaloMETCollection * v)
    {
        caloMHTs = v;
        ++nInitialized;
    }
    void set(const l1extra::L1EtMissParticleCollection * v)
    {
        l1METs = v;
        ++nInitialized;
    }
    void set(const reco::PFTauCollection * v)
    {
        pfTaus = v;
        ++nInitialized;
    }
    void set(const reco::PFJetCollection * v)
    {
        pfJets = v;
        ++nInitialized;
    }
    void set(const reco::CaloJetCollection * v)
    {
        caloJets = v;
        ++nInitialized;
    }

    /// Get size of collections
    const unsigned int getSize(const unsigned int & objtype) const
    {
        unsigned int size = 0;
        if (objtype == EVTColContainer::MUON && muons != 0) {
            size = muons->size();
        } else if (objtype == EVTColContainer::MUTRK && tracks != 0) {
            size = tracks->size();
        } else if (objtype == EVTColContainer::TRACK && tracks != 0) {
            size = tracks->size();
        } else if (objtype == EVTColContainer::ELEC && electrons != 0) {
            size = electrons->size();
        } else if (objtype == EVTColContainer::PHOTON && photons != 0) {
            size = photons->size();
        } else if (objtype == EVTColContainer::MET && METs != 0) {
            size = METs->size();
        } else if (objtype == EVTColContainer::PFMET && pfMETs != 0) {
            size = pfMETs->size();
        } else if (objtype == EVTColContainer::PFMHT && pfMHTs != 0) {
            size = pfMHTs->size();
        } else if (objtype == EVTColContainer::GENMET && genMETs != 0) {
            size = genMETs->size();
        } else if (objtype == EVTColContainer::CALOMET && caloMETs != 0) {
            size = caloMETs->size();
        } else if (objtype == EVTColContainer::CALOMHT && caloMHTs != 0) {
            size = caloMHTs->size();
        } else if (objtype == EVTColContainer::L1MET && l1METs != 0) {
            size = l1METs->size();
        } else if (objtype == EVTColContainer::PFTAU && pfTaus != 0) {
            size = pfTaus->size();
        } else if (objtype == EVTColContainer::PFJET && pfJets != 0) {
            size = pfJets->size();
        } else if (objtype == EVTColContainer::CALOJET && caloJets != 0) {
            size = caloJets->size();
        }

        return size;
    }

    /// Tranform types into strings
    const static std::string getTypeString(const unsigned int & objtype)
    {
        std::string objTypestr;

        if (objtype == EVTColContainer::MUON) {
            objTypestr = "Mu";
        } else if (objtype == EVTColContainer::MUTRK) {
            objTypestr = "refittedStandAloneMuons";
        } else if (objtype == EVTColContainer::TRACK) {
            objTypestr = "Track";
        } else if (objtype == EVTColContainer::ELEC) {
            objTypestr = "Ele";
        } else if (objtype == EVTColContainer::PHOTON) {
            objTypestr = "Photon";
        } else if (objtype == EVTColContainer::MET) {
            objTypestr = "MET";
        } else if (objtype == EVTColContainer::PFMET) {
            objTypestr = "PFMET";
        } else if (objtype == EVTColContainer::PFMHT) {
            objTypestr = "PFMHT";
        } else if (objtype == EVTColContainer::GENMET) {
            objTypestr = "GenMET";
        } else if (objtype == EVTColContainer::CALOMET) {
            objTypestr = "CaloMET";
        } else if (objtype == EVTColContainer::CALOMHT) {
            objTypestr = "CaloMHT";
        } else if (objtype == EVTColContainer::L1MET) {
            objTypestr = "l1MET";
        } else if (objtype == EVTColContainer::PFTAU) {
            objTypestr = "PFTau";
        } else if (objtype == EVTColContainer::PFJET) {
            objTypestr = "PFJet";
        } else if (objtype == EVTColContainer::CALOJET) {
            objTypestr = "CaloJet";
        } else {
            edm::LogError("ExoticaValidations") << "EVTColContainer::getTypeString, "
                                                << "NOT Implemented error (object type id='" << objtype << "')" << std::endl;;
        }

        return objTypestr;
    }
};
#endif
