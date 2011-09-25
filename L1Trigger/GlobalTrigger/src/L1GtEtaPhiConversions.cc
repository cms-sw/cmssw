/**
 * \class L1GtEtaPhiConversions
 * 
 * 
 * Description: convert eta and phi between various L1 trigger objects.
 * 
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: Vasile Mihai Ghete   - HEPHY Vienna 
 * 
 * $Date$
 * $Revision$
 *
 */

// this class header
#include "L1Trigger/GlobalTrigger/interface/L1GtEtaPhiConversions.h"

// system include files
#include <iostream>
#include <iomanip>

// user include files

//   base class

//
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapFwd.h"

#include "CondFormats/L1TObjects/interface/L1MuTriggerScales.h"
#include "CondFormats/L1TObjects/interface/L1CaloGeometry.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

#include "FWCore/Utilities/interface/Exception.h"

// constructor
L1GtEtaPhiConversions::L1GtEtaPhiConversions() :
    m_nrBinsPhiMu(0), m_nrBinsPhiJetEg(0), m_nrBinsPhiEtm(0),
            m_nrBinsPhiHtm(0), m_verbosity(0),
            m_isDebugEnabled(edm::isDebugEnabled()) {

    // prepare the pairs of L1GtObjects, reserve (by hand) memory for vectors
    // the index of the pair in m_gtObjectPairVec is used to extract
    // the information from the other vectors, so the push_back must be done
    // coherently

    std::pair < L1GtObject, L1GtObject > gtObjPair;
    m_gtObjectPairVec.reserve(56);
    m_pairConvertPhiFirstGtObject.reserve(56);
    m_pairNrPhiBinsVec.reserve(56);
    m_pairPhiConvVec.reserve(56);


    // Mu -> Jet & EG & ETM & HTM
    //
    gtObjPair = std::make_pair(Mu, CenJet);
    m_gtObjectPairVec.push_back(gtObjPair);
    m_pairConvertPhiFirstGtObject.push_back(true);
    m_pairNrPhiBinsVec.push_back(&m_nrBinsPhiJetEg);
    m_pairPhiConvVec.push_back(&m_lutPhiMuToJetEg);

    gtObjPair = std::make_pair(CenJet, Mu);
    m_gtObjectPairVec.push_back(gtObjPair);
    m_pairConvertPhiFirstGtObject.push_back(false);
    m_pairNrPhiBinsVec.push_back(&m_nrBinsPhiJetEg);
    m_pairPhiConvVec.push_back(&m_lutPhiMuToJetEg);

    //
    gtObjPair = std::make_pair(Mu, ForJet);
    m_gtObjectPairVec.push_back(gtObjPair);
    m_pairConvertPhiFirstGtObject.push_back(true);
    m_pairNrPhiBinsVec.push_back(&m_nrBinsPhiJetEg);
    m_pairPhiConvVec.push_back(&m_lutPhiMuToJetEg);

    gtObjPair = std::make_pair(ForJet, Mu);
    m_gtObjectPairVec.push_back(gtObjPair);
    m_pairConvertPhiFirstGtObject.push_back(false);
    m_pairNrPhiBinsVec.push_back(&m_nrBinsPhiJetEg);
    m_pairPhiConvVec.push_back(&m_lutPhiMuToJetEg);

    //
    gtObjPair = std::make_pair(Mu, TauJet);
    m_gtObjectPairVec.push_back(gtObjPair);
    m_pairConvertPhiFirstGtObject.push_back(true);
    m_pairNrPhiBinsVec.push_back(&m_nrBinsPhiJetEg);
    m_pairPhiConvVec.push_back(&m_lutPhiMuToJetEg);

    gtObjPair = std::make_pair(TauJet, Mu);
    m_gtObjectPairVec.push_back(gtObjPair);
    m_pairConvertPhiFirstGtObject.push_back(false);
    m_pairNrPhiBinsVec.push_back(&m_nrBinsPhiJetEg);
    m_pairPhiConvVec.push_back(&m_lutPhiMuToJetEg);

    //
    gtObjPair = std::make_pair(Mu, NoIsoEG);
    m_gtObjectPairVec.push_back(gtObjPair);
    m_pairConvertPhiFirstGtObject.push_back(true);
    m_pairNrPhiBinsVec.push_back(&m_nrBinsPhiJetEg);
    m_pairPhiConvVec.push_back(&m_lutPhiMuToJetEg);

    gtObjPair = std::make_pair(NoIsoEG, Mu);
    m_gtObjectPairVec.push_back(gtObjPair);
    m_pairConvertPhiFirstGtObject.push_back(false);
    m_pairNrPhiBinsVec.push_back(&m_nrBinsPhiJetEg);
    m_pairPhiConvVec.push_back(&m_lutPhiMuToJetEg);

    //
    gtObjPair = std::make_pair(Mu, IsoEG);
    m_gtObjectPairVec.push_back(gtObjPair);
    m_pairConvertPhiFirstGtObject.push_back(true);
    m_pairNrPhiBinsVec.push_back(&m_nrBinsPhiJetEg);
    m_pairPhiConvVec.push_back(&m_lutPhiMuToJetEg);

    gtObjPair = std::make_pair(IsoEG, Mu);
    m_gtObjectPairVec.push_back(gtObjPair);
    m_pairConvertPhiFirstGtObject.push_back(false);
    m_pairNrPhiBinsVec.push_back(&m_nrBinsPhiJetEg);
    m_pairPhiConvVec.push_back(&m_lutPhiMuToJetEg);

    //
    gtObjPair = std::make_pair(Mu, ETM);
    m_gtObjectPairVec.push_back(gtObjPair);
    m_pairConvertPhiFirstGtObject.push_back(true);
    m_pairNrPhiBinsVec.push_back(&m_nrBinsPhiEtm);
    m_pairPhiConvVec.push_back(&m_lutPhiMuToEtm);

    gtObjPair = std::make_pair(ETM, Mu);
    m_gtObjectPairVec.push_back(gtObjPair);
    m_pairConvertPhiFirstGtObject.push_back(false);
    m_pairNrPhiBinsVec.push_back(&m_nrBinsPhiEtm);
    m_pairPhiConvVec.push_back(&m_lutPhiMuToEtm);

    //
    gtObjPair = std::make_pair(Mu, HTM);
    m_gtObjectPairVec.push_back(gtObjPair);
    m_pairConvertPhiFirstGtObject.push_back(true);
    m_pairNrPhiBinsVec.push_back(&m_nrBinsPhiHtm);
    m_pairPhiConvVec.push_back(&m_lutPhiMuToHtm);

    gtObjPair = std::make_pair(HTM, Mu);
    m_gtObjectPairVec.push_back(gtObjPair);
    m_pairConvertPhiFirstGtObject.push_back(false);
    m_pairNrPhiBinsVec.push_back(&m_nrBinsPhiHtm);
    m_pairPhiConvVec.push_back(&m_lutPhiMuToHtm);

    // ETM -> Jet & EG
    //
    gtObjPair = std::make_pair(ETM, CenJet);
    m_gtObjectPairVec.push_back(gtObjPair);
    m_pairConvertPhiFirstGtObject.push_back(true);
    m_pairNrPhiBinsVec.push_back(&m_nrBinsPhiJetEg);
    m_pairPhiConvVec.push_back(&m_lutPhiEtmToJetEg);

    gtObjPair = std::make_pair(CenJet, ETM);
    m_gtObjectPairVec.push_back(gtObjPair);
    m_pairConvertPhiFirstGtObject.push_back(false);
    m_pairNrPhiBinsVec.push_back(&m_nrBinsPhiJetEg);
    m_pairPhiConvVec.push_back(&m_lutPhiEtmToJetEg);

    //
    gtObjPair = std::make_pair(ETM, ForJet);
    m_gtObjectPairVec.push_back(gtObjPair);
    m_pairConvertPhiFirstGtObject.push_back(true);
    m_pairNrPhiBinsVec.push_back(&m_nrBinsPhiJetEg);
    m_pairPhiConvVec.push_back(&m_lutPhiEtmToJetEg);

    gtObjPair = std::make_pair(ForJet, ETM);
    m_gtObjectPairVec.push_back(gtObjPair);
    m_pairConvertPhiFirstGtObject.push_back(false);
    m_pairNrPhiBinsVec.push_back(&m_nrBinsPhiJetEg);
    m_pairPhiConvVec.push_back(&m_lutPhiEtmToJetEg);

    //
    gtObjPair = std::make_pair(ETM, TauJet);
    m_gtObjectPairVec.push_back(gtObjPair);
    m_pairConvertPhiFirstGtObject.push_back(true);
    m_pairNrPhiBinsVec.push_back(&m_nrBinsPhiJetEg);
    m_pairPhiConvVec.push_back(&m_lutPhiEtmToJetEg);

    gtObjPair = std::make_pair(TauJet, ETM);
    m_gtObjectPairVec.push_back(gtObjPair);
    m_pairConvertPhiFirstGtObject.push_back(false);
    m_pairNrPhiBinsVec.push_back(&m_nrBinsPhiJetEg);
    m_pairPhiConvVec.push_back(&m_lutPhiEtmToJetEg);

    //
    gtObjPair = std::make_pair(ETM, NoIsoEG);
    m_gtObjectPairVec.push_back(gtObjPair);
    m_pairConvertPhiFirstGtObject.push_back(true);
    m_pairNrPhiBinsVec.push_back(&m_nrBinsPhiJetEg);
    m_pairPhiConvVec.push_back(&m_lutPhiEtmToJetEg);

    gtObjPair = std::make_pair(NoIsoEG, ETM);
    m_gtObjectPairVec.push_back(gtObjPair);
    m_pairConvertPhiFirstGtObject.push_back(false);
    m_pairNrPhiBinsVec.push_back(&m_nrBinsPhiJetEg);
    m_pairPhiConvVec.push_back(&m_lutPhiEtmToJetEg);

    //
    gtObjPair = std::make_pair(ETM, IsoEG);
    m_gtObjectPairVec.push_back(gtObjPair);
    m_pairConvertPhiFirstGtObject.push_back(true);
    m_pairNrPhiBinsVec.push_back(&m_nrBinsPhiJetEg);
    m_pairPhiConvVec.push_back(&m_lutPhiEtmToJetEg);

    gtObjPair = std::make_pair(IsoEG, ETM);
    m_gtObjectPairVec.push_back(gtObjPair);
    m_pairConvertPhiFirstGtObject.push_back(false);
    m_pairNrPhiBinsVec.push_back(&m_nrBinsPhiJetEg);
    m_pairPhiConvVec.push_back(&m_lutPhiEtmToJetEg);

    // HTM -> Jet & EG
    //
    gtObjPair = std::make_pair(HTM, CenJet);
    m_gtObjectPairVec.push_back(gtObjPair);
    m_pairConvertPhiFirstGtObject.push_back(true);
    m_pairNrPhiBinsVec.push_back(&m_nrBinsPhiJetEg);
    m_pairPhiConvVec.push_back(&m_lutPhiHtmToJetEg);

    gtObjPair = std::make_pair(CenJet, HTM);
    m_gtObjectPairVec.push_back(gtObjPair);
    m_pairConvertPhiFirstGtObject.push_back(false);
    m_pairNrPhiBinsVec.push_back(&m_nrBinsPhiJetEg);
    m_pairPhiConvVec.push_back(&m_lutPhiHtmToJetEg);

    //
    gtObjPair = std::make_pair(HTM, ForJet);
    m_gtObjectPairVec.push_back(gtObjPair);
    m_pairConvertPhiFirstGtObject.push_back(true);
    m_pairNrPhiBinsVec.push_back(&m_nrBinsPhiJetEg);
    m_pairPhiConvVec.push_back(&m_lutPhiHtmToJetEg);

    gtObjPair = std::make_pair(ForJet, HTM);
    m_gtObjectPairVec.push_back(gtObjPair);
    m_pairConvertPhiFirstGtObject.push_back(false);
    m_pairNrPhiBinsVec.push_back(&m_nrBinsPhiJetEg);
    m_pairPhiConvVec.push_back(&m_lutPhiHtmToJetEg);

    //
    gtObjPair = std::make_pair(HTM, TauJet);
    m_gtObjectPairVec.push_back(gtObjPair);
    m_pairConvertPhiFirstGtObject.push_back(true);
    m_pairNrPhiBinsVec.push_back(&m_nrBinsPhiJetEg);
    m_pairPhiConvVec.push_back(&m_lutPhiHtmToJetEg);

    gtObjPair = std::make_pair(TauJet, HTM);
    m_gtObjectPairVec.push_back(gtObjPair);
    m_pairConvertPhiFirstGtObject.push_back(false);
    m_pairNrPhiBinsVec.push_back(&m_nrBinsPhiJetEg);
    m_pairPhiConvVec.push_back(&m_lutPhiHtmToJetEg);

    //
    gtObjPair = std::make_pair(HTM, NoIsoEG);
    m_gtObjectPairVec.push_back(gtObjPair);
    m_pairNrPhiBinsVec.push_back(&m_nrBinsPhiJetEg);
    m_pairConvertPhiFirstGtObject.push_back(true);
    m_pairPhiConvVec.push_back(&m_lutPhiHtmToJetEg);

    gtObjPair = std::make_pair(NoIsoEG, HTM);
    m_gtObjectPairVec.push_back(gtObjPair);
    m_pairConvertPhiFirstGtObject.push_back(false);
    m_pairNrPhiBinsVec.push_back(&m_nrBinsPhiJetEg);
    m_pairPhiConvVec.push_back(&m_lutPhiHtmToJetEg);

    //
    gtObjPair = std::make_pair(HTM, IsoEG);
    m_gtObjectPairVec.push_back(gtObjPair);
    m_pairConvertPhiFirstGtObject.push_back(true);
    m_pairNrPhiBinsVec.push_back(&m_nrBinsPhiJetEg);
    m_pairPhiConvVec.push_back(&m_lutPhiHtmToJetEg);

    gtObjPair = std::make_pair(IsoEG, HTM);
    m_gtObjectPairVec.push_back(gtObjPair);
    m_pairConvertPhiFirstGtObject.push_back(false);
    m_pairNrPhiBinsVec.push_back(&m_nrBinsPhiJetEg);
    m_pairPhiConvVec.push_back(&m_lutPhiHtmToJetEg);


    // ETM -> HTM
    //
    gtObjPair = std::make_pair(ETM, HTM);
    m_gtObjectPairVec.push_back(gtObjPair);
    m_pairConvertPhiFirstGtObject.push_back(true);
    m_pairNrPhiBinsVec.push_back(&m_nrBinsPhiHtm);
    m_pairPhiConvVec.push_back(&m_lutPhiEtmToHtm);

    gtObjPair = std::make_pair(HTM, ETM);
    m_gtObjectPairVec.push_back(gtObjPair);
    m_pairConvertPhiFirstGtObject.push_back(false);
    m_pairNrPhiBinsVec.push_back(&m_nrBinsPhiHtm);
    m_pairPhiConvVec.push_back(&m_lutPhiEtmToHtm);


    // Jet & EG -> Jet & EG
    //
    gtObjPair = std::make_pair(CenJet, ForJet);
    m_gtObjectPairVec.push_back(gtObjPair);
    m_pairConvertPhiFirstGtObject.push_back(true);
    m_pairNrPhiBinsVec.push_back(&m_nrBinsPhiJetEg);
    m_pairPhiConvVec.push_back(&m_lutPhiJetEgToJetEg);

    gtObjPair = std::make_pair(ForJet, CenJet);
    m_gtObjectPairVec.push_back(gtObjPair);
    m_pairConvertPhiFirstGtObject.push_back(false);
    m_pairNrPhiBinsVec.push_back(&m_nrBinsPhiJetEg);
    m_pairPhiConvVec.push_back(&m_lutPhiJetEgToJetEg);

    //
    gtObjPair = std::make_pair(CenJet, TauJet);
    m_gtObjectPairVec.push_back(gtObjPair);
    m_pairConvertPhiFirstGtObject.push_back(true);
    m_pairNrPhiBinsVec.push_back(&m_nrBinsPhiJetEg);
    m_pairPhiConvVec.push_back(&m_lutPhiJetEgToJetEg);

    gtObjPair = std::make_pair(TauJet, CenJet);
    m_gtObjectPairVec.push_back(gtObjPair);
    m_pairConvertPhiFirstGtObject.push_back(false);
    m_pairNrPhiBinsVec.push_back(&m_nrBinsPhiJetEg);
    m_pairPhiConvVec.push_back(&m_lutPhiJetEgToJetEg);

    //
    gtObjPair = std::make_pair(CenJet, NoIsoEG);
    m_gtObjectPairVec.push_back(gtObjPair);
    m_pairConvertPhiFirstGtObject.push_back(true);
    m_pairNrPhiBinsVec.push_back(&m_nrBinsPhiJetEg);
    m_pairPhiConvVec.push_back(&m_lutPhiJetEgToJetEg);

    gtObjPair = std::make_pair(NoIsoEG, CenJet);
    m_gtObjectPairVec.push_back(gtObjPair);
    m_pairConvertPhiFirstGtObject.push_back(false);
    m_pairNrPhiBinsVec.push_back(&m_nrBinsPhiJetEg);
    m_pairPhiConvVec.push_back(&m_lutPhiJetEgToJetEg);

    //
    gtObjPair = std::make_pair(CenJet, IsoEG);
    m_gtObjectPairVec.push_back(gtObjPair);
    m_pairConvertPhiFirstGtObject.push_back(true);
    m_pairNrPhiBinsVec.push_back(&m_nrBinsPhiJetEg);
    m_pairPhiConvVec.push_back(&m_lutPhiJetEgToJetEg);

    gtObjPair = std::make_pair(IsoEG, CenJet);
    m_gtObjectPairVec.push_back(gtObjPair);
    m_pairConvertPhiFirstGtObject.push_back(false);
    m_pairNrPhiBinsVec.push_back(&m_nrBinsPhiJetEg);
    m_pairPhiConvVec.push_back(&m_lutPhiJetEgToJetEg);

    //
    gtObjPair = std::make_pair(ForJet, TauJet);
    m_gtObjectPairVec.push_back(gtObjPair);
    m_pairConvertPhiFirstGtObject.push_back(true);
    m_pairNrPhiBinsVec.push_back(&m_nrBinsPhiJetEg);
    m_pairPhiConvVec.push_back(&m_lutPhiJetEgToJetEg);

    gtObjPair = std::make_pair(TauJet, ForJet);
    m_gtObjectPairVec.push_back(gtObjPair);
    m_pairConvertPhiFirstGtObject.push_back(false);
    m_pairNrPhiBinsVec.push_back(&m_nrBinsPhiJetEg);
    m_pairPhiConvVec.push_back(&m_lutPhiJetEgToJetEg);

    //
    gtObjPair = std::make_pair(ForJet, NoIsoEG);
    m_gtObjectPairVec.push_back(gtObjPair);
    m_pairConvertPhiFirstGtObject.push_back(true);
    m_pairNrPhiBinsVec.push_back(&m_nrBinsPhiJetEg);
    m_pairPhiConvVec.push_back(&m_lutPhiJetEgToJetEg);

    gtObjPair = std::make_pair(NoIsoEG, ForJet);
    m_gtObjectPairVec.push_back(gtObjPair);
    m_pairConvertPhiFirstGtObject.push_back(false);
    m_pairNrPhiBinsVec.push_back(&m_nrBinsPhiJetEg);
    m_pairPhiConvVec.push_back(&m_lutPhiJetEgToJetEg);

    //
    gtObjPair = std::make_pair(ForJet, IsoEG);
    m_gtObjectPairVec.push_back(gtObjPair);
    m_pairConvertPhiFirstGtObject.push_back(true);
    m_pairNrPhiBinsVec.push_back(&m_nrBinsPhiJetEg);
    m_pairPhiConvVec.push_back(&m_lutPhiJetEgToJetEg);

    gtObjPair = std::make_pair(IsoEG, ForJet);
    m_gtObjectPairVec.push_back(gtObjPair);
    m_pairConvertPhiFirstGtObject.push_back(false);
    m_pairNrPhiBinsVec.push_back(&m_nrBinsPhiJetEg);
    m_pairPhiConvVec.push_back(&m_lutPhiJetEgToJetEg);

    //
    gtObjPair = std::make_pair(TauJet, NoIsoEG);
    m_gtObjectPairVec.push_back(gtObjPair);
    m_pairConvertPhiFirstGtObject.push_back(true);
    m_pairNrPhiBinsVec.push_back(&m_nrBinsPhiJetEg);
    m_pairPhiConvVec.push_back(&m_lutPhiJetEgToJetEg);

    gtObjPair = std::make_pair(NoIsoEG, TauJet);
    m_gtObjectPairVec.push_back(gtObjPair);
    m_pairConvertPhiFirstGtObject.push_back(false);
    m_pairNrPhiBinsVec.push_back(&m_nrBinsPhiJetEg);
    m_pairPhiConvVec.push_back(&m_lutPhiJetEgToJetEg);

    //
    gtObjPair = std::make_pair(TauJet, IsoEG);
    m_gtObjectPairVec.push_back(gtObjPair);
    m_pairConvertPhiFirstGtObject.push_back(true);
    m_pairNrPhiBinsVec.push_back(&m_nrBinsPhiJetEg);
    m_pairPhiConvVec.push_back(&m_lutPhiJetEgToJetEg);

    gtObjPair = std::make_pair(IsoEG, TauJet);
    m_gtObjectPairVec.push_back(gtObjPair);
    m_pairConvertPhiFirstGtObject.push_back(false);
    m_pairNrPhiBinsVec.push_back(&m_nrBinsPhiJetEg);
    m_pairPhiConvVec.push_back(&m_lutPhiJetEgToJetEg);

    //
    gtObjPair = std::make_pair(NoIsoEG, IsoEG);
    m_gtObjectPairVec.push_back(gtObjPair);
    m_pairConvertPhiFirstGtObject.push_back(true);
    m_pairNrPhiBinsVec.push_back(&m_nrBinsPhiJetEg);
    m_pairPhiConvVec.push_back(&m_lutPhiJetEgToJetEg);

    gtObjPair = std::make_pair(IsoEG, NoIsoEG);
    m_gtObjectPairVec.push_back(gtObjPair);
    m_pairConvertPhiFirstGtObject.push_back(false);
    m_pairNrPhiBinsVec.push_back(&m_nrBinsPhiJetEg);
    m_pairPhiConvVec.push_back(&m_lutPhiJetEgToJetEg);

    // m_verbosity can not be used here, as L1GtEtaPhiConversions is called
    // in L1GlobalTriggerGTL constructor,  where m_verbosity is not yet set
    if (m_isDebugEnabled) {
        LogTrace("L1GlobalTrigger") << "\nm_gtObjectPairVec size: "
                << (m_gtObjectPairVec.size()) << std::endl;

        unsigned int iPair = 0;

        for (std::vector<std::pair<L1GtObject, L1GtObject> >::const_iterator
                cIter = m_gtObjectPairVec.begin();
                cIter != m_gtObjectPairVec.end(); ++cIter) {
            LogTrace("L1GlobalTrigger") << "m_gtObjectPairVec vector element ["
                    << l1GtObjectEnumToString((*cIter).first) << ", "
                    << l1GtObjectEnumToString((*cIter).second)
                    << "], \t\tpair index =  " << iPair << std::endl;

            iPair++;

        }
    }

}

// destructor
L1GtEtaPhiConversions::~L1GtEtaPhiConversions() {

    // do nothing

}

// methods

const unsigned int L1GtEtaPhiConversions::gtObjectPairIndex(
        const L1GtObject& obj0, const L1GtObject& obj1) const {

    std::pair < L1GtObject, L1GtObject > gtObjPair;
    gtObjPair = std::make_pair(obj0, obj1);

    //LogTrace("L1GlobalTrigger") << "\nCompute index for pair ["
    //        << (l1GtObjectEnumToString(obj0)) << ", "
    //        << (l1GtObjectEnumToString(obj1)) << "]\n" << std::endl;

    unsigned int iPair = 0;
    for (std::vector<std::pair<L1GtObject, L1GtObject> >::const_iterator cIter =
            m_gtObjectPairVec.begin(); cIter != m_gtObjectPairVec.end(); ++cIter) {

        if (*cIter == gtObjPair) {
            LogTrace("L1GlobalTrigger") << "\n  Index for pair ["
                    << l1GtObjectEnumToString(obj0) << ", "
                    << l1GtObjectEnumToString(obj1) << "] = "
                    << iPair << std::endl;

            return iPair;
        }

        iPair++;
    }

    // if the pair was not found, return index outside vector size
    // it should never happen, except due to programming error
    // by using .at one gets an exception when using index outside vector size,
    // due to the programming error...
    return m_gtObjectPairVec.size();
}

const unsigned int L1GtEtaPhiConversions::convertPhiIndex(
        unsigned int pairIndex, unsigned int positionPair,
        unsigned int initialIndex) const {

    unsigned int newIndex = 99999;

    switch (positionPair) {
        case 0: {
            if (m_pairConvertPhiFirstGtObject.at(pairIndex)) {

                newIndex = (*(m_pairPhiConvVec.at(pairIndex))).at(initialIndex);

                if (m_verbosity && m_isDebugEnabled) {
                    LogTrace("L1GlobalTrigger") << (positionPair ? "    Second"
                            : "\n  First") << " object from pair " << pairIndex
                            << ": initial phi index " << initialIndex
                            << " converted to " << newIndex << std::endl;
                }

            } else {
                newIndex = initialIndex;

                if (m_verbosity && m_isDebugEnabled) {
                    LogTrace("L1GlobalTrigger") << (positionPair ? "    Second"
                            : "\n  First") << " object from pair " << pairIndex
                            << ": initial phi index " << initialIndex
                            << " not converted, return index " << newIndex
                            << std::endl;
                }
            }
        }

            break;
        case 1: {
            if (m_pairConvertPhiFirstGtObject.at(pairIndex)) {
                newIndex = initialIndex;

                if (m_verbosity && m_isDebugEnabled) {
                    LogTrace("L1GlobalTrigger") << (positionPair ? "    Second"
                            : "\n  First") << " object from pair " << pairIndex
                            << ": initial phi index " << initialIndex
                            << " not converted, return index " << newIndex
                            << std::endl;
                }
            } else {

                newIndex = (*(m_pairPhiConvVec.at(pairIndex))).at(initialIndex);

                if (m_verbosity && m_isDebugEnabled) {
                    LogTrace("L1GlobalTrigger") << (positionPair ? "    Second"
                            : "\n  First") << " object from pair " << pairIndex
                            << ": initial phi index " << initialIndex
                            << " converted to " << newIndex << std::endl;
                }
            }

        }

            break;
        default: {

            // should not happen (programming error)
            throw cms::Exception("FailModule")
                    << "\n  Wrong position in the object pair " << positionPair
                    << "\n  Programming error - position must be either 0 or 1..."
                    << std::endl;

        }
            break;
    }

    return newIndex;

}


const unsigned int L1GtEtaPhiConversions::convertEtaIndex(
        unsigned int pairIndex, unsigned int initialIndex) const {

    // FIXME write the conversion code
    return initialIndex;

}

const unsigned int L1GtEtaPhiConversions::gtObjectNrBinsPhi(
        const L1GtObject& gtObject) const {

    switch (gtObject) {

        case Mu: {
            return m_nrBinsPhiMu;
        }
            break;

        case NoIsoEG:
        case IsoEG:
        case CenJet:
        case ForJet:
        case TauJet: {
            return m_nrBinsPhiJetEg;
        }
            break;

        case ETM: {
            return m_nrBinsPhiEtm;
        }
            break;

        case ETT:
        case HTT: {
            return 0;
        }
            break;

        case HTM: {
            return m_nrBinsPhiHtm;
        }
            break;

        case JetCounts:
        case HfBitCounts:
        case HfRingEtSums:
        case TechTrig:
        case Castor:
        case BPTX:
        case GtExternal:
        case ObjNull: {
            return 0;
        }
            break;

        default: {
            edm::LogInfo("L1GtObject") << "\n  '" << gtObject
                    << "' is not a recognized L1GtObject. "
                    << "\n Return 0 bins.";
            return 0;
        }
            break;
    }

}

const unsigned int L1GtEtaPhiConversions::gtObjectNrBinsPhi(
        const L1GtObject& obj0, const L1GtObject& obj1) const {

    std::pair < L1GtObject, L1GtObject > gtObjPair;
    gtObjPair = std::make_pair(obj0, obj1);

    //LogTrace("L1GlobalTrigger") << "\nCompute gtObjectNrBinsPhi ["
    //        << (l1GtObjectEnumToString(obj0)) << ", "
    //        << (l1GtObjectEnumToString(obj1)) << "]\n" << std::endl;

    int iPair = 0;
    for (std::vector<std::pair<L1GtObject, L1GtObject> >::const_iterator cIter =
            m_gtObjectPairVec.begin(); cIter != m_gtObjectPairVec.end(); ++cIter) {

        if (*cIter == gtObjPair) {
            LogTrace("L1GlobalTrigger") << "\n  gtObjectNrBinsPhi ["
                    << l1GtObjectEnumToString(obj0) << ", "
                    << l1GtObjectEnumToString(obj1) << "] = "
                    << (*(m_pairNrPhiBinsVec.at(iPair))) << std::endl;

            return *(m_pairNrPhiBinsVec.at(iPair));
        }

        iPair++;
    }

    return 0;
}

const unsigned int L1GtEtaPhiConversions::gtObjectNrBinsPhi(
        const unsigned int pairIndex) const {

    if (m_verbosity && m_isDebugEnabled) {
        LogTrace("L1GlobalTrigger")
                << "\n  gtObjectNrBinsPhi for L1 GT object pair index "
                << pairIndex << " = " << (*(m_pairNrPhiBinsVec.at(pairIndex)))
                << std::endl;
    }

    return *(m_pairNrPhiBinsVec.at(pairIndex));
}



// perform all conversions
void L1GtEtaPhiConversions::convert(const L1CaloGeometry* l1CaloGeometry,
        const L1MuTriggerScales* l1MuTriggerScales,
        const int ifCaloEtaNumberBits, const int ifMuEtaNumberBits) {

    // no bullet-proof method, depends on binning ...
    // decide "by hand / by eyes" which object converts to which object

    // number of bins for all scales used

    m_nrBinsPhiMu = 144; // FIXME ask Ivan for size() ...
    //m_nrBinsPhiMu = l1MuTriggerScales->getPhiScale()->size();

    m_nrBinsPhiJetEg = l1CaloGeometry->numberGctEmJetPhiBins();
    m_nrBinsPhiEtm = l1CaloGeometry->numberGctEtSumPhiBins();
    m_nrBinsPhiHtm = l1CaloGeometry->numberGctHtSumPhiBins();

    //
    // convert phi scale for muon (finer) to phi scale for (*Jet, EG) / ETM / HTM (coarser)
    //

    m_lutPhiMuToJetEg.clear();
    m_lutPhiMuToJetEg.resize(m_nrBinsPhiMu);

    m_lutPhiMuToEtm.clear();
    m_lutPhiMuToEtm.resize(m_nrBinsPhiMu);

    m_lutPhiMuToHtm.clear();
    m_lutPhiMuToHtm.resize(m_nrBinsPhiMu);

    for (unsigned int phiMuInd = 0; phiMuInd < m_nrBinsPhiMu; ++phiMuInd) {

        double phiMuLowEdge = l1MuTriggerScales->getPhiScale()->getLowEdge(
                phiMuInd);
        double phiMuHighEdge = l1MuTriggerScales->getPhiScale()->getHighEdge(
                phiMuInd);

        // phi Mu -> (*Jet, EG)

        unsigned int nrBins = m_nrBinsPhiJetEg;

        for (unsigned int iBin = nrBins;; --iBin) {

            double phiLowEdge = l1CaloGeometry->emJetPhiBinLowEdge(iBin);
            double phiHighEdge = l1CaloGeometry->emJetPhiBinHighEdge(iBin);

            if (phiMuLowEdge >= phiLowEdge) {
                m_lutPhiMuToJetEg[phiMuInd] = iBin % nrBins;

                LogTrace("L1GlobalTrigger") << " phiMuIndex \t" << phiMuInd
                        << " [ " << phiMuLowEdge << " \t, "
                        << phiMuHighEdge << "] \t==>\t phiMuJetEG \t"
                        << m_lutPhiMuToJetEg[phiMuInd] << " [ "
                        << phiLowEdge << "\t, " << phiHighEdge << " ]"
                        << std::endl;

                break;
            }
        }

        // phi Mu -> ETM

        nrBins = m_nrBinsPhiEtm;

        for (unsigned int iBin = nrBins;; --iBin) {

            double phiLowEdge = l1CaloGeometry->etSumPhiBinLowEdge(iBin);
            double phiHighEdge = l1CaloGeometry->etSumPhiBinHighEdge(iBin);

            if (phiMuLowEdge >= phiLowEdge) {
                m_lutPhiMuToEtm[phiMuInd] = iBin % nrBins;

                LogTrace("L1GlobalTrigger") << " phiMuIndex \t" << phiMuInd
                        << " [ " << phiMuLowEdge << " \t, "
                        << phiMuHighEdge << "] \t==>\t phiMuToEtm \t"
                        << m_lutPhiMuToEtm[phiMuInd] << " [ " << phiLowEdge
                        << "\t, " << phiHighEdge << " ]" << std::endl;

                break;
            }
        }

        // phi Mu -> HTM

        nrBins = m_nrBinsPhiHtm;

        for (unsigned int iBin = nrBins;; --iBin) {

            double phiLowEdge = l1CaloGeometry->htSumPhiBinLowEdge(iBin);
            double phiHighEdge = l1CaloGeometry->htSumPhiBinHighEdge(iBin);

            if (phiMuLowEdge >= phiLowEdge) {
                m_lutPhiMuToHtm[phiMuInd] = iBin % nrBins;

                LogTrace("L1GlobalTrigger") << " phiMuIndex \t" << phiMuInd
                        << " [ " << phiMuLowEdge << " \t, "
                        << phiMuHighEdge << "] \t==>\t phiMuToHtm \t"
                        << m_lutPhiMuToHtm[phiMuInd] << " [ " << phiLowEdge
                        << "\t, " << phiHighEdge << " ]" << std::endl;

                break;
            }
        }

        LogTrace("L1GlobalTrigger") << std::endl;

    }

    if (m_verbosity && m_isDebugEnabled) {
        LogTrace("L1GlobalTrigger") << "Mu phi conversions" << std::endl;
        for (unsigned int iBin = 0; iBin < m_nrBinsPhiMu; ++iBin) {
            LogTrace("L1GlobalTrigger") << "  Mu phiIndex \t" << iBin
                    << "\t converted to index:"
                    << "\t Jet-EG \t" << m_lutPhiMuToJetEg.at(iBin)
                    << "\t ETM \t"    << m_lutPhiMuToEtm.at(iBin)
                    << "\t HTM \t"    << m_lutPhiMuToHtm.at(iBin)
                    << std::endl;

        }
        LogTrace("L1GlobalTrigger") << std::endl;
    }

    //
    // convert phi scale for ETM to phi scale for (*Jet, EG) / HTM (coarser)
    //

    m_lutPhiEtmToJetEg.clear();
    m_lutPhiEtmToJetEg.resize(m_nrBinsPhiEtm);

    m_lutPhiEtmToHtm.clear();
    m_lutPhiEtmToHtm.resize(m_nrBinsPhiEtm);


    for (unsigned int phiEtmInd = 0; phiEtmInd < m_nrBinsPhiEtm; ++phiEtmInd) {

        double phiEtmLowEdge = l1CaloGeometry->etSumPhiBinLowEdge(phiEtmInd);
        double phiEtmHighEdge = l1CaloGeometry->etSumPhiBinHighEdge(phiEtmInd);

        // phi ETM -> (*Jet, EG)

        unsigned int nrBins = m_nrBinsPhiJetEg;

        for (unsigned int iBin = nrBins;; --iBin) {

            double phiLowEdge = l1CaloGeometry->emJetPhiBinLowEdge(iBin);
            double phiHighEdge = l1CaloGeometry->emJetPhiBinHighEdge(iBin);

            if (phiEtmLowEdge >= phiLowEdge) {
                m_lutPhiEtmToJetEg[phiEtmInd] = iBin % nrBins;

                LogTrace("L1GlobalTrigger") << " phiEtmIndex \t" << phiEtmInd
                        << " [ " << phiEtmLowEdge << " \t, "
                        << phiEtmHighEdge << "] \t==>\t phiEtmJetEG \t"
                        << m_lutPhiEtmToJetEg[phiEtmInd] << " [ "
                        << phiLowEdge << "\t, " << phiHighEdge << " ]"
                        << std::endl;

                break;
            }
        }

        // phi ETM -> HTM

        nrBins = m_nrBinsPhiHtm;

        for (unsigned int iBin = nrBins;; --iBin) {

            double phiLowEdge = l1CaloGeometry->htSumPhiBinLowEdge(iBin);
            double phiHighEdge = l1CaloGeometry->htSumPhiBinHighEdge(iBin);

            if (phiEtmLowEdge >= phiLowEdge) {
                m_lutPhiEtmToHtm[phiEtmInd] = iBin % nrBins;

                LogTrace("L1GlobalTrigger") << " phiEtmIndex \t" << phiEtmInd
                        << " [ " << phiEtmLowEdge << " \t, "
                        << phiEtmHighEdge << "] \t==>\t phiEtmToHtm \t"
                        << m_lutPhiEtmToHtm[phiEtmInd] << " [ " << phiLowEdge
                        << "\t, " << phiHighEdge << " ]" << std::endl;

                break;
            }
        }

        LogTrace("L1GlobalTrigger") << std::endl;

    }

    //
    // convert phi scale for HTM to phi scale for (*Jet, EG)
    //

    m_lutPhiHtmToJetEg.clear();
    m_lutPhiHtmToJetEg.resize(m_nrBinsPhiHtm);


    for (unsigned int phiHtmInd = 0; phiHtmInd < m_nrBinsPhiHtm; ++phiHtmInd) {

        double phiHtmLowEdge = l1CaloGeometry->htSumPhiBinLowEdge(phiHtmInd);
        double phiHtmHighEdge = l1CaloGeometry->htSumPhiBinHighEdge(phiHtmInd);

        unsigned int nrBins = m_nrBinsPhiJetEg;

        for (unsigned int iBin = nrBins;; --iBin) {

            double phiLowEdge = l1CaloGeometry->emJetPhiBinLowEdge(iBin);
            double phiHighEdge = l1CaloGeometry->emJetPhiBinHighEdge(iBin);

            if (phiHtmLowEdge >= phiLowEdge) {
                m_lutPhiHtmToJetEg[phiHtmInd] = iBin % nrBins;

                LogTrace("L1GlobalTrigger") << " phiHtmIndex \t" << phiHtmInd
                        << " [ " << phiHtmLowEdge << " \t, "
                        << phiHtmHighEdge << "] \t==>\t phiHtmJetEG \t"
                        << m_lutPhiHtmToJetEg[phiHtmInd] << " [ "
                        << phiLowEdge << "\t, " << phiHighEdge << " ]"
                        << std::endl;

                break;
            }
        }

    }


    //
    // convert phi scale for (*Jet, EG) to (*Jet, EG)
    // dummy - return the same index as the input index

    m_lutPhiJetEgToJetEg.clear();
    m_lutPhiJetEgToJetEg.resize(m_nrBinsPhiJetEg);

    for (unsigned int phiInd = 0; phiInd < m_nrBinsPhiJetEg; ++phiInd) {
        m_lutPhiJetEgToJetEg[phiInd] = phiInd;
    }


    //
    // convert calo eta scale for CenJet/TauJet & IsoEG/NoIsoEG to a common 
    // central / forward eta scale
    //

    unsigned int nrGctCentralEtaBinsPerHalf =
            l1CaloGeometry->numberGctCentralEtaBinsPerHalf();

    unsigned int nrGctForwardEtaBinsPerHalf =
            l1CaloGeometry->numberGctForwardEtaBinsPerHalf();

    unsigned int nrGctTotalEtaBinsPerHalf = nrGctCentralEtaBinsPerHalf
            + nrGctForwardEtaBinsPerHalf;

    m_lutEtaCenCaloCommon.clear();
    m_lutEtaCenCaloCommon.resize(nrGctTotalEtaBinsPerHalf);

    for (unsigned int cenInd = 0; cenInd < nrGctCentralEtaBinsPerHalf; ++cenInd) {

        // FIXME 

    }

    if (m_verbosity && m_isDebugEnabled) {
        LogTrace("L1GlobalTrigger") << std::endl;
        LogTrace("L1GlobalTrigger") << std::endl;
    }

}

// print all the performed conversions
void L1GtEtaPhiConversions::print(std::ostream& myCout) const {

    myCout << "\n  L1GtEtaPhiConversions print...\n" << std::endl;
    // FIXME

}

