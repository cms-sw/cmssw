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
            m_nrBinsPhiHtm(0), m_nrBinsEtaCommon(0), m_verbosity(0),
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

const bool L1GtEtaPhiConversions::convertPhiIndex(const unsigned int pairIndex,
        const unsigned int positionPair, const unsigned int initialIndex,
        unsigned int& convertedIndex) const {

    unsigned int newIndex = badIndex;
    bool conversionStatus = false;

    // check if initial index is within the scale size
    // could be outside the scale size if there are hardware errors
    // or wrong scale conversions
    if (initialIndex >= (*(m_pairPhiConvVec.at(pairIndex))).size()) {

        conversionStatus = false;

        if (m_verbosity && m_isDebugEnabled) {
            LogTrace("L1GlobalTrigger") << (positionPair ? "    Second"
                    : "\n  First") << " object from pair " << pairIndex
                    << ": initial phi index " << initialIndex << " >= "
                    << ((*(m_pairPhiConvVec.at(pairIndex))).size())
                    << " Conversion failed." << std::endl;
        }
    } else {
        if (m_verbosity && m_isDebugEnabled) {
            LogTrace("L1GlobalTrigger") << (positionPair ? "    Second"
                    : "\n  First") << " object from pair " << pairIndex
                    << ": initial phi index " << initialIndex
                    << " within scale size " <<
                    ((*(m_pairPhiConvVec.at(pairIndex))).size())
                    << std::endl;
        }

    }

    // convert the index
    switch (positionPair) {
        case 0: {
            if (m_pairConvertPhiFirstGtObject.at(pairIndex)) {

                newIndex = (*(m_pairPhiConvVec.at(pairIndex))).at(initialIndex);

                if (newIndex != badIndex) {

                    conversionStatus = true;

                    if (m_verbosity && m_isDebugEnabled) {
                        LogTrace("L1GlobalTrigger")
                                << (positionPair ? "    Second" : "\n  First")
                                << " object from pair " << pairIndex
                                << ": initial phi index " << initialIndex
                                << " converted to " << newIndex << std::endl;
                    }

                } else {

                    conversionStatus = false;

                    if (m_verbosity && m_isDebugEnabled) {
                        LogTrace("L1GlobalTrigger")
                                << (positionPair ? "    Second" : "\n  First")
                                << " object from pair " << pairIndex
                                << ": converted phi index " << newIndex
                                << "is equal to badIndex " << badIndex
                                << " Conversion failed." << std::endl;
                    }
                }

            } else {
                newIndex = initialIndex;
                conversionStatus = true;

                if (m_verbosity && m_isDebugEnabled) {
                    LogTrace("L1GlobalTrigger") << (positionPair ? "    Second"
                            : "\n  First") << " object from pair " << pairIndex
                            << ": initial phi index " << initialIndex
                            << " not requested to be converted, return index "
                            << newIndex << std::endl;
                }
            }
        }

            break;
        case 1: {
            if (m_pairConvertPhiFirstGtObject.at(pairIndex)) {

                newIndex = initialIndex;
                conversionStatus = true;

                if (m_verbosity && m_isDebugEnabled) {
                    LogTrace("L1GlobalTrigger") << (positionPair ? "    Second"
                            : "\n  First") << " object from pair " << pairIndex
                            << ": initial phi index " << initialIndex
                            << " not requested to be converted, return index, return index "
                            << newIndex << std::endl;
                }
            } else {

                newIndex = (*(m_pairPhiConvVec.at(pairIndex))).at(initialIndex);

                if (newIndex != badIndex) {

                    conversionStatus = true;

                    if (m_verbosity && m_isDebugEnabled) {
                        LogTrace("L1GlobalTrigger")
                                << (positionPair ? "    Second" : "\n  First")
                                << " object from pair " << pairIndex
                                << ": initial phi index " << initialIndex
                                << " converted to " << newIndex << std::endl;
                    }

                } else {

                    conversionStatus = false;

                    if (m_verbosity && m_isDebugEnabled) {
                        LogTrace("L1GlobalTrigger")
                                << (positionPair ? "    Second" : "\n  First")
                                << " object from pair " << pairIndex
                                << ": converted phi index " << newIndex
                                << "is equal to badIndex " << badIndex
                                << " Conversion failed." << std::endl;
                    }
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

    //
    convertedIndex = newIndex;
    return conversionStatus;

}

const bool L1GtEtaPhiConversions::convertEtaIndex(const L1GtObject& gtObject,
        const unsigned int initialIndex, unsigned int& convertedIndex) const {

    unsigned int newIndex = badIndex;
    bool conversionStatus = false;

    switch (gtObject) {

        case Mu: {

            // check if initial index is within the scale size
            // could be outside the scale size if there are hardware errors
            // or wrong scale conversions
            if (initialIndex >= m_lutEtaMuToCommonCalo.size()) {

                conversionStatus = false;

                if (m_verbosity && m_isDebugEnabled) {
                    LogTrace("L1GlobalTrigger") << "    L1 GT object "
                            << (l1GtObjectEnumToString(gtObject))
                            << " has initial eta index " << initialIndex
                            << " >= " << (m_lutEtaMuToCommonCalo.size())
                            << " scale size. Conversion failed." << std::endl;
                }
            } else {

                // convert the index
                newIndex = m_lutEtaMuToCommonCalo[initialIndex];

                if (newIndex != badIndex) {

                    conversionStatus = true;

                    if (m_verbosity && m_isDebugEnabled) {
                        LogTrace("L1GlobalTrigger") << "    L1 GT object "
                                << (l1GtObjectEnumToString(gtObject))
                                << " initial eta index " << initialIndex
                                << " (within scale size "
                                << (m_lutEtaMuToCommonCalo.size())
                                << ") converted to " << newIndex << std::endl;
                    }

                } else {

                    conversionStatus = false;

                    if (m_verbosity && m_isDebugEnabled) {
                        LogTrace("L1GlobalTrigger") << "    L1 GT object "
                                << (l1GtObjectEnumToString(gtObject))
                                << " initial eta index " << initialIndex
                                << " (within scale size "
                                << (m_lutEtaMuToCommonCalo.size())
                                << ") converted to badIndex" << newIndex
                                << " Conversion failed." << std::endl;
                    }
                }

            }

        }
            break;

        case NoIsoEG:
        case IsoEG:
        case CenJet:
        case TauJet: {

            // check if initial index is within the scale size
            // could be outside the scale size if there are hardware errors
            // or wrong scale conversions
            if (initialIndex >= m_lutEtaCentralToCommonCalo.size()) {

                conversionStatus = false;

                if (m_verbosity && m_isDebugEnabled) {
                    LogTrace("L1GlobalTrigger") << "    L1 GT object "
                            << (l1GtObjectEnumToString(gtObject))
                            << " has initial eta index " << initialIndex
                            << " >= " << (m_lutEtaCentralToCommonCalo.size())
                            << " scale size. Conversion failed." << std::endl;
                }
            } else {

                // convert the index
                newIndex = m_lutEtaCentralToCommonCalo[initialIndex];

                if (newIndex != badIndex) {

                    conversionStatus = true;

                    if (m_verbosity && m_isDebugEnabled) {
                        LogTrace("L1GlobalTrigger") << "    L1 GT object "
                                << (l1GtObjectEnumToString(gtObject))
                                << " initial eta index " << initialIndex
                                << " (within scale size "
                                << (m_lutEtaMuToCommonCalo.size())
                                << ") converted to " << newIndex << std::endl;
                    }

                } else {

                    conversionStatus = false;

                    if (m_verbosity && m_isDebugEnabled) {
                        LogTrace("L1GlobalTrigger") << "    L1 GT object "
                                << (l1GtObjectEnumToString(gtObject))
                                << " initial eta index " << initialIndex
                                << " (within scale size "
                                << (m_lutEtaCentralToCommonCalo.size())
                                << ") converted to badIndex" << newIndex
                                << " Conversion failed." << std::endl;
                    }
                }

            }

        }
            break;

        case ForJet: {

            // check if initial index is within the scale size
            // could be outside the scale size if there are hardware errors
            // or wrong scale conversions
            if (initialIndex >= m_lutEtaForJetToCommonCalo.size()) {

                conversionStatus = false;

                if (m_verbosity && m_isDebugEnabled) {
                    LogTrace("L1GlobalTrigger") << "    L1 GT object "
                            << (l1GtObjectEnumToString(gtObject))
                            << " has initial eta index " << initialIndex
                            << " >= " << (m_lutEtaForJetToCommonCalo.size())
                            << " scale size. Conversion failed." << std::endl;
                }
            } else {

                // convert the index
                newIndex = m_lutEtaForJetToCommonCalo[initialIndex];

                if (newIndex != badIndex) {

                    conversionStatus = true;

                    if (m_verbosity && m_isDebugEnabled) {
                        LogTrace("L1GlobalTrigger") << "    L1 GT object "
                                << (l1GtObjectEnumToString(gtObject))
                                << " initial eta index " << initialIndex
                                << " (within scale size "
                                << (m_lutEtaMuToCommonCalo.size())
                                << ") converted to " << newIndex << std::endl;
                    }

                } else {

                    conversionStatus = false;

                    if (m_verbosity && m_isDebugEnabled) {
                        LogTrace("L1GlobalTrigger") << "    L1 GT object "
                                << (l1GtObjectEnumToString(gtObject))
                                << " initial eta index " << initialIndex
                                << " (within scale size "
                                << (m_lutEtaForJetToCommonCalo.size())
                                << ") converted to badIndex" << newIndex
                                << " Conversion failed." << std::endl;
                    }
                }

            }
        }
            break;

        case ETM:
        case ETT:
        case HTT:
        case HTM:
        case JetCounts:
        case HfBitCounts:
        case HfRingEtSums:
        case TechTrig:
        case Castor:
        case BPTX:
        case GtExternal:
        case ObjNull: {

            //no conversions needed, there is no eta quantity for these objects
            conversionStatus = false;
        }
            break;

        default: {
            edm::LogInfo("L1GtObject") << "\n  '" << (l1GtObjectEnumToString(
                    gtObject)) << "' is not a recognized L1GtObject. "
                    << "\n Conversion failed. " << std::endl;
            conversionStatus = false;
        }
            break;
    }

    //
    convertedIndex = newIndex;

    return conversionStatus;

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
            edm::LogInfo("L1GtObject") << "\n  '"
                    << (l1GtObjectEnumToString(gtObject))
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
void L1GtEtaPhiConversions::convertL1Scales(
        const L1CaloGeometry* l1CaloGeometry,
        const L1MuTriggerScales* l1MuTriggerScales,
        const int ifCaloEtaNumberBits, const int ifMuEtaNumberBits) {

    // no bullet-proof method, depends on binning ...
    // decide "by hand / by eyes" which object converts to which object

    // update the scales used
    m_l1CaloGeometry = l1CaloGeometry;
    m_l1MuTriggerScales =  l1MuTriggerScales;

    // number of bins for all phi scales used

    m_nrBinsPhiMu = 144; // FIXME ask Ivan for size() ...
    //m_nrBinsPhiMu = m_l1MuTriggerScales->getPhiScale()->size();

    m_nrBinsPhiJetEg = m_l1CaloGeometry->numberGctEmJetPhiBins();
    m_nrBinsPhiEtm = m_l1CaloGeometry->numberGctEtSumPhiBins();
    m_nrBinsPhiHtm = m_l1CaloGeometry->numberGctHtSumPhiBins();

    //
    // convert phi scale for muon (finer) to phi scale for (*Jet, EG) / ETM / HTM (coarser)
    //

    m_lutPhiMuToJetEg.clear();
    m_lutPhiMuToJetEg.resize(m_nrBinsPhiMu, badIndex);

    m_lutPhiMuToEtm.clear();
    m_lutPhiMuToEtm.resize(m_nrBinsPhiMu, badIndex);

    m_lutPhiMuToHtm.clear();
    m_lutPhiMuToHtm.resize(m_nrBinsPhiMu, badIndex);

    for (unsigned int phiMuInd = 0; phiMuInd < m_nrBinsPhiMu; ++phiMuInd) {

        double phiMuLowEdge = m_l1MuTriggerScales->getPhiScale()->getLowEdge(
                phiMuInd);
        double phiMuHighEdge = m_l1MuTriggerScales->getPhiScale()->getHighEdge(
                phiMuInd);

        // to avoid precision problems, add a small quantity to phiMuLowEdge
        double phiMuLowEdgeSmallShiftRight = phiMuLowEdge + (phiMuHighEdge
                - phiMuLowEdge) / 100.;

        // phi Mu -> (*Jet, EG)

        unsigned int nrBins = m_nrBinsPhiJetEg;

        for (unsigned int iBin = nrBins;; --iBin) {

            double phiLowEdge = m_l1CaloGeometry->emJetPhiBinLowEdge(iBin);
            double phiHighEdge = m_l1CaloGeometry->emJetPhiBinHighEdge(iBin);

            if (phiMuLowEdgeSmallShiftRight >= phiLowEdge) {
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

            double phiLowEdge = m_l1CaloGeometry->etSumPhiBinLowEdge(iBin);
            double phiHighEdge = m_l1CaloGeometry->etSumPhiBinHighEdge(iBin);

            if (phiMuLowEdgeSmallShiftRight >= phiLowEdge) {
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

            double phiLowEdge = m_l1CaloGeometry->htSumPhiBinLowEdge(iBin);
            double phiHighEdge = m_l1CaloGeometry->htSumPhiBinHighEdge(iBin);

            if (phiMuLowEdgeSmallShiftRight >= phiLowEdge) {
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
    m_lutPhiEtmToJetEg.resize(m_nrBinsPhiEtm, badIndex);

    m_lutPhiEtmToHtm.clear();
    m_lutPhiEtmToHtm.resize(m_nrBinsPhiEtm, badIndex);


    for (unsigned int phiEtmInd = 0; phiEtmInd < m_nrBinsPhiEtm; ++phiEtmInd) {

        double phiEtmLowEdge = m_l1CaloGeometry->etSumPhiBinLowEdge(phiEtmInd);
        double phiEtmHighEdge = m_l1CaloGeometry->etSumPhiBinHighEdge(phiEtmInd);

        // to avoid precision problems, add a small quantity to phiEtmLowEdge
        double phiEtmLowEdgeSmallShiftRight = phiEtmLowEdge + (phiEtmHighEdge
                - phiEtmLowEdge) / 100.;

        // phi ETM -> (*Jet, EG)

        unsigned int nrBins = m_nrBinsPhiJetEg;

        for (unsigned int iBin = nrBins;; --iBin) {

            double phiLowEdge = m_l1CaloGeometry->emJetPhiBinLowEdge(iBin);
            double phiHighEdge = m_l1CaloGeometry->emJetPhiBinHighEdge(iBin);

            if (phiEtmLowEdgeSmallShiftRight >= phiLowEdge) {
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

            double phiLowEdge = m_l1CaloGeometry->htSumPhiBinLowEdge(iBin);
            double phiHighEdge = m_l1CaloGeometry->htSumPhiBinHighEdge(iBin);

            if (phiEtmLowEdgeSmallShiftRight >= phiLowEdge) {
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
    m_lutPhiHtmToJetEg.resize(m_nrBinsPhiHtm, badIndex);


    for (unsigned int phiHtmInd = 0; phiHtmInd < m_nrBinsPhiHtm; ++phiHtmInd) {

        double phiHtmLowEdge = m_l1CaloGeometry->htSumPhiBinLowEdge(phiHtmInd);
        double phiHtmHighEdge = m_l1CaloGeometry->htSumPhiBinHighEdge(phiHtmInd);

        // to avoid precision problems, add a small quantity to phiHtmLowEdge
        double phiHtmLowEdgeSmallShiftRight = phiHtmLowEdge + (phiHtmHighEdge
                - phiHtmLowEdge) / 100.;

        unsigned int nrBins = m_nrBinsPhiJetEg;

        for (unsigned int iBin = nrBins;; --iBin) {

            double phiLowEdge = m_l1CaloGeometry->emJetPhiBinLowEdge(iBin);
            double phiHighEdge = m_l1CaloGeometry->emJetPhiBinHighEdge(iBin);

            if (phiHtmLowEdgeSmallShiftRight >= phiLowEdge) {
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
    m_lutPhiJetEgToJetEg.resize(m_nrBinsPhiJetEg, badIndex);

    for (unsigned int phiInd = 0; phiInd < m_nrBinsPhiJetEg; ++phiInd) {
        m_lutPhiJetEgToJetEg[phiInd] = phiInd;
    }

    //
    // eta conversions
    //

    // all objects are converted to a common central / forward calorimeter eta scale,
    // built by setting together the forward scale and the central scale
    //
    // eta is signed,  MSB is the sign for all objects - must be taken into account
    // in conversion - the common scale is, from 0 to m_nrBinsEtaCommon:
    //
    // [ForJet negative bins][Central Jet/IsoEG/NoIsoEG negative bins][Central Jet/IsoEG/NoIsoEG positive bins][ForJet positive bins]
    //

    unsigned int nrGctCentralEtaBinsPerHalf =
            m_l1CaloGeometry->numberGctCentralEtaBinsPerHalf();

    unsigned int nrGctForwardEtaBinsPerHalf =
            m_l1CaloGeometry->numberGctForwardEtaBinsPerHalf();

    unsigned int nrGctTotalEtaBinsPerHalf = nrGctCentralEtaBinsPerHalf
            + nrGctForwardEtaBinsPerHalf;

    m_nrBinsEtaCommon = 2*nrGctTotalEtaBinsPerHalf;

    //
    // convert eta scale for CenJet/TauJet & IsoEG/NoIsoEG to a common
    // central / forward calorimeter eta scale
    //
    // get the sign and the index absolute value

    LogTrace("L1GlobalTrigger")
            << " \nEta conversion: CenJet/TauJet & IsoEG/NoIsoEG to a common calorimeter scale\n"
            << std::endl;

    m_lutEtaCentralToCommonCalo.clear();
    m_lutEtaCentralToCommonCalo.resize(
            (nrGctCentralEtaBinsPerHalf | (1 << (ifCaloEtaNumberBits - 1))),
            badIndex);

    for (unsigned int etaInd = 0; etaInd < nrGctCentralEtaBinsPerHalf; ++etaInd) {

        // for positive values, the index is etaInd
        unsigned int globalIndex = m_l1CaloGeometry->globalEtaIndex(
                m_l1CaloGeometry->etaBinCenter(etaInd, true));
        m_lutEtaCentralToCommonCalo[etaInd] = globalIndex;

        LogTrace("L1GlobalTrigger") << " etaIndex " << etaInd << "\t [hex: "
                << std::hex << etaInd << "] " << std::dec
                << " ==> etaIndexGlobal " << globalIndex << std::endl;

        // for negative values, one adds (binary) 1 as MSB to the index
        unsigned int etaIndNeg = etaInd | (1 << (ifCaloEtaNumberBits - 1));
        globalIndex = m_l1CaloGeometry->globalEtaIndex(
                m_l1CaloGeometry->etaBinCenter(etaIndNeg, true));
        m_lutEtaCentralToCommonCalo[etaIndNeg] = globalIndex;

        LogTrace("L1GlobalTrigger") << " etaIndex " << etaIndNeg
                << "\t [hex: " << std::hex << etaIndNeg << "] " << std::dec
                << " ==> etaIndexGlobal " << globalIndex << std::endl;

    }

    //
    // convert eta scale for ForJet to a common
    // central / forward calorimeter eta scale
    //

    LogTrace("L1GlobalTrigger")
            << " \nEta conversion: ForJet to a common calorimeter scale\n"
            << std::endl;

    m_lutEtaForJetToCommonCalo.clear();
    m_lutEtaForJetToCommonCalo.resize(
            (nrGctForwardEtaBinsPerHalf | (1 << (ifCaloEtaNumberBits - 1))),
            badIndex);

    for (unsigned int etaInd = 0; etaInd < nrGctForwardEtaBinsPerHalf; ++etaInd) {

        // for positive values, the index is etaInd
        unsigned int globalIndex = m_l1CaloGeometry->globalEtaIndex(
                m_l1CaloGeometry->etaBinCenter(etaInd, false));
        m_lutEtaForJetToCommonCalo[etaInd] = globalIndex;

        LogTrace("L1GlobalTrigger") << " etaIndex " << etaInd << "\t [hex: "
                << std::hex << etaInd << "] " << std::dec
                << " ==> etaIndexGlobal " << globalIndex << std::endl;

        // for negative values, one adds (binary) 1 as MSB to the index
        unsigned int etaIndNeg = etaInd | (1 << (ifCaloEtaNumberBits - 1));
        globalIndex = m_l1CaloGeometry->globalEtaIndex(
                m_l1CaloGeometry->etaBinCenter(etaIndNeg, false));
        m_lutEtaForJetToCommonCalo[etaIndNeg] = globalIndex;

        LogTrace("L1GlobalTrigger") << " etaIndex " << etaIndNeg << "\t [hex: "
                << std::hex << etaIndNeg << "] " << std::dec
                << " ==> etaIndexGlobal " << globalIndex << std::endl;

    }

    //
    // convert eta scale for Mu to a common
    // central / forward calorimeter eta scale
    //

    LogDebug("L1GlobalTrigger")
            << " \nEta conversion: Mu to a common calorimeter scale\n"
            << std::endl;

    // eta scale defined for positive values - need to be symmetrized
    unsigned int nrBinsEtaMuPerHalf =
            m_l1MuTriggerScales->getGMTEtaScale()->getNBins();
    LogTrace("L1GlobalTrigger") << " \nnrBinsEtaMuPerHalf = "
            << nrBinsEtaMuPerHalf << "\n" << std::endl;

    m_lutEtaMuToCommonCalo.clear();
    m_lutEtaMuToCommonCalo.resize(
            (nrBinsEtaMuPerHalf | (1 << (ifMuEtaNumberBits - 1))), badIndex);

    for (unsigned int etaMuInd = 0; etaMuInd < nrBinsEtaMuPerHalf; ++etaMuInd) {

        double etaMuLowEdge = m_l1MuTriggerScales->getGMTEtaScale()->getValue(
                etaMuInd);
        double etaMuHighEdge = m_l1MuTriggerScales->getGMTEtaScale()->getValue(
                etaMuInd + 1);

        // to avoid precision problems, add a small quantity to etaMuLowEdge
        double etaMuLowEdgeSmallShiftRight = etaMuLowEdge + (etaMuHighEdge
                - etaMuLowEdge) / 100.;

        // positive values
        for (unsigned int iBin = m_nrBinsEtaCommon;; --iBin) {

            double etaLowEdge = m_l1CaloGeometry->globalEtaBinLowEdge(iBin);

            double etaHighEdge = 0.0;
            if (iBin == m_nrBinsEtaCommon) {
                etaHighEdge = etaLowEdge;
            } else {
                etaHighEdge = m_l1CaloGeometry->globalEtaBinLowEdge(iBin + 1);
            }

            if (etaMuLowEdgeSmallShiftRight >= etaLowEdge) {
                m_lutEtaMuToCommonCalo[etaMuInd] = iBin % m_nrBinsEtaCommon;

                LogTrace("L1GlobalTrigger") << " etaMuIndex \t" << etaMuInd
                        << "\t [ " << etaMuLowEdge << ", \t" << etaMuHighEdge
                        << "] ==> etaMuJetEG \t"
                        << m_lutEtaMuToCommonCalo[etaMuInd] << "\t [ "
                        << etaLowEdge << ", \t" << etaHighEdge << " ]"
                        << std::endl;

                break;
            }
        }

        // for negative values, one adds (binary) 1 as MSB to the index
        unsigned int etaMuIndNeg = etaMuInd | (1 << (ifMuEtaNumberBits - 1));
        m_lutEtaMuToCommonCalo[etaMuIndNeg] = m_lutEtaMuToCommonCalo[0]
                - (m_lutEtaMuToCommonCalo[etaMuInd] - m_lutEtaMuToCommonCalo[0]
                        + 1);

        LogTrace("L1GlobalTrigger") << " etaMuIndexNeg \t" << etaMuIndNeg
                << "\t [ " << (-1.0 * etaMuLowEdge) << ", \t" << (-1.0
                * etaMuHighEdge) << "] ==> etaMuJetEG \t"
                << m_lutEtaMuToCommonCalo[etaMuIndNeg] << "\t [ "
                << m_l1CaloGeometry->globalEtaBinLowEdge(
                        m_lutEtaMuToCommonCalo[etaMuIndNeg]) << ", \t"
                << m_l1CaloGeometry->globalEtaBinLowEdge(
                        m_lutEtaMuToCommonCalo[etaMuIndNeg] + 1) << " ]"
                << std::endl;

    }

    if (m_verbosity && m_isDebugEnabled) {
        LogTrace("L1GlobalTrigger") << std::endl;
        LogTrace("L1GlobalTrigger") << std::endl;
    }

}

// print all the performed conversions
void L1GtEtaPhiConversions::print(std::ostream& myCout) const {

    // force a page break before each group
    myCout << "<p style=\"page-break-before: always\">&nbsp;</p>";

    myCout
            << "\n---++Conversion tables for phi and eta variables of the trigger objects used in correlation conditions \n"
            << std::endl;

    //
    // phi conversions
    //

    // phi Mu -> (*Jet, EG)

    myCout
            << "\n---+++Phi conversion for muons to jets and e-gamma common phi scale \n"
            << std::endl;

    size_t lutPhiMuToJetEgSize = m_lutPhiMuToJetEg.size();
    myCout << "Size of look-up table = " << lutPhiMuToJetEgSize << "\n"
            << std::endl;

    myCout << "|  *Initial Phi Hardware Index*  "
            << "||  *Initial Phi Range*  ||"
            << "  *Converted Phi Hardware Index*  "
            << "||  *Converted Phi Range*  ||" << "\n"
            << "|  *hex*  |  *dec*  | ^|^|  *hex*  |  *dec*  |^|^|"
            << std::endl;

    for (unsigned int indexToConv = 0; indexToConv < lutPhiMuToJetEgSize; ++indexToConv) {

        double lowEdgeToConv = m_l1MuTriggerScales->getPhiScale()->getLowEdge(
                indexToConv);
        double highEdgeToConv =
                m_l1MuTriggerScales->getPhiScale()->getHighEdge(indexToConv);

        unsigned int convIndex = m_lutPhiMuToJetEg[indexToConv];

        double convLowEdge = 0.;
        double convHighEdge = 0.;

        if (convIndex != badIndex) {
            convLowEdge = m_l1CaloGeometry->emJetPhiBinLowEdge(convIndex);
            convHighEdge = m_l1CaloGeometry->emJetPhiBinHighEdge(convIndex);
        } else {
            // badIndex means a bad initialIndex
            lowEdgeToConv = 0.;
            highEdgeToConv = 0.;
        }

        myCout << "|  0x" << std::setw(3) << std::hex << std::left
                << indexToConv << "  |  " << std::dec << std::setw(3)
                << std::left << indexToConv << "  |[ " << std::setw(10)
                << std::left << (rad2deg(lowEdgeToConv)) << ",  |"
                << std::setw(10) << std::left << (rad2deg(highEdgeToConv))
                << " )  |  0x" << std::setw(6) << std::hex << std::left
                << convIndex << " |  " << std::dec << std::setw(6) << convIndex
                << "  |[ " << std::setw(10) << std::left << (rad2deg(
                convLowEdge)) << ",  |" << std::setw(10) << std::left
                << (rad2deg(convHighEdge)) << " )  |" << std::right
                << std::endl;

    }

    // phi Mu -> ETM

    // force a page break before each group
    myCout << "<p style=\"page-break-before: always\">&nbsp;</p>";

    myCout << "\n---+++Phi conversion for muons to ETM phi scale \n"
            << std::endl;

    size_t lutPhiMuToEtmSize = m_lutPhiMuToEtm.size();
    myCout << "Size of look-up table = " << lutPhiMuToEtmSize << "\n"
            << std::endl;

    myCout << "|  *Initial Phi Hardware Index*  "
            << "||  *Initial Phi Range*  ||"
            << "  *Converted Phi Hardware Index*  "
            << "||  *Converted Phi Range*  ||" << "\n"
            << "|  *hex*  |  *dec*  | ^|^|  *hex*  |  *dec*  |^|^|"
            << std::endl;

    for (unsigned int indexToConv = 0; indexToConv < lutPhiMuToEtmSize; ++indexToConv) {

        double lowEdgeToConv = m_l1MuTriggerScales->getPhiScale()->getLowEdge(
                indexToConv);
        double highEdgeToConv =
                m_l1MuTriggerScales->getPhiScale()->getHighEdge(indexToConv);

        unsigned int convIndex = m_lutPhiMuToEtm[indexToConv];

        double convLowEdge = 0.;
        double convHighEdge = 0.;

        if (convIndex != badIndex) {
            convLowEdge = m_l1CaloGeometry->etSumPhiBinLowEdge(convIndex);
            convHighEdge = m_l1CaloGeometry->etSumPhiBinHighEdge(convIndex);
        } else {
            // badIndex means a bad initialIndex
            lowEdgeToConv = 0.;
            highEdgeToConv = 0.;
        }

        myCout << "|  0x" << std::setw(3) << std::hex << std::left
                << indexToConv << "  |  " << std::dec << std::setw(3)
                << std::left << indexToConv << "  |[ " << std::setw(10)
                << std::left << (rad2deg(lowEdgeToConv)) << ",  |"
                << std::setw(10) << std::left << (rad2deg(highEdgeToConv))
                << " )  |  0x" << std::setw(6) << std::hex << std::left
                << convIndex << " |  " << std::dec << std::setw(6) << convIndex
                << "  |[ " << std::setw(10) << std::left << (rad2deg(
                convLowEdge)) << ",  |" << std::setw(10) << std::left
                << (rad2deg(convHighEdge)) << " )  |" << std::right
                << std::endl;

    }

    // phi Mu -> HTM

    // force a page break before each group
    myCout << "<p style=\"page-break-before: always\">&nbsp;</p>";

    myCout << "\n---+++Phi conversion for muons to HTM phi scale \n"
            << std::endl;

    size_t lutPhiMuToHtmSize = m_lutPhiMuToHtm.size();
    myCout << "Size of look-up table = " << lutPhiMuToHtmSize << "\n"
            << std::endl;

    myCout << "|  *Initial Phi Hardware Index*  "
            << "||  *Initial Phi Range*  ||"
            << "  *Converted Phi Hardware Index*  "
            << "||  *Converted Phi Range*  ||" << "\n"
            << "|  *hex*  |  *dec*  | ^|^|  *hex*  |  *dec*  |^|^|"
            << std::endl;

    for (unsigned int indexToConv = 0; indexToConv < lutPhiMuToHtmSize; ++indexToConv) {

        double lowEdgeToConv = m_l1MuTriggerScales->getPhiScale()->getLowEdge(
                indexToConv);
        double highEdgeToConv =
                m_l1MuTriggerScales->getPhiScale()->getHighEdge(indexToConv);

        unsigned int convIndex = m_lutPhiMuToHtm[indexToConv];

        double convLowEdge = 0.;
        double convHighEdge = 0.;

        if (convIndex != badIndex) {
            convLowEdge = m_l1CaloGeometry->htSumPhiBinLowEdge(convIndex);
            convHighEdge = m_l1CaloGeometry->htSumPhiBinHighEdge(convIndex);
        } else {
            // badIndex means a bad initialIndex
            lowEdgeToConv = 0.;
            highEdgeToConv = 0.;
        }

        myCout << "|  0x" << std::setw(3) << std::hex << std::left
                << indexToConv << "  |  " << std::dec << std::setw(3)
                << std::left << indexToConv << "  |[ " << std::setw(10)
                << std::left << (rad2deg(lowEdgeToConv)) << ",  |"
                << std::setw(10) << std::left << (rad2deg(highEdgeToConv))
                << " )  |  0x" << std::setw(6) << std::hex << std::left
                << convIndex << " |  " << std::dec << std::setw(6) << convIndex
                << "  |[ " << std::setw(10) << std::left << (rad2deg(
                convLowEdge)) << ",  |" << std::setw(10) << std::left
                << (rad2deg(convHighEdge)) << " )  |" << std::right
                << std::endl;

    }

    // phi ETM -> (*Jet, EG)

    // force a page break before each group
    myCout << "<p style=\"page-break-before: always\">&nbsp;</p>";

    myCout
            << "\n---+++Phi conversion for ETM to jets and e-gamma scale common phi scale \n"
            << std::endl;

    size_t lutPhiEtmToJetEgSize = m_lutPhiEtmToJetEg.size();
    myCout << "Size of look-up table = " << lutPhiEtmToJetEgSize << "\n"
            << std::endl;

    myCout << "|  *Initial Phi Hardware Index*  "
            << "||  *Initial Phi Range*  ||"
            << "  *Converted Phi Hardware Index*  "
            << "||  *Converted Phi Range*  ||" << "\n"
            << "|  *hex*  |  *dec*  | ^|^|  *hex*  |  *dec*  |^|^|"
            << std::endl;

    for (unsigned int indexToConv = 0; indexToConv < lutPhiEtmToJetEgSize; ++indexToConv) {

        double lowEdgeToConv =
                m_l1CaloGeometry->etSumPhiBinLowEdge(indexToConv);
        double highEdgeToConv = m_l1CaloGeometry->etSumPhiBinHighEdge(
                indexToConv);

        unsigned int convIndex = m_lutPhiEtmToJetEg[indexToConv];

        double convLowEdge = 0.;
        double convHighEdge = 0.;

        if (convIndex != badIndex) {
            convLowEdge = m_l1CaloGeometry->emJetPhiBinLowEdge(convIndex);
            convHighEdge = m_l1CaloGeometry->emJetPhiBinHighEdge(convIndex);
        } else {
            // badIndex means a bad initialIndex
            lowEdgeToConv = 0.;
            highEdgeToConv = 0.;
        }

        myCout << "|  0x" << std::setw(3) << std::hex << std::left
                << indexToConv << "  |  " << std::dec << std::setw(3)
                << std::left << indexToConv << "  |[ " << std::setw(10)
                << std::left << (rad2deg(lowEdgeToConv)) << ",  |"
                << std::setw(10) << std::left << (rad2deg(highEdgeToConv))
                << " )  |  0x" << std::setw(6) << std::hex << std::left
                << convIndex << " |  " << std::dec << std::setw(6) << convIndex
                << "  |[ " << std::setw(10) << std::left << (rad2deg(
                convLowEdge)) << ",  |" << std::setw(10) << std::left
                << (rad2deg(convHighEdge)) << " )  |" << std::right
                << std::endl;

    }

    // phi ETM -> HTM

    // force a page break before each group
    myCout << "<p style=\"page-break-before: always\">&nbsp;</p>";

    myCout << "\n---+++Phi conversion for ETM to HTM phi scale \n" << std::endl;

    size_t lutPhiEtmToHtmSize = m_lutPhiEtmToHtm.size();
    myCout << "Size of look-up table = " << lutPhiEtmToHtmSize << "\n"
            << std::endl;

    myCout << "|  *Initial Phi Hardware Index*  "
            << "||  *Initial Phi Range*  ||"
            << "  *Converted Phi Hardware Index*  "
            << "||  *Converted Phi Range*  ||" << "\n"
            << "|  *hex*  |  *dec*  | ^|^|  *hex*  |  *dec*  |^|^|"
            << std::endl;

    for (unsigned int indexToConv = 0; indexToConv < lutPhiEtmToHtmSize; ++indexToConv) {

        double lowEdgeToConv =
                m_l1CaloGeometry->etSumPhiBinLowEdge(indexToConv);
        double highEdgeToConv = m_l1CaloGeometry->etSumPhiBinHighEdge(
                indexToConv);

        unsigned int convIndex = m_lutPhiEtmToHtm[indexToConv];

        double convLowEdge = 0.;
        double convHighEdge = 0.;

        if (convIndex != badIndex) {
            convLowEdge = m_l1CaloGeometry->htSumPhiBinLowEdge(convIndex);
            convHighEdge = m_l1CaloGeometry->htSumPhiBinHighEdge(convIndex);
        } else {
            // badIndex means a bad initialIndex
            lowEdgeToConv = 0.;
            highEdgeToConv = 0.;
        }

        myCout << "|  0x" << std::setw(3) << std::hex << std::left
                << indexToConv << "  |  " << std::dec << std::setw(3)
                << std::left << indexToConv << "  |[ " << std::setw(10)
                << std::left << (rad2deg(lowEdgeToConv)) << ",  |"
                << std::setw(10) << std::left << (rad2deg(highEdgeToConv))
                << " )  |  0x" << std::setw(6) << std::hex << std::left
                << convIndex << " |  " << std::dec << std::setw(6) << convIndex
                << "  |[ " << std::setw(10) << std::left << (rad2deg(
                convLowEdge)) << ",  |" << std::setw(10) << std::left
                << (rad2deg(convHighEdge)) << " )  |" << std::right
                << std::endl;

    }

    // phi HTM -> (*Jet, EG)


    // force a page break before each group
    myCout << "<p style=\"page-break-before: always\">&nbsp;</p>";

    myCout
            << "\n---+++Phi conversion for HTM to jets and e-gamma scale common phi scale \n"
            << std::endl;

    size_t lutPhiHtmToJetEgSize = m_lutPhiHtmToJetEg.size();
    myCout << "Size of look-up table = " << lutPhiHtmToJetEgSize << "\n"
            << std::endl;

    myCout << "|  *Initial Phi Hardware Index*  "
            << "||  *Initial Phi Range*  ||"
            << "  *Converted Phi Hardware Index*  "
            << "||  *Converted Phi Range*  ||" << "\n"
            << "|  *hex*  |  *dec*  | ^|^|  *hex*  |  *dec*  |^|^|"
            << std::endl;

    for (unsigned int indexToConv = 0; indexToConv < lutPhiHtmToJetEgSize; ++indexToConv) {

        double lowEdgeToConv =
                m_l1CaloGeometry->htSumPhiBinLowEdge(indexToConv);
        double highEdgeToConv = m_l1CaloGeometry->htSumPhiBinHighEdge(
                indexToConv);

        unsigned int convIndex = m_lutPhiHtmToJetEg[indexToConv];

        double convLowEdge = 0.;
        double convHighEdge = 0.;

        if (convIndex != badIndex) {
            convLowEdge = m_l1CaloGeometry->emJetPhiBinLowEdge(convIndex);
            convHighEdge = m_l1CaloGeometry->emJetPhiBinHighEdge(convIndex);
        } else {
            // badIndex means a bad initialIndex
            lowEdgeToConv = 0.;
            highEdgeToConv = 0.;
        }

        myCout << "|  0x" << std::setw(3) << std::hex << std::left
                << indexToConv << "  |  " << std::dec << std::setw(3)
                << std::left << indexToConv << "  |[ " << std::setw(10)
                << std::left << (rad2deg(lowEdgeToConv)) << ",  |"
                << std::setw(10) << std::left << (rad2deg(highEdgeToConv))
                << " )  |  0x" << std::setw(6) << std::hex << std::left
                << convIndex << " |  " << std::dec << std::setw(6) << convIndex
                << "  |[ " << std::setw(10) << std::left << (rad2deg(
                convLowEdge)) << ",  |" << std::setw(10) << std::left
                << (rad2deg(convHighEdge)) << " )  |" << std::right
                << std::endl;

    }

    //
    // eta conversions
    //


    // CenJet/TauJet & IsoEG/NoIsoEG to a common central / forward calorimeter eta scale


    // force a page break before each group
    myCout << "<p style=\"page-break-before: always\">&nbsp;</p>";

    myCout
            << "\n---+++Eta conversion for central and tau jets and e-gamma objects to a common central and forward calorimeter eta scale \n"
            << std::endl;

    size_t lutEtaCentralToCommonCaloSize = m_lutEtaCentralToCommonCalo.size();
    myCout << "Size of look-up table = " << lutEtaCentralToCommonCaloSize
            << "\n" << std::endl;

    myCout << "|  *Initial Eta Hardware Index*  "
            << "||  *Initial Eta Range*  ||"
            << "  *Converted Eta Hardware Index*  "
            << "||  *Converted Eta Range*  ||" << "\n"
            << "|  *hex*  |  *dec*  | ^|^|  *hex*  |  *dec*  |^|^|"
            << std::endl;

    for (unsigned int indexToConv = 0; indexToConv
            < lutEtaCentralToCommonCaloSize; ++indexToConv) {

        double lowEdgeToConv = m_l1CaloGeometry->globalEtaBinLowEdge(
                m_l1CaloGeometry->globalEtaIndex(
                        m_l1CaloGeometry->etaBinCenter(indexToConv, true)));
        double highEdgeToConv = m_l1CaloGeometry->globalEtaBinLowEdge(
                m_l1CaloGeometry->globalEtaIndex(
                        m_l1CaloGeometry->etaBinCenter(indexToConv, true)) + 1);

        unsigned int convIndex = m_lutEtaCentralToCommonCalo[indexToConv];

        double convLowEdge = 0.;
        double convHighEdge = 0.;

        if (convIndex != badIndex) {
            convLowEdge = m_l1CaloGeometry->globalEtaBinLowEdge(convIndex);
            convHighEdge = m_l1CaloGeometry->globalEtaBinLowEdge(convIndex + 1);
        } else {
            // badIndex means a bad initialIndex
            lowEdgeToConv = 0.;
            highEdgeToConv = 0.;
        }

        myCout << "|  0x" << std::setw(3) << std::hex << std::left
                << indexToConv << "  |  " << std::dec << std::setw(3)
                << std::left << indexToConv << "  |[ " << std::setw(10)
                << std::left << lowEdgeToConv << ",  |" << std::setw(10)
                << std::left << highEdgeToConv << " )  |  0x" << std::setw(6)
                << std::hex << std::left << convIndex << " |  " << std::dec
                << std::setw(6) << convIndex << "  |[ " << std::setw(10)
                << std::left << convLowEdge << ",  |" << std::setw(10)
                << std::left << convHighEdge << " )  |" << std::right
                << std::endl;
    }

    // ForJet to a common central / forward calorimeter eta scale


    // force a page break before each group
    myCout << "<p style=\"page-break-before: always\">&nbsp;</p>";

    myCout
            << "\n---+++Eta conversion for forward jets to a common central and forward calorimeter eta scale \n"
            << std::endl;

    size_t lutEtaForJetToCommonCaloSize = m_lutEtaForJetToCommonCalo.size();
    myCout << "Size of look-up table = " << lutEtaForJetToCommonCaloSize
            << "\n" << std::endl;

    myCout << "|  *Initial Eta Hardware Index*  "
            << "||  *Initial Eta Range*  ||"
            << "  *Converted Eta Hardware Index*  "
            << "||  *Converted Eta Range*  ||" << "\n"
            << "|  *hex*  |  *dec*  | ^|^|  *hex*  |  *dec*  |^|^|"
            << std::endl;

    for (unsigned int indexToConv = 0; indexToConv
            < lutEtaForJetToCommonCaloSize; ++indexToConv) {

        double lowEdgeToConv = m_l1CaloGeometry->globalEtaBinLowEdge(
                m_l1CaloGeometry->globalEtaIndex(
                        m_l1CaloGeometry->etaBinCenter(indexToConv, false)));
        double highEdgeToConv =
                m_l1CaloGeometry->globalEtaBinLowEdge(
                        m_l1CaloGeometry->globalEtaIndex(
                                m_l1CaloGeometry->etaBinCenter(indexToConv,
                                        false)) + 1);

        unsigned int convIndex = m_lutEtaForJetToCommonCalo[indexToConv];

        double convLowEdge = 0.;
        double convHighEdge = 0.;

        if (convIndex != badIndex) {
            convLowEdge = m_l1CaloGeometry->globalEtaBinLowEdge(convIndex);
            convHighEdge = m_l1CaloGeometry->globalEtaBinLowEdge(convIndex + 1);
        } else {
            // badIndex means a bad initialIndex
            lowEdgeToConv = 0.;
            highEdgeToConv = 0.;
        }

        myCout << "|  0x" << std::setw(3) << std::hex << std::left
                << indexToConv << "  |  " << std::dec << std::setw(3)
                << std::left << indexToConv << "  |[ " << std::setw(10)
                << std::left << lowEdgeToConv << ",  |" << std::setw(10)
                << std::left << highEdgeToConv << " )  |  0x" << std::setw(6)
                << std::hex << std::left << convIndex << " |  " << std::dec
                << std::setw(6) << convIndex << "  |[ " << std::setw(10)
                << std::left << convLowEdge << ",  |" << std::setw(10)
                << std::left << convHighEdge << " )  |" << std::right
                << std::endl;
    }

    // Mu to a common central / forward calorimeter eta scale

    // force a page break before each group
    myCout << "<p style=\"page-break-before: always\">&nbsp;</p>";

    myCout
            << "\n---+++Eta conversion for muons to a common central and forward calorimeter eta scale \n"
            << std::endl;

    size_t lutEtaMuToCommonCaloSize = m_lutEtaMuToCommonCalo.size();
    myCout << "Size of look-up table = " << lutEtaMuToCommonCaloSize << "\n"
            << std::endl;

    unsigned int nrBinsEtaMuPerHalf =
            m_l1MuTriggerScales->getGMTEtaScale()->getNBins();

    myCout << "|  *Initial Eta Hardware Index*  "
            << "||  *Initial Eta Range*  ||"
            << "  *Converted Eta Hardware Index*  "
            << "||  *Converted Eta Range*  ||" << "\n"
            << "|  *hex*  |  *dec*  | ^|^|  *hex*  |  *dec*  |^|^|"
            << std::endl;

    for (unsigned int indexToConv = 0; indexToConv < lutEtaMuToCommonCaloSize; ++indexToConv) {

        // Mu scale defined for positive values only, need to be symmetrized
        unsigned int iBinOffset = 0;
        double etaSign = 1.;

        if (indexToConv > nrBinsEtaMuPerHalf) {
            iBinOffset = nrBinsEtaMuPerHalf + 1;
            etaSign = -1.;
        }

        double lowEdgeToConv = etaSign
                * m_l1MuTriggerScales->getGMTEtaScale()->getValue(
                        indexToConv - iBinOffset);
        double highEdgeToConv = etaSign
                * m_l1MuTriggerScales->getGMTEtaScale()->getValue(
                        indexToConv + 1 - iBinOffset);

        unsigned int convIndex = m_lutEtaMuToCommonCalo[indexToConv];

        double convLowEdge = 0.;
        double convHighEdge = 0.;

        if (convIndex != badIndex) {
            convLowEdge = m_l1CaloGeometry->globalEtaBinLowEdge(convIndex);
            convHighEdge = m_l1CaloGeometry->globalEtaBinLowEdge(convIndex + 1);
        } else {
            // badIndex means a bad initialIndex
            lowEdgeToConv = 0.;
            highEdgeToConv = 0.;
        }

        myCout << "|  0x" << std::setw(3) << std::hex << std::left
                << indexToConv << "  |  " << std::dec << std::setw(3)
                << std::left << indexToConv << "  |[ " << std::setw(10)
                << std::left << lowEdgeToConv << ",  |" << std::setw(10)
                << std::left << highEdgeToConv << " )  |  0x" << std::setw(6)
                << std::hex << std::left << convIndex << " |  " << std::dec
                << std::setw(6) << convIndex << "  |[ " << std::setw(10)
                << std::left << convLowEdge << ",  |" << std::setw(10)
                << std::left << convHighEdge << " )  |" << std::right
                << std::endl;
    }

}

// convert phi from rad (-pi, pi] to deg (0, 360)
const double L1GtEtaPhiConversions::rad2deg(const double& phiRad) const {

    if (phiRad < 0.) {
        return (phiRad * PiConversion) + 360.;
    } else {
        return (phiRad * PiConversion);
    }
}

// static members

const unsigned int L1GtEtaPhiConversions::badIndex = 999999;
const double L1GtEtaPhiConversions::PiConversion = 180. / acos(-1.);

