#ifndef GlobalTrigger_L1GtEtaPhiConversions_h
#define GlobalTrigger_L1GtEtaPhiConversions_h

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

// system include files
#include <iostream>
#include <vector>
#include <utility>

// user include files

//   base class

//
#include "DataFormats/L1GlobalTrigger/interface/L1GtObject.h"

// forward declarations
class L1CaloGeometry;
class L1MuTriggerScales;

// class interface
class L1GtEtaPhiConversions {

public:

    /// constructor
    L1GtEtaPhiConversions();

    /// destructor
    virtual ~L1GtEtaPhiConversions();

public:

    /// return the index of a pair in the vector m_gtObjectPairVec, to be used to extract
    /// the number of phi bins, the conversion of the indices, etc
    const unsigned int
            gtObjectPairIndex(const L1GtObject&, const L1GtObject&) const;

    /// convert the phi index initialIndex for an object from pair pairIndex, with position of
    /// object in pair positionPair to common scale for the L1GtObject pair
    const unsigned int convertPhiIndex(unsigned int pairIndex,
            unsigned int positionPair, unsigned int initialIndex) const ;

    /// convert the eta index for an object to common scale for the L1GtObject pair
    const unsigned int convertEtaIndex(unsigned int pairIndex,
            unsigned int initialIndex) const;

    /// return the number of phi bins for a GT object
    const unsigned int gtObjectNrBinsPhi(const L1GtObject&) const;

    /// return the number of phi bins for a pair of GT objects, according to conversion rules
    const unsigned int
            gtObjectNrBinsPhi(const L1GtObject&, const L1GtObject&) const;

    /// return the number of phi bins for a pair of GT objects, according to conversion rules,
    /// when the index of the pair is used
    const unsigned int gtObjectNrBinsPhi(const unsigned int) const;

    /// perform all conversions
    void convert(const L1CaloGeometry*, const L1MuTriggerScales*, const int,
            const int);

    inline void setVerbosity(const int verbosity) {
        m_verbosity = verbosity;
    }

    /// print all the performed conversions
    virtual void print(std::ostream& myCout) const;

private:

    /// vector of all L1GtObject pairs
    std::vector<std::pair<L1GtObject, L1GtObject> > m_gtObjectPairVec;

    /// decide which object to convert:
    /// if m_pairConvertPhiFirstGtObject true, convert pair.first and do not convert pair.second
    /// if m_pairConvertPhiFirstGtObject false, do not convert pair.first and convert pair.second
    std::vector<bool> m_pairConvertPhiFirstGtObject;

    /// number of phi bins for each L1GtObject pair in the scale used for that pair
    /// it is filled correlated with m_gtObjectPairVec, so the index of the pair in
    /// m_gtObjectPairVec is the index of the m_pairNrPhiBinsVec element containing
    /// the number of phi bins
    std::vector<const unsigned int*> m_pairNrPhiBinsVec;

    /// constant references to conversion LUT for a given L1GtObject pair
    /// it is filled correlated with m_gtObjectPairVec, so the index of the pair in
    /// m_gtObjectPairVec is the index of the m_pairPhiConvVec element containing the
    /// reference
    std::vector<const std::vector<unsigned int>* > m_pairPhiConvVec;

private:

    /// number of phi bins for muons
    unsigned int m_nrBinsPhiMu;

    /// number of phi bins for calorimeter objects (*Jet, *EG)
    unsigned int m_nrBinsPhiJetEg;

    /// number of phi bins for ETM
    unsigned int m_nrBinsPhiEtm;

    /// number of phi bins for HTM
    unsigned int m_nrBinsPhiHtm;

private:

    /// phi conversion for Mu to (*Jet, EG)
    std::vector<unsigned int> m_lutPhiMuToJetEg;

    /// phi conversion for Mu to ETM
    std::vector<unsigned int> m_lutPhiMuToEtm;

    /// phi conversion for Mu to HTM
    std::vector<unsigned int> m_lutPhiMuToHtm;

    /// phi conversion for ETM to (*Jet, EG)
    std::vector<unsigned int> m_lutPhiEtmToJetEg;

    /// phi conversion for ETM to HTM
    std::vector<unsigned int> m_lutPhiEtmToHtm;

    /// phi conversion for HTM to (*Jet, EG)
    std::vector<unsigned int> m_lutPhiHtmToJetEg;

    /// phi conversion for (*Jet, EG) to (*Jet, EG)
    /// return the same index as the input index, introduced only
    /// to simplify convertPhiIndex
    std::vector<unsigned int> m_lutPhiJetEgToJetEg;

private:

    /// eta conversion of calorimeter object to a common scale
    std::vector<unsigned int> m_lutEtaCenCaloCommon;

private:

    /// verbosity level
    int m_verbosity;

    /// cached edm::isDebugEnabled()
    bool m_isDebugEnabled;

};

#endif
