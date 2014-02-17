#ifndef L1GtConfigProducers_L1GtXmlParserTags_h
#define L1GtConfigProducers_L1GtXmlParserTags_h

/**
 * \class L1GtXmlParserTags
 *
 *
 * Description: Tags for the Xerces-C XML parser for the L1 Trigger menu.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 *
 * $Date: 2009/10/29 16:37:58 $
 * $Revision: 1.9 $
 *
 */

// system include files
#include <string>

// user include files

// forward declarations

// class declaration
class L1GtXmlParserTags
{

public:

    /// constructor
    L1GtXmlParserTags();

    /// destructor
    virtual ~L1GtXmlParserTags();

protected:

    // strings for the def.xml-syntax
    static const std::string m_xmlTagDef;
    static const std::string m_xmlTagHeader;

    static const std::string m_xmlTagMenuInterface;
    static const std::string m_xmlTagMenuInterfaceDate;
    static const std::string m_xmlTagMenuInterfaceAuthor;
    static const std::string m_xmlTagMenuInterfaceDescription;

    static const std::string m_xmlTagMenuDate;
    static const std::string m_xmlTagMenuAuthor;
    static const std::string m_xmlTagMenuDescription;

    static const std::string m_xmlTagMenuAlgImpl;

    static const std::string m_xmlTagScaleDbKey;

    static const std::string m_xmlTagChip;
    static const std::string m_xmlTagConditions;
    static const std::string m_xmlTagAlgorithms;
    static const std::string m_xmlTagTechTriggers;

    static const std::string m_xmlAlgorithmAttrAlias;

    static const std::string m_xmlConditionAttrCondition;
    static const std::string m_xmlConditionAttrObject;
    static const std::string m_xmlConditionAttrType;
    static const std::string m_xmlConditionAttrConditionMuon;
    static const std::string m_xmlConditionAttrConditionCalo;
    static const std::string m_xmlConditionAttrConditionEnergySum;
    static const std::string m_xmlConditionAttrConditionJetCounts;
    static const std::string m_xmlConditionAttrConditionCastor;
    static const std::string m_xmlConditionAttrConditionHfBitCounts;
    static const std::string m_xmlConditionAttrConditionHfRingEtSums;
    static const std::string m_xmlConditionAttrConditionCorrelation;
    static const std::string m_xmlConditionAttrConditionBptx;
    static const std::string m_xmlConditionAttrConditionExternal;

    static const std::string m_xmlConditionAttrObjectMu;
    static const std::string m_xmlConditionAttrObjectNoIsoEG;
    static const std::string m_xmlConditionAttrObjectIsoEG;
    static const std::string m_xmlConditionAttrObjectCenJet;
    static const std::string m_xmlConditionAttrObjectForJet;
    static const std::string m_xmlConditionAttrObjectTauJet;
    static const std::string m_xmlConditionAttrObjectETM;
    static const std::string m_xmlConditionAttrObjectETT;
    static const std::string m_xmlConditionAttrObjectHTT;
    static const std::string m_xmlConditionAttrObjectHTM;
    static const std::string m_xmlConditionAttrObjectJetCounts;
    static const std::string m_xmlConditionAttrObjectCastor;
    static const std::string m_xmlConditionAttrObjectHfBitCounts;
    static const std::string m_xmlConditionAttrObjectHfRingEtSums;
    static const std::string m_xmlConditionAttrObjectBptx;
    static const std::string m_xmlConditionAttrObjectGtExternal;

    static const std::string m_xmlConditionAttrType1s;
    static const std::string m_xmlConditionAttrType2s;
    static const std::string m_xmlConditionAttrType2wsc;
    static const std::string m_xmlConditionAttrType2cor;
    static const std::string m_xmlConditionAttrType3s;
    static const std::string m_xmlConditionAttrType4s;
    static const std::string m_xmlConditionAttrTypeCastor;
    static const std::string m_xmlConditionAttrTypeBptx;
    static const std::string m_xmlConditionAttrTypeExternal;


    static const std::string m_xmlAttrMode;
    static const std::string m_xmlAttrModeBit;
    static const std::string m_xmlAttrMax;

    static const std::string m_xmlAttrNr;
    static const std::string m_xmlAttrPin;
    static const std::string m_xmlAttrPinA;

    static const std::string m_xmlTagPtHighThreshold;
    static const std::string m_xmlTagPtLowThreshold;
    static const std::string m_xmlTagQuality;
    static const std::string m_xmlTagEta;
    static const std::string m_xmlTagPhi;
    static const std::string m_xmlTagPhiHigh;
    static const std::string m_xmlTagPhiLow;
    static const std::string m_xmlTagChargeCorrelation;
    static const std::string m_xmlTagEnableMip;
    static const std::string m_xmlTagEnableIso;
    static const std::string m_xmlTagRequestIso;
    static const std::string m_xmlTagDeltaEta;
    static const std::string m_xmlTagDeltaPhi;

    static const std::string m_xmlTagEtThreshold;
    static const std::string m_xmlTagEnergyOverflow;

    static const std::string m_xmlTagCountThreshold;
    static const std::string m_xmlTagCountOverflow;

    static const std::string m_xmlTagOutput;
    static const std::string m_xmlTagOutputPin;

    static const std::string m_xmlTagGEq;
    static const std::string m_xmlTagValue;

    static const std::string m_xmlTagChipDef;
    static const std::string m_xmlTagChip1;
    static const std::string m_xmlTagCa;

    // strings for the vme xml file syntax
    static const std::string m_xmlTagVme;
    static const std::string m_xmlTagVmeAddress;

};

#endif /*L1GtConfigProducers_L1GtXmlParserTags_h*/
