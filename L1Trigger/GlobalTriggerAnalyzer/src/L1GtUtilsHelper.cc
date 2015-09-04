#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtUtilsHelper.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/BranchType.h"
#include "FWCore/Utilities/interface/TypeID.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"


L1GtUtilsHelper::L1GtUtilsHelper(edm::ParameterSet const& pset,
        edm::ConsumesCollector& iC,
        bool useL1GtTriggerMenuLite) :
            m_consumesCollector(std::move(iC)),
            m_l1GtRecordInputTag(pset.getParameter<edm::InputTag>("l1GtRecordInputTag")),
            m_l1GtReadoutRecordInputTag(pset.getParameter<edm::InputTag>("l1GtReadoutRecordInputTag")),
            m_l1GtTriggerMenuLiteInputTag(pset.getParameter<edm::InputTag>("l1GtTriggerMenuLiteInputTag")),
            m_findRecord(false),
            m_findReadoutRecord(false),
            m_findMenuLite(false),
            m_foundPreferredRecord(false),
            m_foundPreferredReadoutRecord(false),
            m_foundPreferredMenuLite(false) {

    m_l1GtRecordToken = iC.consumes<L1GlobalTriggerRecord>(
            m_l1GtRecordInputTag);
    m_l1GtReadoutRecordToken = iC.consumes<L1GlobalTriggerReadoutRecord>(
            m_l1GtReadoutRecordInputTag);
    if (useL1GtTriggerMenuLite) {
        m_l1GtTriggerMenuLiteToken =
                iC.consumes<L1GtTriggerMenuLite, edm::InRun>(
                        m_l1GtTriggerMenuLiteInputTag);
    }
}

void L1GtUtilsHelper::fillDescription(edm::ParameterSetDescription & desc) {
    desc.add<edm::InputTag>("l1GtRecordInputTag", edm::InputTag());
    desc.add<edm::InputTag>("l1GtReadoutRecordInputTag", edm::InputTag());
    desc.add<edm::InputTag>("l1GtTriggerMenuLiteInputTag", edm::InputTag());
}

void L1GtUtilsHelper::operator()(
        edm::BranchDescription const& branchDescription) {

    // This is only used if required InputTags were not specified already.
    // This is called early in the process, once for each product in the ProductRegistry.
    // The callback is registered when callWhenNewProductsRegistered is called.
    // It finds products by type and sets the token so that it can be used
    // later when getting the product.

    // The code will look for the corresponding product in ProductRegistry.
    // If the product is found, it checks the product label in
    // a vector of preferred input tags (hardwired now to "gtDigis" and
    // "hltGtDigis"). The first input tag from the vector of preferred input tags, with the
    // same label as the input tag found from provenance, is kept as input tag, if there are no
    // multiple products with the same label.

    // If multiple products are found and no one has a label in the vector of preferred input tags,
    // or if multiple products are found with the label in the vector of preferred input tags
    // (with different instance or process) the input tag is set to empty input tag, and L1GtUtil
    // will produce an error, as it is not possible to safely choose a product. In this case, one must
    // provide explicitly the correct input tag via configuration or in the constructor.

    // TODO decide if the preferred input tags must be given as input parameters
    // or stay hardwired

    std::vector<edm::InputTag> preferredL1GtRecordInputTag = { edm::InputTag(
            "gtDigis"), edm::InputTag("hltGtDigis") };

    std::vector<edm::InputTag> preferredL1GtReadoutRecordInputTag = {
            edm::InputTag("gtDigis"), edm::InputTag("hltGtDigis") };

    std::vector<edm::InputTag> preferredL1GtTriggerMenuLiteInputTag = {
            edm::InputTag("gtDigis"), edm::InputTag("hltGtDigis") };

    // L1GlobalTriggerRecord

    if (m_findRecord && (!m_foundMultipleL1GtRecord)
            && (branchDescription.unwrappedTypeID()
                    == edm::TypeID(typeid(L1GlobalTriggerRecord)))
            && (branchDescription.branchType() == edm::InEvent)) {

        edm::InputTag tag { branchDescription.moduleLabel(),
                branchDescription.productInstanceName(),
                branchDescription.processName() };

        if (m_foundPreferredRecord) {

            // check if a preferred input tag was already found and compare it with the actual tag
            // if the instance or the process names are different, one has incompatible tags - set
            // the tag to empty input tag and indicate that multiple preferred input tags are found
            // so it is not possibly to choose safely an input tag

            if ((m_l1GtRecordInputTag.label() == branchDescription.moduleLabel())
                    && ((m_l1GtRecordInputTag.instance()
                            != branchDescription.productInstanceName())
                            || (m_l1GtRecordInputTag.process()
                                    != branchDescription.processName()))) {

                LogDebug("L1GtUtils")
                        << "\nWARNING: Found multiple preferred input tags for L1GlobalTriggerRecord product, "
                        << "\nwith different instaces or processes."
                        << "\nInput tag already found: "
                        << (m_l1GtRecordInputTag) << "\nActual tag: " << (tag)
                        << "\nInput tag set to empty tag." << std::endl;

                m_foundMultipleL1GtRecord = true;
                m_l1GtRecordInputTag = edm::InputTag();
            }
        } else {
            // no preferred input tag found yet, check now with the actual tag

            for (std::vector<edm::InputTag>::const_iterator itPrefTag =
                    preferredL1GtRecordInputTag.begin(), itPrefTagEnd =
                    preferredL1GtRecordInputTag.end();
                    itPrefTag != itPrefTagEnd; ++itPrefTag) {

                if (branchDescription.moduleLabel() == itPrefTag->label()) {
                    m_l1GtRecordInputTag = tag;
                    m_l1GtRecordToken = m_consumesCollector.consumes<
                            L1GlobalTriggerRecord>(tag);
                    m_foundPreferredRecord = true;
                    m_inputTagsL1GtRecord.push_back(tag);

                    LogDebug("L1GtUtils")
                            << "\nWARNING: Input tag for L1GlobalTriggerRecord product set to preferred input tag"
                            << (tag) << std::endl;
                    break;
                }
            }
        }

        if (!m_foundPreferredRecord) {

            // check if other input tag was found - if true, there are multiple input tags in the event,
            // none in the preferred input tags, so it is not possibly to choose safely an input tag

            if (m_inputTagsL1GtRecord.size() > 1) {

                LogDebug("L1GtUtils")
                        << "\nWARNING: Found multiple input tags for L1GlobalTriggerRecord product."
                        << "\nNone is in the preferred input tags - no safe choice."
                        << "\nInput tag already found: "
                        << (m_l1GtRecordInputTag) << "\nActual tag: " << (tag)
                        << "\nInput tag set to empty tag." << std::endl;
                m_l1GtRecordInputTag = edm::InputTag();
                m_foundMultipleL1GtRecord = true;

            } else {
                if (m_l1GtRecordToken.isUninitialized()) {

                    m_l1GtRecordInputTag = tag;
                    m_inputTagsL1GtRecord.push_back(tag);
                    m_l1GtRecordToken = m_consumesCollector.consumes<
                            L1GlobalTriggerRecord>(tag);

                    LogDebug("L1GtUtils")
                            << "\nWARNING: No preferred input tag found for L1GlobalTriggerReadoutRecord product."
                            << "\nInput tag set to " << (tag) << std::endl;
                }
            }
        }
    }


    // L1GlobalTriggerReadoutRecord

    if (m_findReadoutRecord && (!m_foundMultipleL1GtReadoutRecord)
            && (branchDescription.unwrappedTypeID()
                    == edm::TypeID(typeid(L1GlobalTriggerReadoutRecord)))
            && (branchDescription.branchType() == edm::InEvent)) {

        edm::InputTag tag { branchDescription.moduleLabel(),
                branchDescription.productInstanceName(),
                branchDescription.processName() };

        if (m_foundPreferredReadoutRecord) {

            // check if a preferred input tag was already found and compare it with the actual tag
            // if the instance or the process names are different, one has incompatible tags - set
            // the tag to empty input tag and indicate that multiple preferred input tags are found
            // so it is not possibly to choose safely an input tag

            if ((m_l1GtReadoutRecordInputTag.label()
                    == branchDescription.moduleLabel())
                    && ((m_l1GtReadoutRecordInputTag.instance()
                            != branchDescription.productInstanceName())
                            || (m_l1GtReadoutRecordInputTag.process()
                                    != branchDescription.processName()))) {

                LogDebug("L1GtUtils")
                        << "\nWARNING: Found multiple preferred input tags for L1GlobalTriggerReadoutRecord product, "
                        << "\nwith different instaces or processes."
                        << "\nInput tag already found: "
                        << (m_l1GtReadoutRecordInputTag) << "\nActual tag: "
                        << (tag) << "\nInput tag set to empty tag."
                        << std::endl;

                m_foundMultipleL1GtReadoutRecord = true;
                m_l1GtReadoutRecordInputTag = edm::InputTag();
            }
        } else {
            // no preferred input tag found yet, check now with the actual tag

            for (std::vector<edm::InputTag>::const_iterator itPrefTag =
                    preferredL1GtReadoutRecordInputTag.begin(), itPrefTagEnd =
                    preferredL1GtReadoutRecordInputTag.end();
                    itPrefTag != itPrefTagEnd; ++itPrefTag) {

                if (branchDescription.moduleLabel() == itPrefTag->label()) {
                    m_l1GtReadoutRecordInputTag = tag;
                    m_l1GtReadoutRecordToken = m_consumesCollector.consumes<
                            L1GlobalTriggerReadoutRecord>(tag);
                    m_foundPreferredReadoutRecord = true;
                    m_inputTagsL1GtReadoutRecord.push_back(tag);

                    LogDebug("L1GtUtils")
                            << "\nWARNING: Input tag for L1GlobalTriggerReadoutRecord product set to preferred input tag"
                            << (tag) << std::endl;
                    break;
                }
            }
        }

        if (!m_foundPreferredReadoutRecord) {

            // check if other input tag was found - if true, there are multiple input tags in the event,
            // none in the preferred input tags, so it is not possibly to choose safely an input tag

            if (m_inputTagsL1GtReadoutRecord.size() > 1) {

                LogDebug("L1GtUtils")
                        << "\nWARNING: Found multiple input tags for L1GlobalTriggerReadoutRecord product."
                        << "\nNone is in the preferred input tags - no safe choice."
                        << "\nInput tag already found: "
                        << (m_l1GtReadoutRecordInputTag) << "\nActual tag: "
                        << (tag) << "\nInput tag set to empty tag."
                        << std::endl;
                m_l1GtReadoutRecordInputTag = edm::InputTag();
                m_foundMultipleL1GtReadoutRecord = true;

            } else {
                if (m_l1GtReadoutRecordToken.isUninitialized()) {

                    m_l1GtReadoutRecordInputTag = tag;
                    m_inputTagsL1GtReadoutRecord.push_back(tag);
                    m_l1GtReadoutRecordToken = m_consumesCollector.consumes<
                            L1GlobalTriggerReadoutRecord>(tag);

                    LogDebug("L1GtUtils")
                            << "\nWARNING: No preferred input tag found for L1GlobalTriggerReadoutRecord product."
                            << "\nInput tag set to " << (tag) << std::endl;
                }
            }
        }
    }




    // L1GtTriggerMenuLite

    if (m_findMenuLite && (!m_foundMultipleL1GtMenuLite)
            && (branchDescription.unwrappedTypeID()
                    == edm::TypeID(typeid(L1GtTriggerMenuLite)))
            && (branchDescription.branchType() == edm::InEvent)) {

        edm::InputTag tag { branchDescription.moduleLabel(),
                branchDescription.productInstanceName(),
                branchDescription.processName() };

        if (m_foundPreferredMenuLite) {

            // check if a preferred input tag was already found and compare it with the actual tag
            // if the instance or the process names are different, one has incompatible tags - set
            // the tag to empty input tag and indicate that multiple preferred input tags are found
            // so it is not possibly to choose safely an input tag

            if ((m_l1GtTriggerMenuLiteInputTag.label()
                    == branchDescription.moduleLabel())
                    && ((m_l1GtTriggerMenuLiteInputTag.instance()
                            != branchDescription.productInstanceName())
                            || (m_l1GtTriggerMenuLiteInputTag.process()
                                    != branchDescription.processName()))) {

                LogDebug("L1GtUtils")
                        << "\nWARNING: Found multiple preferred input tags for L1GtTriggerMenuLite product, "
                        << "\nwith different instaces or processes."
                        << "\nInput tag already found: "
                        << (m_l1GtTriggerMenuLiteInputTag) << "\nActual tag: " << (tag)
                        << "\nInput tag set to empty tag." << std::endl;

                m_foundMultipleL1GtMenuLite = true;
                m_l1GtTriggerMenuLiteInputTag = edm::InputTag();
            }
        } else {
            // no preferred input tag found yet, check now with the actual tag

            for (std::vector<edm::InputTag>::const_iterator itPrefTag =
                    preferredL1GtTriggerMenuLiteInputTag.begin(), itPrefTagEnd =
                    preferredL1GtTriggerMenuLiteInputTag.end();
                    itPrefTag != itPrefTagEnd; ++itPrefTag) {

                if (branchDescription.moduleLabel() == itPrefTag->label()) {
                    m_l1GtTriggerMenuLiteInputTag = tag;
                    m_l1GtTriggerMenuLiteToken = m_consumesCollector.consumes<
                            L1GtTriggerMenuLite>(tag);
                    m_foundPreferredMenuLite = true;
                    m_inputTagsL1GtMenuLite.push_back(tag);

                    LogDebug("L1GtUtils")
                            << "\nWARNING: Input tag for L1GtTriggerMenuLite product set to preferred input tag"
                            << (tag) << std::endl;
                    break;
                }
            }
        }

        if (!m_foundPreferredMenuLite) {

            // check if other input tag was found - if true, there are multiple input tags in the event,
            // none in the preferred input tags, so it is not possibly to choose safely an input tag

            if (m_inputTagsL1GtMenuLite.size() > 1) {

                LogDebug("L1GtUtils")
                        << "\nWARNING: Found multiple input tags for L1GtTriggerMenuLite product."
                        << "\nNone is in the preferred input tags - no safe choice."
                        << "\nInput tag already found: "
                        << (m_l1GtTriggerMenuLiteInputTag) << "\nActual tag: " << (tag)
                        << "\nInput tag set to empty tag." << std::endl;
                m_l1GtTriggerMenuLiteInputTag = edm::InputTag();
                m_foundMultipleL1GtMenuLite = true;

            } else {
                if (m_l1GtTriggerMenuLiteToken.isUninitialized()) {

                    m_l1GtTriggerMenuLiteInputTag = tag;
                    m_inputTagsL1GtMenuLite.push_back(tag);
                    m_l1GtTriggerMenuLiteToken = m_consumesCollector.consumes<
                            L1GtTriggerMenuLite>(tag);

                    LogDebug("L1GtUtils")
                            << "\nWARNING: No preferred input tag found for L1GtTriggerMenuLite product."
                            << "\nInput tag set to " << (tag) << std::endl;
                }
            }
        }
    }
}
