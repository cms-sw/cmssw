#ifndef GlobalTriggerAnalyzer_L1GtTrigReportEntry_h
#define GlobalTriggerAnalyzer_L1GtTrigReportEntry_h

/**
 * \class L1GtTrigReportEntry
 * 
 * 
 * Description: an individual L1 GT report entry.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 * 
 * $Date$
 * $Revision$
 *
 */

// system include files
#include <string>

// class declaration

class L1GtTrigReportEntry
{

public:

    /// constructor
    explicit L1GtTrigReportEntry(const std::string& menuName, const std::string& algName,
        const int prescaleFactor, const int triggerMask);

    /// destructor
    virtual ~L1GtTrigReportEntry();

public:

    /// assignment operator
    L1GtTrigReportEntry& operator=(const L1GtTrigReportEntry&);

    /// equal operator
    bool operator==(const L1GtTrigReportEntry&) const;

    /// unequal operator
    bool operator!=(const L1GtTrigReportEntry&) const;


public:

    /// get the trigger menu name
    inline const std::string gtTriggerMenuName() const {
        return m_triggerMenuName;
    }

    ///  get the algorithm name
    inline const std::string gtAlgoName() const {
        return m_algoName;
    }

    ///  get the prescale factor
    inline const int gtPrescaleFactor() const {
        return m_prescaleFactor;
    }

    ///  get the trigger mask
    inline const unsigned int gtTriggerMask() const {
        return m_triggerMask;
    }

    ///  get the number of events accepted for this entry
    inline const int gtNrEventsAccept() const {
        return m_nrEventsAccept;
    }

    /// get the number of events rejected for this entry
    inline const int gtNrEventsReject() const {
        return m_nrEventsReject;
    }

    /// get the number of events with error for this entry
    inline const int gtNrEventsError() const {
        return m_nrEventsError;
    }

public:

    /// increase # of events accepted/rejected for this entry
    void addValidEntry(const bool algResult);

    /// increase # of events with error 
    void addErrorEntry();

private:

    /// menu name 
    std::string m_triggerMenuName;

    /// algorithm name
    std::string m_algoName;

    /// prescale factor
    int m_prescaleFactor;

    /// trigger mask
    unsigned int m_triggerMask;

    /// counters

    /// number of events accepted for this entry
    int m_nrEventsAccept;

    /// number of events rejected for this entry
    int m_nrEventsReject;

    /// number of events with error
    int m_nrEventsError;

};

#endif /*GlobalTriggerAnalyzer_L1GtTrigReportEntry_h*/
