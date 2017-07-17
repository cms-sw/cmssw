#ifndef L1Trigger_L1TGlobal_ExternalTemplate_h
#define L1Trigger_L1TGlobal_ExternalTemplate_h

/**
 * \class ExternalTemplate
 *
 *
 * Description: L1 Global Trigger energy-sum template.
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
#include <iosfwd>

// user include files

//   base class
#include "L1Trigger/L1TGlobal/interface/GlobalCondition.h"

// forward declarations

// class declaration
class ExternalTemplate : public GlobalCondition
{

public:

    // constructor
    ExternalTemplate();

    // constructor
    ExternalTemplate(const std::string&);

    // constructor
    ExternalTemplate(const std::string&, const l1t::GtConditionType&);

    // copy constructor
    ExternalTemplate(const ExternalTemplate&);

    // destructor
    virtual ~ExternalTemplate();

    // assign operator
    ExternalTemplate& operator=(const ExternalTemplate&);

public:




public:

    /// get external channel number
    inline const unsigned int& extChannel() const
    {
        return m_extChannel;
    }

    /// set functions
    inline void setExternalChannel(unsigned int extCh) { m_extChannel = extCh; }

    /// print the condition
    virtual void print(std::ostream& myCout) const;

    /// output stream operator
    friend std::ostream& operator<<(std::ostream&, const ExternalTemplate&);

private:

    /// copy function for copy constructor and operator=
    void copy(const ExternalTemplate& cp);

private:

    //Channel for this signal
    unsigned int m_extChannel;
    

};

#endif
