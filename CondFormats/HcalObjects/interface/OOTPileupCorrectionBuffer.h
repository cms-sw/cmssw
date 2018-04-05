#ifndef CondFormats_HcalObjects_OOTPileupCorrectionBuffer_h
#define CondFormats_HcalObjects_OOTPileupCorrectionBuffer_h

//
// A simple storage buffer for Hcal OOT pileup corrections.
//
// I. Volobouev, 08/29/2014
//

#include "CondFormats/Serialization/interface/Serializable.h"

#include <string>

class OOTPileupCorrectionBuffer
{
public:
    // Constructors
    inline OOTPileupCorrectionBuffer() {}
    inline explicit OOTPileupCorrectionBuffer(const std::string& s) 
        : m_buffer(s) {}
    inline OOTPileupCorrectionBuffer(const char* c, std::size_t len)
        : m_buffer(c, len) {}
    inline explicit OOTPileupCorrectionBuffer(std::size_t len) 
        : m_buffer(len, '\0') {}

    // Inspectors
    inline const std::string& str() const {return m_buffer;}
    inline std::size_t length() const {return m_buffer.size();}
    inline bool empty() const {return m_buffer.empty();}
    inline const char* getBuffer() const
        {return m_buffer.empty() ? static_cast<const char*>(nullptr) : &m_buffer[0];}

    // Modifiers
    inline char* getBuffer() 
        {return m_buffer.empty() ? static_cast<char*>(nullptr) : &m_buffer[0];}
    inline void setStr(const std::string& s) {m_buffer = s;}

private:
    std::string m_buffer;

    COND_SERIALIZABLE;
};

#endif // CondFormats_HcalObjects_OOTPileupCorrectionBuffer_h
