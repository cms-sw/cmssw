#ifndef CondFormats_HcalObjects_HcalOOTPileupCorrectionData_h
#define CondFormats_HcalObjects_HcalOOTPileupCorrectionData_h

//
// A storage buffer for HCAL OOT pileup corrections
//
// I. Volobouev, 04/03/2014
//

#include <string>

class HcalOOTPileupCorrectionData
{
public:
    // Constructors
    inline HcalOOTPileupCorrectionData() {}
    inline explicit HcalOOTPileupCorrectionData(const std::string& s) 
        : m_buffer(s) {}
    inline HcalOOTPileupCorrectionData(const char* c, std::size_t len)
        : m_buffer(c, len) {}
    inline explicit HcalOOTPileupCorrectionData(std::size_t len) 
        : m_buffer(len, '\0') {}

    // Inspectors
    inline const std::string& str() const {return m_buffer;}
    inline std::size_t length() const {return m_buffer.size();}
    inline bool empty() const {return m_buffer.empty();}
    inline const char* getBuffer() const
        {return m_buffer.empty() ? static_cast<const char*>(0) : &m_buffer[0];}

    // Modifiers
    inline char* getBuffer() 
        {return m_buffer.empty() ? static_cast<char*>(0) : &m_buffer[0];}
    inline void setStr(const std::string& s) {m_buffer = s;}

private:
    std::string m_buffer;
};

#endif // CondFormats_HcalObjects_HcalOOTPileupCorrectionData_h
