//
// A simple storage buffer for FFTJet corrections data.
//
// I. Volobouev, 07/31/2012
//

#ifndef CondFormats_JetMETObjects_FFTJetCorrectorParameters_h
#define CondFormats_JetMETObjects_FFTJetCorrectorParameters_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include <string>

class FFTJetCorrectorParameters
{
public:
    // Constructors
    inline FFTJetCorrectorParameters() {}
    inline explicit FFTJetCorrectorParameters(const std::string& s) 
        : m_buffer(s) {}
    inline FFTJetCorrectorParameters(const char* c, std::size_t len)
        : m_buffer(c, len) {}
    inline explicit FFTJetCorrectorParameters(std::size_t len) 
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

    COND_SERIALIZABLE;
};

#endif // CondFormats_JetMETObjects_FFTJetCorrectorParameters_h
