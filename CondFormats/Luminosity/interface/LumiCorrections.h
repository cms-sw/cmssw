#ifndef LUMICORRECTIONS_H
#define LUMICORRECTIONS_H


/*
*Author: Sam Higginbotham
*Purpose: to save the corrections for the luminosity to the database 
*
*/
#include <sstream>
#include <cstring>
#include <vector>
#include <boost/serialization/vector.hpp>
#include "CondFormats/Serialization/interface/Serializable.h"

class LumiCorrections {
    public:
        void  setOverallCorrection(float overallCorrection){m_overallCorrection=overallCorrection;}
        void  setType1Fraction(float type1frac){m_type1Fraction=type1frac;}
        void  setType1Residual(float type1res){m_type1Residual=type1res;}
        void  setType2Residual(float type2res){m_type2Residual=type2res;}
        void  setCorrectionsBX(std::vector<float>& correctBX){m_correctionsBX.assign(correctBX.begin(),correctBX.end());} 
        float getOverallCorrection(){return m_overallCorrection;}
        float getType1Fraction(){return m_type1Fraction;}
        float getType1Residual(){return m_type1Residual;}
        float getType2Residual(){return m_type2Residual;}
        const std::vector<float>& getCorrectionsBX() const{return m_correctionsBX;} 
    private:
        float m_overallCorrection;
        float m_type1Fraction;
        float m_type1Residual;
        float m_type2Residual;
        std::vector<float> m_correctionsBX;
        COND_SERIALIZABLE; 
};
#endif 
