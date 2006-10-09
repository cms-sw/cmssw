
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetEtCalibrationLut.h"

#include "L1Trigger/L1Scales/interface/L1CaloEtScale.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJet.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Utilities/interface/Exception.h"

#include <iostream>
#include <fstream>
#include <sstream>

#include <math.h>

using namespace std;
using std::cout;

//DEFINE STATICS
const unsigned L1GctJetEtCalibrationLut::JET_ENERGY_BITWIDTH = L1GctJet::RAWSUM_BITWIDTH;
const unsigned L1GctJetEtCalibrationLut::NUMBER_ETA_VALUES = 11;

L1GctJetEtCalibrationLut::L1GctJetEtCalibrationLut()
{
}

L1GctJetEtCalibrationLut::L1GctJetEtCalibrationLut(string fileName)
{
  //Opens the file
  ifstream ifs(fileName.c_str(),ios::in);

  if(!ifs.good())
    {
      throw cms::Exception("L1GctFileReadError")
        << "L1GctJetEtCalibrationLut::L1GctJetEtCalibrationLut(string filename)"
        << " couldn't open the file " + fileName + " for reading!\n";
    }

  // Read in the parameters (11 regions of eta from 0-10 corresponding to eta values of 0 to 5)
  string line;
  float tmp;

  m_calibFunc.resize(NUMBER_ETA_VALUES);

  for (unsigned i=0; i<NUMBER_ETA_VALUES; i++){
    getline(ifs,line);
    istringstream iss(line);
    do {
      iss >> tmp;
      m_calibFunc.at(i).push_back(tmp);
    } while (iss.good()); 
  }
}    

void L1GctJetEtCalibrationLut::setOutputEtScale(const L1CaloEtScale* scale) {
  m_outputEtScale = scale;
  
  cout << m_outputEtScale;
}

ostream& operator << (ostream& os, const L1GctJetEtCalibrationLut& lut)
{
  os << "===L1GctJetEtCalibrationLut===" << endl;
  for (unsigned i=0; i<lut.m_calibFunc.size(); i++){
    os << "Eta = " << i << " Coefficients = ";
    for (unsigned j=0; j<lut.m_calibFunc.at(i).size();j++){
      os << lut.m_calibFunc.at(i).at(j) << " "; 
    }
    os << endl;
  }
  return os;
}

L1GctJetEtCalibrationLut::~L1GctJetEtCalibrationLut()
{
}

uint16_t L1GctJetEtCalibrationLut::rank(uint16_t jetEt, unsigned eta) const
{
  return m_outputEtScale->rank(this->calibratedEt(jetEt, eta));
}

uint16_t L1GctJetEtCalibrationLut::calibratedEt(uint16_t jetEt, unsigned eta) const
{
  double corrEt = 0;
  
  if (eta>(NUMBER_ETA_VALUES-1)) eta=eta-NUMBER_ETA_VALUES; 

  if(eta>(NUMBER_ETA_VALUES-1))
    {
      throw cms::Exception("L1GctJetEtCalibraionLut")
        << "L1GctJetEtCalibrationLut::convertToTebBitRank(uint16_t jetEt, unsigned eta)"
        << " eta value out of range eta=" <<  eta <<  "\n";
    }

  for (unsigned i=0; i<m_calibFunc.at(eta).size();i++){
    corrEt += m_calibFunc.at(eta).at(i)*pow((double)jetEt,(int)i); 
  }

  uint16_t jetEtOut = (uint16_t)corrEt;

  if(jetEtOut < (1 << JET_ENERGY_BITWIDTH)) {
    return jetEtOut;
  }
  return 1023;
}
