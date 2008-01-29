//
// Original Author:  Fedor Ratnikov Nov 9, 2007
// $Id: SimpleJetCorrectorParameters.h,v 1.1 2007/11/01 21:50:30 fedor Exp $
//
// Generic parameters for Jet corrections
//
#include "CondFormats/JetMETObjects/interface/SimpleJetCorrectorParameters.h"


#include <iostream>
#include <fstream>
#include <stdlib.h>

#include "FWCore/Utilities/interface/Exception.h"

namespace {
  float getFloat (const std::string& token) {
    char* endptr;
    float result = strtod (token.c_str(), &endptr);
    if (endptr == token.c_str()) {
      throw cms::Exception ("SimpleJetCorrectorParameters") << "can not convert token " << token << " to float value";
    }
    return result;
  }
  unsigned getUnsigned (const std::string& token) {
    char* endptr;
    unsigned result = strtoul (token.c_str(), &endptr, 0);
    if (endptr == token.c_str()) {
      throw cms::Exception ("SimpleJetCorrectorParameters") << "can not convert token " << token << " to unsigned value";
    }
    return result;
  }
}

SimpleJetCorrectorParameters::Record::Record (const std::string& fLine) 
  : mEtaMin (0), mEtaMax (0)
{
  // quckly parse the line
  std::vector<std::string> tokens;
  std::string currentToken;
  for (unsigned ipos = 0; ipos < fLine.length (); ++ipos) {
    char c = fLine[ipos];
    if (c == '#') break; // ignore comments
    else if (c == ' ') { // flush current token if any
      if (!currentToken.empty()) {
	tokens.push_back (currentToken);
	currentToken.clear();
      }
    }
    else {
      currentToken += c;
    }
  }
  if (!currentToken.empty()) tokens.push_back (currentToken); // flush end
  if (!tokens.empty ()) { // pure comment line
    if (tokens.size() < 3) {
      throw cms::Exception ("SimpleJetCorrectorParameters") << "at least 3 tokens are expected: minEta, maxEta, # of parameters. " 
							    << tokens.size() << " are provided.\n" 
							    << "Line ->>" << fLine << "<<-";  
    }
    // get parameters
    mEtaMin = getFloat (tokens[0]);
    mEtaMax = getFloat (tokens[1]);
    unsigned nParam = getUnsigned (tokens[2]);
    if (nParam != tokens.size() - 3) {
      throw cms::Exception ("SimpleJetCorrectorParameters") << "Actual # of parameters " << tokens.size() - 3 << " doesn't correspond to requested #: " << nParam << "\n"
							    << "Line ->>" << fLine << "<<-";  
    }
    for (unsigned i = 3; i < tokens.size(); ++i) {
      mParameters.push_back (getFloat (tokens[i]));
    }
  }
}

SimpleJetCorrectorParameters::SimpleJetCorrectorParameters (const std::string& fFile) {
  std::ifstream input (fFile.c_str());
  std::string line;
  while (std::getline (input, line)) {
    Record record (line);
    if (!(record.etaMin() == 0. && record.etaMax() == 0. && record.nParameters() == 0)) {
      mRecords.push_back (record);
    }
  }
  if (mRecords.empty()) mRecords.push_back (Record ());
  std::sort (mRecords.begin(), mRecords.end());
}

  /// get band index for eta
int SimpleJetCorrectorParameters::bandIndex (float fEta) const {
  int result = -1;
  for (unsigned i = 0; i < mRecords.size(); ++i) {
    if (fEta >= mRecords[i].etaMin() && fEta < mRecords[i].etaMax()) {
      result = i;
      break;
    } 
  }
  return result;
}

  /// get vector of centers of bands
std::vector<float> SimpleJetCorrectorParameters::bandCenters () const {
  std::vector<float> result;
  for (unsigned i = 0; i < size(); ++i) {
    result.push_back (record(i).etaMiddle());
  }
  return result;
}
