//
// Original Author:  Fedor Ratnikov Nov 9, 2007
// $Id: SimpleJetCorrectorParameters.cc,v 1.4 2009/01/30 09:26:56 elmer Exp $
//
// Generic parameters for Jet corrections
//
#include "CondFormats/JetMETObjects/interface/SimpleJetCorrectorParameters.h"


#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <algorithm>

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
  std::string getSection (const std::string& token) {
    unsigned iFirst = token.find ('[');
    unsigned iLast = token.find (']');
    if (iFirst != std::string::npos && iLast != std::string::npos && iFirst < iLast) {
      return std::string (token, iFirst+1, iLast-iFirst-1);
    } 
    return "";
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

SimpleJetCorrectorParameters::SimpleJetCorrectorParameters (const std::string& fFile, const std::string& fSection) {
  std::ifstream input (fFile.c_str());
  std::string currentSection = "";
  std::string line;
  while (std::getline (input, line)) {
    std::string section = getSection (line);
    if (!section.empty ()) {
      currentSection = section;
      continue;
    }
    if (currentSection == fSection) {
      Record record (line);
      if (!(record.etaMin() == 0. && record.etaMax() == 0. && record.nParameters() == 0)) {
	mRecords.push_back (record);
      }
    }
  }
  if (mRecords.empty() && currentSection == "") mRecords.push_back (Record ());
  if (mRecords.empty() && currentSection != "") {
    throw cms::Exception ("SimpleJetCorrectorParameters") << "The requested section "<< fSection << " doesn't exist !!!" << "\n";
  }
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
