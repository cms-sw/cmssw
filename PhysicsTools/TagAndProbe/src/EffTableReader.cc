//---------------------------------------------------------------------------
// Adapted From Following Description:
// Original Author:  Fedor Ratnikov Nov 9, 2007
// $Id: SimpleJetCorrectorParameters.cc,v 1.2 2007/11/16 00:14:32 fedor Exp $
// Generic parameters for Jet corrections
//----------------------------------------------------------------------------
#include "PhysicsTools/TagAndProbe/interface/EffTableReader.h"
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include "FWCore/Utilities/interface/Exception.h"

namespace {
  float getFloat (const std::string& token) {
    char* endptr;
    float result = strtod (token.c_str(), &endptr);
    if (endptr == token.c_str()) {
     throw cms::Exception ("SimpleEffCorrectorParameters")
	<< "can not convert token " << token << " to float value";
    }
    return result;
  }
  unsigned getUnsigned (const std::string& token) {
    char* endptr;
    unsigned result = strtoul (token.c_str(), &endptr, 0);
    if (endptr == token.c_str()) {
     throw cms::Exception ("SimpleEffCorrectorParameters") 
	<< "can not convert token " << token << " to unsigned value";
     throw "EffTableReader: failed to convert token to unsigned value";
  }
    return result;
  }
}


EffTableReader::Record::Record (const std::string& fLine) 
  : mEtaMin (0), mEtaMax (0), mEtMax(0), mEtMin(0) 
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
    if (tokens.size() < 5) {
     throw cms::Exception ("EffTableReader") << "at least 5 tokens are expected: minEta, maxEta, # of parameters.,"
                                                            <<  " minEt, maxEt " 
							    << tokens.size() << " are provided.\n" 
							    << "Line ->>" << fLine << "<<-";  
    }
    // get parameters
    mEtMin = getFloat (tokens[0]);
    mEtMax = getFloat (tokens[1]);
    mEtaMin = getFloat (tokens[2]);
    mEtaMax = getFloat (tokens[3]);
    unsigned nParam = getUnsigned (tokens[4]);
    if (nParam != tokens.size() - 5) {
     throw cms::Exception ("EffTableReader") << "Actual # of parameters " 
                                                           << tokens.size() - 5 
                                                           << " doesn't correspond to requested #: " << nParam << "\n"
							    << "Line ->>" << fLine << "<<-";  
    }
    for (unsigned i = 4; i < tokens.size(); ++i) {
      mParameters.push_back (getFloat (tokens[i]));
    }
  }
}


EffTableReader::EffTableReader (const std::string& fFile) {
  std::ifstream input (fFile.c_str());
  std::string line;
  while (std::getline (input, line)) {
    Record record (line);
    if (!(record.etaMin() == 0. && record.etaMax() == 0. && record.nParameters() == 0)) {
      mRecords.push_back (record);
    }
  }
  if (mRecords.empty()) mRecords.push_back (Record ());

}


int EffTableReader::bandIndex (float fEt, float fEta) const{
  int bandInd =0;
       for (unsigned i = 0; i < mRecords.size(); ++i) {
         if(fEt>=mRecords[i].EtMin() && fEt<mRecords[i].EtMax()){
	   if(fEta>=mRecords[i].etaMin() && fEta<mRecords[i].etaMax()){
	     bandInd=i;
              break;                        
                   }
	 }

       }
       return bandInd;
}

