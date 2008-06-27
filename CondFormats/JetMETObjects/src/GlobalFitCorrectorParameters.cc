//
// Original Author:  Roger Wolf Jun 25, 2008
// $Id: GlobalJetCorrectorParameters.cc,v 1.3 2008/02/29 20:28:31 fedor Exp $
//
// Generic parameters for Jet corrections
//
#include "CondFormats/JetMETObjects/interface/GlobalFitCorrectorParameters.h"


#include <iostream>
#include <fstream>
#include <stdlib.h>

#include "FWCore/Utilities/interface/Exception.h"

namespace {
  int getInt (const std::string& token) {
    int result = atoi (token.c_str());
    if (result == 0) {
      throw cms::Exception ("GlobalJetCorrectorParameters") << "can not convert token " << token << " to int value";
    }
    return result;
  }
  float getFloat (const std::string& token) {
    char* endptr;
    float result = strtod (token.c_str(), &endptr);
    if (endptr == token.c_str()) {
      throw cms::Exception ("GlobalJetCorrectorParameters") << "can not convert token " << token << " to float value";
    }
    return result;
  }
  unsigned getUnsigned (const std::string& token) {
    char* endptr;
    unsigned result = strtoul (token.c_str(), &endptr, 0);
    if (endptr == token.c_str()) {
      throw cms::Exception ("GlobalJetCorrectorParameters") << "can not convert token " << token << " to unsigned value";
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

GlobalFitCorrectorParameters::Record::Record(const std::string& line) : 
  iEta_(0), iPhi_(0)
{
  // qickly parse the line
  std::vector<std::string> tokens;
  std::string currentToken;
  for(unsigned ipos=0; ipos<line.length(); ++ipos){
    char c=line[ipos];
    if( c=='#' ) break; // ignore comments
    else if( c==' ' ){ // flush current token if any
      if( !currentToken.empty() ){
	tokens.push_back( currentToken );
	currentToken.clear();
      }
    }
    else{
      currentToken += c;
    }
  }
  if( !currentToken.empty() ) tokens.push_back( currentToken ); // flush end
  if( !tokens.empty() ){ // pure comment line
    if( tokens.size()<4 ){
      throw cms::Exception("GlobalJetCorrectorParameters") << "at least 4 tokens are expected: iEta, iPhi, "
							   << "# of tower parameters, # of jet parameters. " 
							   << tokens.size() << " are provided.\n" 
							   << "Line ->>" << line << "<<-";  
    }
    // get parameters
    iEta_= getInt( tokens[0] );
    iPhi_= getInt( tokens[1] );
    unsigned nTowerParams = getUnsigned( tokens[2] );
    unsigned nJetParams   = getUnsigned( tokens[3] );
    for( unsigned i=4; i<tokens.size(); ++i){
      if( i<nTowerParams+4 )
	towerParameters_.push_back( getFloat( tokens[i] ) );
      else
	jetParameters_  .push_back( getFloat( tokens[i] ) );
    }
    if( nTowerParams != towerParameters_.size() ) {
      throw cms::Exception ("GlobalJetCorrectorParameters") << "Actual # of tower parameters " << towerParameters_.size() 
							    << " doesn't correspond to requested #: " << nTowerParams << "\n"
							    << "Line ->>" << line << "<<-";  
    }
    if( nJetParams   != jetParameters_  .size() ) {
      throw cms::Exception ("GlobalJetCorrectorParameters") << "Actual # of jet parameters "   << jetParameters_  .size() 
							    << " doesn't correspond to requested #: " << nJetParams   << "\n"
							    << "Line ->>" << line << "<<-";  
    }
  }
}

GlobalFitCorrectorParameters::GlobalFitCorrectorParameters (const std::string& file, const std::string& section) : etaSize_(0)
{
  // define parametrization
  std::string param = file.substr(file.find_last_of("."), file.find_last_of("_")+1);
  if     ( param=="StepParametrization"      ) parametrization_= new StepParametrization();
  else if( param=="StepEfracParametrization" ) parametrization_= new StepEfracParametrization();
  else if( param=="MyParametrization"        ) parametrization_= new MyParametrization();
  else if( param=="JetMETParametrization"    ) parametrization_= new JetMETParametrization();
  else{
    throw cms::Exception("GlobalJetCorrectorParameters") 
      << "cannot instantiate a Parametrization of name " << param << "\n";
  }
  
  // parse input file
  std::ifstream input (file.c_str());
  std::string currentSection = "";
  std::string line;

  int minEta=0, maxEta=0;
  while( std::getline (input, line) ){
    std::string section = getSection( line );
    if( !section.empty () ){
      currentSection = section;
      continue;
    }
    if( currentSection == section ){
      Record record( line );
      if( !(record.iEta()==0 || record.iPhi()==0 || record.nTowerParameters()==0 || record.nJetParameters()==0) ){
	if( minEta>record.iEta() ) // determine max # off eta bins
	  minEta=record.iEta();
	if( maxEta<record.iEta() ) // determine max # off eta bins
	  maxEta=record.iEta();
	records_.push_back (record);
      }
    }
  }
  etaSize_=(unsigned)(maxEta-minEta);
  if( records_.empty() ) records_.push_back( Record() );
  std::sort( records_.begin(), records_.end() );
}
