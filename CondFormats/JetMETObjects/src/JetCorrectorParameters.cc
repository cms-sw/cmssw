//
// Original Author:  Fedor Ratnikov Nov 9, 2007
// $Id: JetCorrectorParameters.cc,v 1.2 2009/11/07 13:46:01 schiefer Exp $
//
// Generic parameters for Jet corrections
//
#include "CondFormats/JetMETObjects/interface/JetCorrectorParameters.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <stdlib.h>
#include <algorithm>
#include <cmath>

#ifdef STANDALONE
#include <sstream>
#include <stdexcept>
#else
#include "FWCore/Utilities/interface/Exception.h"
#endif

namespace 
{
  //----------------------------------------------------------------------
  float getFloat(const std::string& token) 
  {
    char* endptr;
    float result = strtod (token.c_str(), &endptr);
    if (endptr == token.c_str()) 
      {
        #ifdef STANDALONE
          std::stringstream sserr; 
          sserr<<"JetCorrectorParameters ERROR: "<<"can't convert token "<<token<<" to float value";
          throw std::runtime_error(sserr.str());
        #else
          throw cms::Exception ("JetCorrectorParameters")<<" can not convert token "<<token<<" to float value";
        #endif
      }
    return result;
  } 
  //----------------------------------------------------------------------
  unsigned getUnsigned(const std::string& token) 
  {
    char* endptr;
    unsigned result = strtoul (token.c_str(), &endptr, 0);
    if (endptr == token.c_str()) 
      {
        #ifdef STANDALONE
           std::stringstream sserr; 
           sserr<<"JetCorrectorParameters ERROR: "<<"can't convert token "<<token<<" to unsigned value";
           throw std::runtime_error(sserr.str());
        #else
           throw cms::Exception ("JetCorrectorParameters")<<" can not convert token "<<token<<" to unsigned value";
        #endif
      }
    return result;
  }
  //----------------------------------------------------------------------
  std::string getSection(const std::string& token) 
  {
    unsigned iFirst = token.find ('[');
    unsigned iLast = token.find (']');
    if (iFirst != std::string::npos && iLast != std::string::npos && iFirst < iLast)
      return std::string (token, iFirst+1, iLast-iFirst-1); 
    return "";
  }
  //----------------------------------------------------------------------
  std::vector<std::string> getTokens(const std::string& fLine)
  {
    std::vector<std::string> tokens;
    std::string currentToken;
    for (unsigned ipos = 0; ipos < fLine.length (); ++ipos) 
      {
        char c = fLine[ipos];
        if (c == '#') break; // ignore comments
        else if (c == ' ') 
          { // flush current token if any
            if (!currentToken.empty()) 
              {
	        tokens.push_back(currentToken);
	        currentToken.clear();
              }
          }
        else
          currentToken += c;
      }
    if (!currentToken.empty()) tokens.push_back(currentToken); // flush end 
    return tokens;
  }
  //---------------------------------------------------------------------- 
  std::string getDefinitions(const std::string& token) 
  {
    unsigned iFirst = token.find ('{');
    unsigned iLast = token.find ('}');
    if (iFirst != std::string::npos && iLast != std::string::npos && iFirst < iLast)
      return std::string (token, iFirst+1, iLast-iFirst-1); 
    return "";
  }
}
//------------------------------------------------------------------------ 
//--- JetCorrectorParameters::Definitions constructor --------------------
//--- takes specific arguments for the member variables ------------------
//------------------------------------------------------------------------
JetCorrectorParameters::Definitions::Definitions(const std::vector<std::string>& fBinVar, const std::vector<std::string>& fParVar, const std::string& fFormula, bool fIsResponse)
{
  for(unsigned i=0;i<fBinVar.size();i++)
    mBinVar.push_back(fBinVar[i]);
  for(unsigned i=0;i<fParVar.size();i++)
    mParVar.push_back(fParVar[i]);
  mFormula    = fFormula;
  mIsResponse = fIsResponse;
}
//------------------------------------------------------------------------
//--- JetCorrectorParameters::Definitions constructor --------------------
//--- reads the member variables from a string ---------------------------
//------------------------------------------------------------------------
JetCorrectorParameters::Definitions::Definitions(const std::string& fLine)
{
  std::vector<std::string> tokens = getTokens(fLine); 
  if (!tokens.empty()) 
    { 
      if (tokens.size() < 6) 
        {
          #ifdef STANDALONE
	     std::stringstream sserr;
	     sserr<<"JetCorrectorParameters::Definitions ERROR (line "<<fLine
                  <<"): less than 6 expected tokens:"<<tokens.size();
	     throw std::runtime_error(sserr.str());
          #else
             throw cms::Exception("JetCorrectorParameters::Definitions")
	       <<" 6 tokens are expected: number of bin variables, "
	       <<"variable name, number of parameter variables, parameters names, formula, Response or Correction identifier. "<<tokens.size()<<" are provided.\n"
	       <<"Line ->>"<<fLine<<"<<-";
          #endif
        }
      unsigned nvar = getUnsigned(tokens[0]);
      unsigned npar = getUnsigned(tokens[nvar+1]);
      for(unsigned i=0;i<nvar;i++)
        mBinVar.push_back(tokens[i+1]);
      for(unsigned i=0;i<npar;i++)
        mParVar.push_back(tokens[nvar+2+i]);
      mFormula = tokens[npar+nvar+2];
      std::string ss = tokens[npar+nvar+3]; 
      if (ss == "Response")
        mIsResponse = true;
      else if (ss == "Correction")
        mIsResponse = false;
      else
        {
          #ifdef STANDALONE
	     std::stringstream sserr;
	     sserr<<"JetCorrectorParameters::Definitions ERROR: unknown option ("<<ss<<")"; 
	     throw std::runtime_error(sserr.str());
          #else
             throw cms::Exception("JetCorrectorParameters::Definitions")<<" unknown option ("<<ss<<")";
          #endif
        }  
    }
}
//------------------------------------------------------------------------
//--- JetCorrectorParameters::Record constructor -------------------------
//--- reads the member variables from a string ---------------------------
//------------------------------------------------------------------------
JetCorrectorParameters::Record::Record(const std::string& fLine,unsigned fNvar) : mMin(0),mMax(0)
{
  mNvar = fNvar;
  // quckly parse the line
  std::vector<std::string> tokens = getTokens(fLine);
  if (!tokens.empty()) 
    { 
      if (tokens.size() < 3) 
        {
          #ifdef STANDALONE
	    std::stringstream sserr;
	    sserr<<"JetCorrectorParameters::Record ERROR (line "<<fLine<<"): "
	         <<"three tokens expected, "<<tokens.size()<<" provided.";
	    throw std::runtime_error(sserr.str());
          #else
            throw cms::Exception("JetCorrectorParameters::Record")
	    <<" at least 3 tokens are expected: minX, maxX, # of parameters. " <<tokens.size()
	    <<" are provided.\n" <<"Line ->>"<<fLine<<"<<-";  
          #endif
        }
      for(unsigned i=0;i<mNvar;i++)
        {
          mMin.push_back(getFloat(tokens[i*mNvar]));
          mMax.push_back(getFloat(tokens[i*mNvar+1])); 
        }
      unsigned nParam = getUnsigned(tokens[2*mNvar]);
      if (nParam != tokens.size()-(2*mNvar+1)) 
        {
          #ifdef STANDALONE
	    std::stringstream sserr;
	    sserr<<"JetCorrectorParameters::Record ERROR (line "<<fLine<<"): "
	         <<tokens.size()-(2*mNvar+1)<<" parameters, but nParam="<<nParam<<".";
	    throw std::runtime_error(sserr.str());
          #else
            throw cms::Exception ("JetCorrectorParameters::Record")
	    <<" actual # of parameters "<<tokens.size()-(2*mNvar+1)
	    <<" doesn't correspond to requested #: "<<nParam<<"\n"<<"Line ->>"<<fLine<<"<<-";
          #endif
        }
      for (unsigned i = (2*mNvar+1); i < tokens.size(); ++i)
        mParameters.push_back(getFloat(tokens[i]));
    } 
}
//------------------------------------------------------------------------
//--- JetCorrectorParameters constructor ---------------------------------
//--- reads the member variables from a string ---------------------------
//------------------------------------------------------------------------
JetCorrectorParameters::JetCorrectorParameters(const std::string& fFile, const std::string& fSection) 
{
  std::ifstream input(fFile.c_str());
  std::string currentSection = "";
  std::string line;
  std::string currentDefinitions = "";
  while (std::getline(input,line)) 
    {
      std::string section = getSection(line);
      std::string tmp = getDefinitions(line);
      if (!section.empty() && tmp.empty()) 
        {
          currentSection = section;
          continue;
        }
      if (currentSection == fSection) 
        {
          if (!tmp.empty()) 
            {
              currentDefinitions = tmp;
              continue; 
            }
          Definitions definitions(currentDefinitions);
          if (!(definitions.nBinVar()==0 && definitions.formula()==""))
            mDefinitions = definitions;
          Record record(line,mDefinitions.nBinVar());
          bool check(true);
          for(unsigned i=0;i<mDefinitions.nBinVar();++i)
            if (record.xMin(i)==0 && record.xMax(i)==0)
              check = false;
          if (record.nParameters() == 0)
            check = false;  
          if (check)
            mRecords.push_back(record);
        } 
    }
  if (mRecords.empty() && currentSection == "") mRecords.push_back(Record());
  if (mRecords.empty() && currentSection != "") 
    {
      #ifdef STANDALONE
         std::stringstream sserr; 
         sserr<<"JetCorrectorParameters ERROR: "<<"the requested section "<<fSection<<" doesn't exist!";
         throw std::runtime_error(sserr.str());
      #else
         throw cms::Exception ("JetCorrectorParameters")<<" the requested section "<<fSection<<" doesn't exist !!!"<< "\n";
      #endif
    }
  std::sort(mRecords.begin(), mRecords.end());
}
//------------------------------------------------------------------------
//--- returns the index of the record defined by fX ----------------------
//------------------------------------------------------------------------
int JetCorrectorParameters::binIndex(const std::vector<float>& fX) const 
{
  int result = -1;
  unsigned N = mDefinitions.nBinVar();
  if (N != fX.size()) 
    {
      #ifdef STANDALONE
         std::stringstream sserr;
         sserr<<"JetCorrectorParameters ERROR: # bin variables "<<N<<" doesn't correspont to requested #: "<<fX.size();
         throw std::runtime_error(sserr.str());
      #else
         throw cms::Exception("JetCorrectorParameters")
         <<" # of bin variables "<<N<<" doesn't correspond to requested #: "<< fX.size();
      #endif
    }
  unsigned tmp;
  for (unsigned i = 0; i < size(); ++i) 
    {
      tmp = 0;
      for (unsigned j=0;j<N;j++)
        if (fX[j] >= record(i).xMin(j) && fX[j] < record(i).xMax(j))
          tmp+=1;
      if (tmp==N)
        { 
          result = i;
          break;
        }
    } 
  return result;
}
//------------------------------------------------------------------------
//--- returns the neighbouring bins of fIndex in the direction of fVar ---
//------------------------------------------------------------------------
int JetCorrectorParameters::neighbourBin(unsigned fIndex, unsigned fVar, bool fNext) const 
{
  int result = -1;
  unsigned N = mDefinitions.nBinVar();
  if (fVar >= N) 
    {
      #ifdef STANDALONE
         std::stringstream sserr;
         sserr<<"JetCorrectorParameters ERROR: # of bin variables "<<N
	      <<" doesn't correspond to requested #: "<<fVar;
         throw std::runtime_error(sserr.str());
      #else
         throw cms::Exception("JetCorrectorParameters")
         <<" # of bin variables "<<N<<" doesn't correspond to requested #: "<<fVar;
      #endif
    }
  unsigned tmp;
  for (unsigned i = 0; i < size(); ++i) 
    {
      tmp = 0;
      for (unsigned j=0;j<fVar;j++)
        if (fabs(record(i).xMin(j)-record(fIndex).xMin(j))<0.0001)
          tmp+=1;
      for (unsigned j=fVar+1;j<N;j++)
        if (fabs(record(i).xMin(j)-record(fIndex).xMin(j))<0.0001)
          tmp+=1;
      if (tmp<N-1)
        continue; 
      if (tmp==N-1)
        {
          if (fNext)
            if (fabs(record(i).xMin(fVar)-record(fIndex).xMax(fVar))<0.0001)
              tmp+=1;
          if (!fNext)
            if (fabs(record(i).xMax(fVar)-record(fIndex).xMin(fVar))<0.0001)
              tmp+=1;
        } 
      if (tmp==N)
        { 
          result = i;
          break;
        }
    } 
  return result;
}
//------------------------------------------------------------------------
//--- returns the number of bins in the direction of fVar ----------------
//------------------------------------------------------------------------
unsigned JetCorrectorParameters::size(unsigned fVar) const
{
  if (fVar >= mDefinitions.nBinVar()) 
    {   
      #ifdef STANDALONE
         std::stringstream sserr;
         sserr<<"JetCorrectorParameters ERROR: requested bin variable index "<<fVar
	      <<" is greater than number of variables "<<mDefinitions.nBinVar();
         throw std::runtime_error(sserr.str());
      #else
         throw cms::Exception ("JetCorrectorParameters")
         <<" requested bin variable index " <<fVar<<" is greater than number of variables "<<mDefinitions.nBinVar();
      #endif
    }
  unsigned result = 0;
  float tmpMin(-9999),tmpMax(-9999);
  for (unsigned i = 0; i < size(); ++i)
    if (record(i).xMin(fVar) > tmpMin && record(i).xMax(fVar) > tmpMax)
      { 
        result++;
        tmpMin = record(i).xMin(fVar);
        tmpMax = record(i).xMax(fVar);
      }
  return result; 
}
//------------------------------------------------------------------------
//--- returns the vector of bin centers of fVar --------------------------
//------------------------------------------------------------------------
std::vector<float> JetCorrectorParameters::binCenters(unsigned fVar) const 
{
  std::vector<float> result;
  for (unsigned i = 0; i < size(); ++i)
    result.push_back(record(i).xMiddle(fVar));
  return result;
}
//------------------------------------------------------------------------
//--- prints parameters on screen ----------------------------------------
//------------------------------------------------------------------------
void JetCorrectorParameters::printScreen() const
{
  std::cout<<"--------------------------------------------"<<std::endl;
  std::cout<<"////////  PARAMETERS: //////////////////////"<<std::endl;
  std::cout<<"--------------------------------------------"<<std::endl;
  std::cout<<"Number of binning variables:   "<<definitions().nBinVar()<<std::endl;
  std::cout<<"Names of binning variables:    ";
  for(unsigned i=0;i<definitions().nBinVar();i++)
    std::cout<<definitions().binVar(i)<<" ";
  std::cout<<std::endl;
  std::cout<<"--------------------------------------------"<<std::endl;
  std::cout<<"Number of parameter variables: "<<definitions().nParVar()<<std::endl;
  std::cout<<"Names of parameter variables:  ";
  for(unsigned i=0;i<definitions().nParVar();i++)
    std::cout<<definitions().parVar(i)<<" ";
  std::cout<<std::endl;
  std::cout<<"--------------------------------------------"<<std::endl;
  std::cout<<"Parametrization Formula:       "<<definitions().formula()<<std::endl;
  if (definitions().isResponse())
    std::cout<<"Type (Response or Correction): "<<"Response"<<std::endl;
  else
    std::cout<<"Type (Response or Correction): "<<"Correction"<<std::endl;
  std::cout<<"--------------------------------------------"<<std::endl;
  std::cout<<"------- Bin contents -----------------------"<<std::endl;
  for(unsigned i=0;i<size();i++)
    {
      for(unsigned j=0;j<definitions().nBinVar();j++)
        std::cout<<record(i).xMin(j)<<" "<<record(i).xMax(j)<<" ";
      std::cout<<record(i).nParameters()<<" ";
      for(unsigned j=0;j<record(i).nParameters();j++)
        std::cout<<record(i).parameter(j)<<" ";
      std::cout<<std::endl;
    }  
}
//------------------------------------------------------------------------
//--- prints parameters on file ----------------------------------------
//------------------------------------------------------------------------
void JetCorrectorParameters::printFile(const std::string& fFileName) const
{
  std::ofstream txtFile;
  txtFile.open(fFileName.c_str());
  txtFile.setf(std::ios::right);
  txtFile<<"{"<<definitions().nBinVar()<<std::setw(11);
  for(unsigned i=0;i<definitions().nBinVar();i++)
    txtFile<<definitions().binVar(i)<<std::setw(11);
  txtFile<<definitions().nParVar()<<std::setw(11);
  for(unsigned i=0;i<definitions().nParVar();i++)
    txtFile<<definitions().parVar(i)<<std::setw(11);
  txtFile<<std::setw(definitions().formula().size()+11)<<definitions().formula()<<std::setw(11);
  if (definitions().isResponse())
    txtFile<<"Response"<<"}"<<"\n";
  else
    txtFile<<"Correction"<<"}"<<"\n";
  for(unsigned i=0;i<size();i++)
    {
      for(unsigned j=0;j<definitions().nBinVar();j++)
        txtFile<<record(i).xMin(j)<<std::setw(11)<<record(i).xMax(j)<<std::setw(11);
      txtFile<<record(i).nParameters()<<std::setw(11);
      for(unsigned j=0;j<record(i).nParameters();j++)
        txtFile<<record(i).parameter(j)<<std::setw(11);
      txtFile<<"\n";
    }
  txtFile.close();
}



