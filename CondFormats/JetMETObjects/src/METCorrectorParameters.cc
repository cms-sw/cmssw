// Generic parameters for MET corrections
//
#include "CondFormats/JetMETObjects/interface/METCorrectorParameters.h"
#include "CondFormats/JetMETObjects/interface/Utilities.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <algorithm>
#include <cmath>
#include <iterator>
//#include <string>

//------------------------------------------------------------------------ 
//--- METCorrectorParameters::Definitions constructor --------------------
//--- takes specific arguments for the member variables ------------------
//------------------------------------------------------------------------
METCorrectorParameters::Definitions::Definitions(const std::vector<std::string>& fBinVar, const std::vector<std::string>& fParVar, const std::string& fFormula )
{
  for(unsigned i=0;i<fBinVar.size();i++)
    mBinVar.push_back(fBinVar[i]);
  for(unsigned i=0;i<fParVar.size();i++)
    mParVar.push_back(fParVar[i]);
  mFormula    = fFormula;
}
//------------------------------------------------------------------------
//--- METCorrectorParameters::Definitions constructor --------------------
//--- reads the member variables from a string ---------------------------
//------------------------------------------------------------------------
METCorrectorParameters::Definitions::Definitions(const std::string& fLine)
{
  std::vector<std::string> tokens = getTokens(fLine);
  // corrType N_bin binVa.. N_var var... formula
  if (!tokens.empty())
  { 
    if (tokens.size() < 6) 
    {
      std::stringstream sserr;
      sserr<<"(line "<<fLine<<"): less than 6 expected tokens:"<<tokens.size();
      handleError("METCorrectorParameters::Definitions",sserr.str());
    }
    // No. of Bin Variable
    std::cout<<"Definitions==========="<<std::endl;
    ptclType = getSigned(tokens[0]);
    std::cout<<tokens[0]<<"\t";
    unsigned nBinVar = getUnsigned(tokens[1]);
    std::cout<<tokens[1]<<"\t";
    // No of Parameterization Variable
    unsigned nParVar = getUnsigned(tokens[nBinVar+2]);
    std::cout<<tokens[2]<<"\t";
    for(unsigned i=0;i<nBinVar;i++)
    {
      mBinVar.push_back(tokens[i+2]);
      std::cout<<tokens[i+2]<<"\t";
    }
    for(unsigned i=0;i<nParVar;i++)
    {
      mParVar.push_back(tokens[nBinVar+3+i]);
      std::cout<<tokens[nBinVar+3+i]<<"\t";
    }
    mFormula = tokens[nParVar+nBinVar+3];
    std::cout<<tokens[nParVar+nBinVar+3]<<std::endl;
  }
}
//------------------------------------------------------------------------
//--- METCorrectorParameters::Record constructor -------------------------
//--- reads the member variables from a string ---------------------------
//------------------------------------------------------------------------
METCorrectorParameters::Record::Record(const std::string& fLine,unsigned fNvar) : mMin(0), mMax(0), mParameters(0)
{
  mNvar = fNvar;
  // quckly parse the line
  std::vector<std::string> tokens = getTokens(fLine);
  if (!tokens.empty())
  { 
    if (tokens.size() < 6) 
    {
      std::stringstream sserr;
      sserr<<"(line "<<fLine<<"): "<<"three tokens expected, "<<tokens.size()<<" provided.";
      handleError("METCorrectorParameters::Record",sserr.str());
    }
    std::cout<<"Record ==============="<<std::endl;
    for(unsigned i=0;i<mNvar;i++)
    {
      mMin.push_back(getFloat(tokens[i*2]));
      mMax.push_back(getFloat(tokens[i*2+1]));
      std::cout<<tokens[i*2]<<"\t";
      std::cout<<tokens[i*2+1]<<"\t";
    }
    unsigned nParam = getUnsigned(tokens[2*mNvar]);
    std::cout<<tokens[2*mNvar]<<"\t";
    if (nParam != tokens.size()-(2*mNvar+1)) 
    {
      std::stringstream sserr;
      sserr<<"(line "<<fLine<<"): "<<tokens.size()-(2*mNvar+1)<<" parameters, but nParam="<<nParam<<".";
      handleError("METCorrectorParameters::Record",sserr.str());
    }
    for (unsigned i = (2*mNvar+1); i < tokens.size(); ++i)
    {
      mParameters.push_back(getFloat(tokens[i]));
      std::cout<<tokens[i]<<"\t";
    }
    std::cout<<std::endl;
  }
}
//------------------------------------------------------------------------
//--- JetCorrectorParameters constructor ---------------------------------
//--- reads the member variables from a string ---------------------------
//------------------------------------------------------------------------
METCorrectorParameters::METCorrectorParameters(const std::string& fFile, const std::string& fSection) 
{ 
  std::ifstream input(fFile.c_str());
  std::string currentSection = "";
  std::string line;
  std::string currentDefinitions = "";
  while (std::getline(input,line)) 
  {
    //std::cout << " Line of parameters " << line << std::endl;
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
  if (currentDefinitions=="")
    handleError("METCorrectorParameters","No definitions found!!!");
  if (mRecords.empty() && currentSection == "") mRecords.push_back(Record());
  if (mRecords.empty() && currentSection != "") 
  {
    std::stringstream sserr; 
    sserr<<"the requested section "<<fSection<<" doesn't exist!";
    handleError("METCorrectorParameters",sserr.str()); 
  }
  std::sort(mRecords.begin(), mRecords.end());
  valid_ = true;
}
//------------------------------------------------------------------------
//--- returns the index of the record defined by fX ----------------------
//------------------------------------------------------------------------
/*
int METCorrectorParameters::binIndex(const std::vector<float>& fX) const 
{
  int result = -1;
  unsigned N = mDefinitions.nVar();
  if (N != fX.size()) 
    {
      std::stringstream sserr; 
      sserr<<"# bin variables "<<N<<" doesn't correspont to requested #: "<<fX.size();
      handleError("METCorrectorParameters",sserr.str());
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
*/
//------------------------------------------------------------------------
//--- returns the neighbouring bins of fIndex in the direction of fVar ---
//------------------------------------------------------------------------
/*
int METCorrectorParameters::neighbourBin(unsigned fIndex, unsigned fVar, bool fNext) const 
{
  int result = -1;
  unsigned N = mDefinitions.nVar();
  if (fVar >= N) 
    {
      std::stringstream sserr; 
      sserr<<"# of bin variables "<<N<<" doesn't correspond to requested #: "<<fVar;
      handleError("METCorrectorParameters",sserr.str()); 
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
*/
//------------------------------------------------------------------------
//--- returns the number of bins in the direction of fVar ----------------
//------------------------------------------------------------------------
/*
unsigned METCorrectorParameters::size(unsigned fVar) const
{
  if (fVar >= mDefinitions.nVar()) 
    { 
      std::stringstream sserr; 
      sserr<<"requested bin variable index "<<fVar<<" is greater than number of variables "<<mDefinitions.nVar();
      handleError("METCorrectorParameters",sserr.str()); 
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
*/
//------------------------------------------------------------------------
//--- returns the vector of bin centers of fVar --------------------------
//------------------------------------------------------------------------
/*
std::vector<float> METCorrectorParameters::binCenters(unsigned fVar) const 
{
  std::vector<float> result;
  for (unsigned i = 0; i < size(); ++i)
    result.push_back(record(i).xMiddle(fVar));
  return result;
}
*/
//------------------------------------------------------------------------
//--- prints parameters on screen ----------------------------------------
//------------------------------------------------------------------------
void METCorrectorParameters::printScreen() const
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
void METCorrectorParameters::printFile(const std::string& fFileName) const
{
  std::ofstream txtFile;
  txtFile.open(fFileName.c_str());
  txtFile.setf(std::ios::right);
  txtFile<<"{"<<definitions().nBinVar()<<std::setw(15);
  for(unsigned i=0;i<definitions().nBinVar();i++)
    txtFile<<definitions().binVar(i)<<std::setw(15);
  txtFile<<definitions().nParVar()<<std::setw(15);
  for(unsigned i=0;i<definitions().nParVar();i++)
    txtFile<<definitions().parVar(i)<<std::setw(15);
  txtFile<<std::setw(definitions().formula().size()+15)<<definitions().formula()<<std::setw(15);
  txtFile<<"}"<<"\n";
  for(unsigned i=0;i<size();i++)
    {
      for(unsigned j=0;j<definitions().nBinVar();j++)
        txtFile<<record(i).xMin(j)<<std::setw(15)<<record(i).xMax(j)<<std::setw(15);
      txtFile<<record(i).nParameters()<<std::setw(15);
      for(unsigned j=0;j<record(i).nParameters();j++)
        txtFile<<record(i).parameter(j)<<std::setw(15);
      txtFile<<"\n";
    }
  txtFile.close();
}

namespace {
const std::vector<std::string> labels_ = {
  "MiniAod"
};
const std::vector<std::string> MiniAodSource_ = {
  "MiniAod_ShiftX",
  "MiniAod_ShiftY"
};

}//namespace

std::string
METCorrectorParametersCollection::findLabel( key_type k ){
  std::cout<<"findLabel with key_type: "<<k<<std::endl;
  if( isMiniAod(k) )
  {
    std::cout<<"is MiniAod"<<std::endl;
    return findMiniAodSource(k);
  }
  return labels_[k];
}
std::string
METCorrectorParametersCollection::findMiniAodSource( key_type k)
{
  if( k == MiniAod) return labels_[MiniAod];
  else
    return MiniAodSource_[k - MiniAod*100 -1];
}
void METCorrectorParametersCollection::getSections( std::string inputFile,
						    std::vector<std::string> & outputs )
{
  outputs.clear();
  std::ifstream input( inputFile.c_str() );
  while( !input.eof() ) {
    char buff[10000];
    input.getline(buff,10000);
    std::string in(buff);
    if ( in[0] == '[' ) {
      std::string tok = getSection(in);
      if ( tok != "" ) {
	outputs.push_back( tok );
      }
    }
  }
  std::cout << "Found these sections for file: " << std::endl;
  copy(outputs.begin(),outputs.end(), std::ostream_iterator<std::string>(std::cout, "\n") );
}

// Add a METCorrectorParameter object. 
void METCorrectorParametersCollection::push_back( key_type i, value_type const & j, label_type const &source )
{ 
  std::cout << "i    = " << i << std::endl;  
  std::cout << "source = " << source << std::endl;
  if( isMiniAod(i))
  {
    std::cout << "This is MiniAod, getMiniAodBin = " << getMiniAodBin(source) << std::endl;
    correctionsMiniAod_.push_back( pair_type(getMiniAodBin(source),j) );
  }else{
    std::cout << "***** NOT ADDING " << source << ", corresponding position in METCorrectorParameters is not found." << std::endl;
  }
}

// Access the METCorrectorParameter via the key k.
// key_type is hashed to deal with the three collections
METCorrectorParameters const & METCorrectorParametersCollection::operator[]( key_type k ) const {
  collection_type::const_iterator ibegin, iend, i;
  if ( isMiniAod(k) ) {
    ibegin = correctionsMiniAod_.begin();
    iend = correctionsMiniAod_.end();
    i = ibegin;
  }
  for ( ; i != iend; ++i ) {
    if ( k == i->first ) return i->second;
  }
  throw cms::Exception("InvalidInput") << " cannot find key " << static_cast<int>(k)
				       << " in the METC payload, this usually means you have to change the global tag" << std::endl;
}

// Get a list of valid keys. These will contain hashed keys
// that are aware of all three collections.
void METCorrectorParametersCollection::validKeys(std::vector<key_type> & keys ) const {
  keys.clear();
  for ( collection_type::const_iterator ibegin = correctionsMiniAod_.begin(),
	  iend = correctionsMiniAod_.end(), i = ibegin; i != iend; ++i ) {
    keys.push_back( i->first );
  }
}


METCorrectorParametersCollection::key_type
METCorrectorParametersCollection::getMiniAodBin( std::string const & source ){
  std::vector<std::string>::const_iterator found =
    find( MiniAodSource_.begin(), MiniAodSource_.end(), source );
  if ( found != MiniAodSource_.end() ) {
    return (found - MiniAodSource_.begin() + 1)+ MiniAod * 100;
  }
  else return MiniAod;
}

bool METCorrectorParametersCollection::isMiniAod( key_type k ) {
  return k == MiniAod ||
    (k > MiniAod*100 && k < MiniAod*100 + 100);
}

#include "FWCore/Utilities/interface/typelookup.h"
 
TYPELOOKUP_DATA_REG(METCorrectorParameters);
TYPELOOKUP_DATA_REG(METCorrectorParametersCollection);
