// Generic parameters for MET corrections
//
#include "FWCore/MessageLogger/interface/MessageLogger.h"
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
      sserr<<"(line "<<fLine<<"): Great than or equal to 6 expected tokens:"<<tokens.size();
      handleError("METCorrectorParameters::Definitions",sserr.str());
    }
    // No. of Bin Variable
    LogDebug ("default")<<"Definitions===========";
    ptclType = getSigned(tokens[0]);
    unsigned nBinVar = getUnsigned(tokens[1]);
    unsigned nParVar = getUnsigned(tokens[nBinVar+2]);
    for(unsigned i=0;i<nBinVar;i++)
    {
      mBinVar.push_back(tokens[i+2]);
    }
    for(unsigned i=0;i<nParVar;i++)
    {
      mParVar.push_back(tokens[nBinVar+3+i]);
    }
    mFormula = tokens[nParVar+nBinVar+3];
    if (tokens.size() != nParVar+nBinVar+4 ) 
    {
      std::stringstream sserr;
      sserr<<"(line "<<fLine<<"): token size should be:"<<nParVar+nBinVar+4<<" but it is "<<tokens.size();
      handleError("METCorrectorParameters::Definitions",sserr.str());
    }

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
    if (tokens.size() < 5) 
    {
      std::stringstream sserr;
      sserr<<"(line "<<fLine<<"): "<<"five tokens expected, "<<tokens.size()<<" provided.";
      handleError("METCorrectorParameters::Record",sserr.str());
    }
    mMetAxis = tokens[0];
    for(unsigned i=0;i<mNvar;i++)
    {
      mMin.push_back(getFloat(tokens[i*2+1]));
      mMax.push_back(getFloat(tokens[i*2+2]));
    }
    unsigned nParam = getUnsigned(tokens[2*mNvar+1]);
    if (nParam != tokens.size()-(2*mNvar+2)) 
    {
      std::stringstream sserr;
      sserr<<"(line "<<fLine<<"): "<<tokens.size()-(2*mNvar+2)<<" parameters, but nParam="<<nParam<<".";
      handleError("METCorrectorParameters::Record",sserr.str());
    }
    for (unsigned i = (2*mNvar+2); i < tokens.size(); ++i)
    {
      mParameters.push_back(getFloat(tokens[i]));
    }
  }
}
//------------------------------------------------------------------------
//--- METCorrectorParameters constructor ---------------------------------
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
void METCorrectorParameters::printScreen(const std::string &Section) const
{
  std::cout<<"--------------------------------------------"<<std::endl;
  std::cout<<"////////  PARAMETERS: //////////////////////"<<std::endl;
  std::cout<<"--------------------------------------------"<<std::endl;
  std::cout<<"["<<Section<<"]"<<"\n";
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
  for(unsigned i=0;i<size();i++) //mRecords size
  {
    std::cout<<record(i).MetAxis()<<"  ";
    std::cout<<"nBinVar ("<<definitions().nBinVar()<<")  ";
    for(unsigned j=0;j<definitions().nBinVar();j++)
      std::cout<<record(i).xMin(j)<<" "<<record(i).xMax(j)<<" ";
    std::cout<<"nParameters ("<<record(i).nParameters()<<") ";
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
void METCorrectorParameters::printFile(const std::string& fFileName,const std::string &Section) const
{
  std::ofstream txtFile;
  txtFile.open(fFileName.c_str(),std::ofstream::app);
  txtFile.setf(std::ios::right);
  txtFile<<"["<<Section<<"]"<<"\n";
  txtFile<<"{"<<" "<<definitions().PtclType()<<"  "<<definitions().nBinVar();
  for(unsigned i=0;i<definitions().nBinVar();i++)
    txtFile<<"  "<<definitions().binVar(i);
  txtFile<<"  "<<definitions().nParVar();
  for(unsigned i=0;i<definitions().nParVar();i++)
    txtFile<<"  "<<definitions().parVar(i);
  txtFile<<"  "<<definitions().formula();
  txtFile<<"}"<<"\n";
  for(unsigned i=0;i<size();i++) //mRecords size
  {
    txtFile<<record(i).MetAxis();
    for(unsigned j=0;j<definitions().nBinVar();j++)
      txtFile<<"  "<<record(i).xMin(j)<<"  "<<record(i).xMax(j);
    txtFile<<"  "<<record(i).nParameters();
    for(unsigned j=0;j<record(i).nParameters();j++)
      txtFile<<"  "<<record(i).parameter(j);
    txtFile<<"\n";
  }
  txtFile.close();
}

namespace {
const std::vector<std::string> labels_ = {
  "XYshiftMC",
  "XYshiftDY",
  "XYshiftTTJets",
  "XYshiftWJets",
  "XYshiftData"
};
// Not used
const std::vector<std::string> MiniAodSource_ = {
  "MiniAod_ShiftX",
  "MiniAod_ShiftY"
};
const std::vector<std::string> XYshiftFlavors_ = {
  "hEtaPlus",
  "hEtaMinus",
  "h0Barrel",
  "h0EndcapPlus",
  "h0EndcapMinus",
  "gammaBarrel",
  "gammaEndcapPlus",
  "gammaEndcapMinus",
  "hHFPlus",
  "hHFMinus",
  "egammaHFPlus",
  "egammaHFMinus"
};

}//namespace

std::string
METCorrectorParametersCollection::findLabel( key_type k ){
  if( isXYshiftMC(k) ){
    return findXYshiftMCflavor(k);
  }else if( isXYshiftDY(k) ){
    return findXYshiftDYflavor(k);
  }else if( isXYshiftTTJets(k) ){
    return findXYshiftTTJetsFlavor(k);
  }else if( isXYshiftWJets(k) ){
    return findXYshiftWJetsFlavor(k);
  }else if( isXYshiftData(k) ){
    return findXYshiftDataFlavor(k);
  }
  return labels_[k];
}

std::string
METCorrectorParametersCollection::levelName( key_type k ){
  if( isXYshiftMC(k) ){
    return labels_[XYshiftMC];
  }else if( isXYshiftDY(k) ){
    return labels_[XYshiftDY];
  }else if( isXYshiftTTJets(k) ){
    return labels_[XYshiftTTJets];
  }else if( isXYshiftWJets(k) ){
    return labels_[XYshiftWJets];
  }else if( isXYshiftData(k) ){
    return labels_[XYshiftData];
  }else{ return "Can't find the level name !!!!";}

}

std::string
METCorrectorParametersCollection::findXYshiftMCflavor( key_type k)
{
  if( k == XYshiftMC) return labels_[XYshiftMC];
  else
    return XYshiftFlavors_[k - (XYshiftMC+1)*100 -1];
}
std::string
METCorrectorParametersCollection::findXYshiftDYflavor( key_type k)
{
  if( k == XYshiftDY) return labels_[XYshiftDY];
  else
    return XYshiftFlavors_[k - (XYshiftDY+1)*100 -1];
}
std::string
METCorrectorParametersCollection::findXYshiftTTJetsFlavor( key_type k)
{
  if( k == XYshiftTTJets) return labels_[XYshiftTTJets];
  else
    return XYshiftFlavors_[k - (XYshiftTTJets+1)*100 -1];
}
std::string
METCorrectorParametersCollection::findXYshiftWJetsFlavor( key_type k)
{
  if( k == XYshiftWJets) return labels_[XYshiftWJets];
  else
    return XYshiftFlavors_[k - (XYshiftWJets+1)*100 -1];
}
std::string
METCorrectorParametersCollection::findXYshiftDataFlavor( key_type k)
{
  if( k == XYshiftData) return labels_[XYshiftData];
  else
    return XYshiftFlavors_[k - (XYshiftData+1)*100 -1];
}

// Obsolete
std::string
METCorrectorParametersCollection::findMiniAodSource( key_type k)
{
  //if( k == MiniAod) return labels_[MiniAod];
  //else
  //  return MiniAodSource_[k - MiniAod*100 -1];
  return MiniAodSource_[0];
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
  copy(outputs.begin(),outputs.end(), std::ostream_iterator<std::string>(std::cout, "\n") );
}

// Add a METCorrectorParameter object. 
void METCorrectorParametersCollection::push_back( key_type i, value_type const & j, label_type const &flav )
{
  if( isXYshiftMC(i))
  {
    correctionsXYshift_.push_back( pair_type(getXYshiftMcFlavBin(flav),j) );
  }else if( isXYshiftDY(i))
  {
    correctionsXYshift_.push_back( pair_type(getXYshiftDyFlavBin(flav),j) );
  }else if( isXYshiftTTJets(i))
  {
    correctionsXYshift_.push_back( pair_type(getXYshiftTTJetsFlavBin(flav),j) );
  }else if( isXYshiftWJets(i))
  {
    correctionsXYshift_.push_back( pair_type(getXYshiftWJetsFlavBin(flav),j) );
  }else if( isXYshiftData(i))
  {
    correctionsXYshift_.push_back( pair_type(getXYshiftDataFlavBin(flav),j) );
  }else{
    std::stringstream sserr;
    sserr<<"The level type: "<<i<<" is not in the level list";
    handleError("METCorrectorParameters::Definitions",sserr.str());
  }

}

// Access the METCorrectorParameter via the key k.
// key_type is hashed to deal with the three collections
METCorrectorParameters const & METCorrectorParametersCollection::operator[]( key_type k ) const {
  collection_type::const_iterator ibegin, iend, i;
  if ( isXYshiftMC(k) || isXYshiftDY(k) || isXYshiftTTJets(k) || isXYshiftWJets(k) || isXYshiftData(k) ) {
    ibegin = correctionsXYshift_.begin();
    iend = correctionsXYshift_.end();
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
  for ( collection_type::const_iterator ibegin = correctionsXYshift_.begin(),
	  iend = correctionsXYshift_.end(), i = ibegin; i != iend; ++i ) {
    keys.push_back( i->first );
  }
}

// Obsolete
METCorrectorParametersCollection::key_type
METCorrectorParametersCollection::getMiniAodBin( std::string const & source ){
  //std::vector<std::string>::const_iterator found =
  //  find( MiniAodSource_.begin(), MiniAodSource_.end(), source );
  //if ( found != MiniAodSource_.end() ) {
  //  return (found - MiniAodSource_.begin() + 1)+ MiniAod * 100;
  //}
  //else return MiniAod;
  return 0;
}


METCorrectorParametersCollection::key_type
METCorrectorParametersCollection::getXYshiftMcFlavBin( std::string const & flav ){
  std::vector<std::string>::const_iterator found =
    find( XYshiftFlavors_.begin(), XYshiftFlavors_.end(), flav );
  if ( found != XYshiftFlavors_.end() ) {
    return (found - XYshiftFlavors_.begin() + 1)+ (XYshiftMC+1) * 100;
  }
  else{
    throw cms::Exception("InvalidInput") <<
    "************** Can't find XYshiftSection: "<<flav<<std::endl;
  }
  return 0;
}

METCorrectorParametersCollection::key_type
METCorrectorParametersCollection::getXYshiftDyFlavBin( std::string const & flav ){
  std::vector<std::string>::const_iterator found =
    find( XYshiftFlavors_.begin(), XYshiftFlavors_.end(), flav );
  if ( found != XYshiftFlavors_.end() ) {
    return (found - XYshiftFlavors_.begin() + 1)+ (XYshiftDY+1) * 100;
  }
  else{
    throw cms::Exception("InvalidInput") <<
    "************** Can't find XYshiftSection: "<<flav<<std::endl;
  }
  return 0;
}
METCorrectorParametersCollection::key_type
METCorrectorParametersCollection::getXYshiftTTJetsFlavBin( std::string const & flav ){
  std::vector<std::string>::const_iterator found =
    find( XYshiftFlavors_.begin(), XYshiftFlavors_.end(), flav );
  if ( found != XYshiftFlavors_.end() ) {
    return (found - XYshiftFlavors_.begin() + 1)+ (XYshiftTTJets+1) * 100;
  }
  else{
    throw cms::Exception("InvalidInput") <<
    "************** Can't find XYshiftSection: "<<flav<<std::endl;
  }
  return 0;
}
METCorrectorParametersCollection::key_type
METCorrectorParametersCollection::getXYshiftWJetsFlavBin( std::string const & flav ){
  std::vector<std::string>::const_iterator found =
    find( XYshiftFlavors_.begin(), XYshiftFlavors_.end(), flav );
  if ( found != XYshiftFlavors_.end() ) {
    return (found - XYshiftFlavors_.begin() + 1)+ (XYshiftWJets+1) * 100;
  }
  else{
    throw cms::Exception("InvalidInput") <<
    "************** Can't find XYshiftSection: "<<flav<<std::endl;
  }
  return 0;
}
METCorrectorParametersCollection::key_type
METCorrectorParametersCollection::getXYshiftDataFlavBin( std::string const & flav ){
  std::vector<std::string>::const_iterator found =
    find( XYshiftFlavors_.begin(), XYshiftFlavors_.end(), flav );
  if ( found != XYshiftFlavors_.end() ) {
    return (found - XYshiftFlavors_.begin() + 1)+ (XYshiftData+1) * 100;
  }
  else{
    throw cms::Exception("InvalidInput") <<
    "************** Can't find XYshiftSection: "<<flav<<std::endl;
  }
  return 0;
}

// Not used
bool METCorrectorParametersCollection::isMiniAod( key_type k ) {
  //return k == MiniAod ||
  //  (k > MiniAod*100 && k < MiniAod*100 + 100);
  return 0;
}

bool METCorrectorParametersCollection::isXYshiftMC( key_type k ) {
  return k == XYshiftMC ||
    (k > (XYshiftMC+1)*100 && k < (XYshiftMC + 2)*100 );
}
bool METCorrectorParametersCollection::isXYshiftDY( key_type k ) {
  return k == XYshiftDY ||
    (k > (XYshiftDY+1)*100 && k < (XYshiftDY + 2)*100 );
}
bool METCorrectorParametersCollection::isXYshiftTTJets( key_type k ) {
  return k == XYshiftTTJets ||
    (k > (XYshiftTTJets+1)*100 && k < (XYshiftTTJets + 2)*100 );
}
bool METCorrectorParametersCollection::isXYshiftWJets( key_type k ) {
  return k == XYshiftWJets ||
    (k > (XYshiftWJets+1)*100 && k < (XYshiftWJets + 2)*100 );
}
bool METCorrectorParametersCollection::isXYshiftData( key_type k ) {
  return k == XYshiftData ||
    (k > (XYshiftData+1)*100 && k < (XYshiftData + 2)*100 );
}


#include "FWCore/Utilities/interface/typelookup.h"
 
TYPELOOKUP_DATA_REG(METCorrectorParameters);
TYPELOOKUP_DATA_REG(METCorrectorParametersCollection);
