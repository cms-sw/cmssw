// Generic parameters for MET corrections
//
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CondFormats/JetMETObjects/interface/MEtXYcorrectParameters.h"
#include "CondFormats/JetMETObjects/interface/Utilities.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <iterator>
//#include <string>

//------------------------------------------------------------------------ 
//--- MEtXYcorrectParameters::Definitions constructor --------------------
//--- takes specific arguments for the member variables ------------------
//------------------------------------------------------------------------
MEtXYcorrectParameters::Definitions::Definitions(const std::vector<std::string>& fBinVar, const std::vector<std::string>& fParVar, const std::string& fFormula )
{
  mBinVar.reserve(fBinVar.size());
  for(unsigned i=0;i<fBinVar.size();i++)
    mBinVar.push_back(fBinVar[i]);

  mParVar.reserve(fParVar.size());
  for(unsigned i=0;i<fParVar.size();i++)
    mParVar.push_back(getUnsigned(fParVar[i]));

  mFormula    = fFormula;
}
//------------------------------------------------------------------------
//--- MEtXYcorrectParameters::Definitions constructor --------------------
//--- reads the member variables from a string ---------------------------
//------------------------------------------------------------------------
MEtXYcorrectParameters::Definitions::Definitions(const std::string& fLine)
{
  std::vector<std::string> tokens = getTokens(fLine);
  // corrType N_bin binVa.. N_var var... formula
  if (!tokens.empty())
  { 
    if (tokens.size() < 6) 
    {
      std::stringstream sserr;
      sserr<<"(line "<<fLine<<"): Great than or equal to 6 expected but the number of tokens:"<<tokens.size();
      handleError("MEtXYcorrectParameters::Definitions",sserr.str());
    }
    // No. of Bin Variable
    //edm::LogInfo ("default")<<"Definitions===========";
    ptclType = getSigned(tokens[0]);
    unsigned nBinVar = getUnsigned(tokens[1]);
    unsigned nParVar = getUnsigned(tokens[nBinVar+2]);
    mBinVar.reserve(nBinVar);
    mParVar.reserve(nParVar);

    for(unsigned i=0;i<nBinVar;i++)
    {
      mBinVar.push_back(tokens[i+2]);
    }
    for(unsigned i=0;i<nParVar;i++)
    {
      mParVar.push_back(getUnsigned(tokens[nBinVar+3+i]));
    }
    mFormula = tokens[nParVar+nBinVar+3];
    if (tokens.size() != nParVar+nBinVar+4 ) 
    {
      std::stringstream sserr;
      sserr<<"(line "<<fLine<<"): token size should be:"<<nParVar+nBinVar+4<<" but it is "<<tokens.size();
      handleError("MEtXYcorrectParameters::Definitions",sserr.str());
    }

  }
}
//------------------------------------------------------------------------
//--- MEtXYcorrectParameters::Record constructor -------------------------
//--- reads the member variables from a string ---------------------------
//------------------------------------------------------------------------
MEtXYcorrectParameters::Record::Record(const std::string& fLine,unsigned fNvar) : mMin(0), mMax(0), mParameters(0)
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
      handleError("MEtXYcorrectParameters::Record",sserr.str());
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
      handleError("MEtXYcorrectParameters::Record",sserr.str());
    }
    for (unsigned i = (2*mNvar+2); i < tokens.size(); ++i)
    {
      mParameters.push_back(getFloat(tokens[i]));
    }
  }
}
//------------------------------------------------------------------------
//--- MEtXYcorrectParameters constructor ---------------------------------
//--- reads the member variables from a string ---------------------------
//------------------------------------------------------------------------
MEtXYcorrectParameters::MEtXYcorrectParameters(const std::string& fFile, const std::string& fSection) 
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
    handleError("MEtXYcorrectParameters","No definitions found!!!");
  if (mRecords.empty() && currentSection == "") mRecords.push_back(Record());
  if (mRecords.empty() && currentSection != "") 
  {
    std::stringstream sserr; 
    sserr<<"the requested section "<<fSection<<" doesn't exist!";
    handleError("MEtXYcorrectParameters",sserr.str()); 
  }
  std::sort(mRecords.begin(), mRecords.end());
  valid_ = true;
}
//------------------------------------------------------------------------
//--- prints parameters on screen ----------------------------------------
//------------------------------------------------------------------------
void MEtXYcorrectParameters::printScreen() const
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
void MEtXYcorrectParameters::printScreen(const std::string &Section) const
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
void MEtXYcorrectParameters::printFile(const std::string& fFileName) const
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
void MEtXYcorrectParameters::printFile(const std::string& fFileName,const std::string &Section) const
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
  "shiftMC",
  "shiftDY",
  "shiftTTJets",
  "shiftWJets",
  "shiftData"
};
const std::vector<std::string> shiftFlavors_ = {
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
MEtXYcorrectParametersCollection::findLabel( key_type k ){
  if( isShiftMC(k) ){
    return findShiftMCflavor(k);
  }else if( MEtXYcorrectParametersCollection::isShiftDY(k) ){
    return findShiftDYflavor(k);
  }else if( MEtXYcorrectParametersCollection::isShiftTTJets(k) ){
    return findShiftTTJetsFlavor(k);
  }else if( MEtXYcorrectParametersCollection::isShiftWJets(k) ){
    return findShiftWJetsFlavor(k);
  }else if( MEtXYcorrectParametersCollection::isShiftData(k) ){
    return findShiftDataFlavor(k);
  }
  return labels_[k];
}

std::string
MEtXYcorrectParametersCollection::levelName( key_type k ){
  if( isShiftMC(k) ){
    return labels_[shiftMC];
  }else if( isShiftDY(k) ){
    return labels_[shiftDY];
  }else if( isShiftTTJets(k) ){
    return labels_[shiftTTJets];
  }else if( isShiftWJets(k) ){
    return labels_[shiftWJets];
  }else if( isShiftData(k) ){
    return labels_[shiftData];
  }else{ return "Can't find the level name !!!!";}

}

std::string
MEtXYcorrectParametersCollection::findShiftMCflavor( key_type k)
{
  if( k == shiftMC) return labels_[shiftMC];
  else
    return shiftFlavors_[k - (shiftMC+1)*100 -1];
}
std::string
MEtXYcorrectParametersCollection::findShiftDYflavor( key_type k)
{
  if( k == shiftDY) return labels_[shiftDY];
  else
    return shiftFlavors_[k - (shiftDY+1)*100 -1];
}
std::string
MEtXYcorrectParametersCollection::findShiftTTJetsFlavor( key_type k)
{
  if( k == shiftTTJets) return labels_[shiftTTJets];
  else
    return shiftFlavors_[k - (shiftTTJets+1)*100 -1];
}
std::string
MEtXYcorrectParametersCollection::findShiftWJetsFlavor( key_type k)
{
  if( k == shiftWJets) return labels_[shiftWJets];
  else
    return shiftFlavors_[k - (shiftWJets+1)*100 -1];
}
std::string
MEtXYcorrectParametersCollection::findShiftDataFlavor( key_type k)
{
  if( k == shiftData) return labels_[shiftData];
  else
    return shiftFlavors_[k - (shiftData+1)*100 -1];
}

void MEtXYcorrectParametersCollection::getSections( std::string inputFile,
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
  //copy(outputs.begin(),outputs.end(), std::ostream_iterator<std::string>(std::cout, "\n") );

  std::string sectionNames;
  for(std::vector<std::string>::const_iterator it=outputs.begin(); it!=outputs.end();it++){
    sectionNames+=*it;
    sectionNames+="\n";
  }
  edm::LogInfo ("getSections")<<"Sections read from file: "<<"\n"<<sectionNames;
}

// Add a METCorrectorParameter object. 
void MEtXYcorrectParametersCollection::push_back( key_type i, value_type const & j, label_type const &flav )
{
  if( isShiftMC(i))
  {
    correctionsShift_.push_back( pair_type(getShiftMcFlavBin(flav),j) );
  }else if( isShiftDY(i))
  {
    correctionsShift_.push_back( pair_type(getShiftDyFlavBin(flav),j) );
  }else if( isShiftTTJets(i))
  {
    correctionsShift_.push_back( pair_type(getShiftTTJetsFlavBin(flav),j) );
  }else if( isShiftWJets(i))
  {
    correctionsShift_.push_back( pair_type(getShiftWJetsFlavBin(flav),j) );
  }else if( isShiftData(i))
  {
    correctionsShift_.push_back( pair_type(getShiftDataFlavBin(flav),j) );
  }else{
    std::stringstream sserr;
    sserr<<"The level type: "<<i<<" is not in the level list";
    handleError("MEtXYcorrectParameters::Definitions",sserr.str());
  }

}

// Access the METCorrectorParameter via the key k.
// key_type is hashed to deal with the three collections
MEtXYcorrectParameters const & MEtXYcorrectParametersCollection::operator[]( key_type k ) const {
  collection_type::const_iterator ibegin, iend, i;
  if ( isShiftMC(k) || isShiftDY(k) || isShiftTTJets(k) || isShiftWJets(k) || isShiftData(k) ) {
    ibegin = correctionsShift_.begin();
    iend = correctionsShift_.end();
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
void MEtXYcorrectParametersCollection::validKeys(std::vector<key_type> & keys ) const {
  keys.clear();
  for ( collection_type::const_iterator ibegin = correctionsShift_.begin(),
	  iend = correctionsShift_.end(), i = ibegin; i != iend; ++i ) {
    keys.push_back( i->first );
  }
}



MEtXYcorrectParametersCollection::key_type
MEtXYcorrectParametersCollection::getShiftMcFlavBin( std::string const & flav ){
  std::vector<std::string>::const_iterator found =
    find( shiftFlavors_.begin(), shiftFlavors_.end(), flav );
  if ( found != shiftFlavors_.end() ) {
    return (found - shiftFlavors_.begin() + 1)+ (shiftMC+1) * 100;
  }
  else{
    throw cms::Exception("InvalidInput") <<
    "************** Can't find ShiftSection: "<<flav<<std::endl;
  }
  return 0;
}

MEtXYcorrectParametersCollection::key_type
MEtXYcorrectParametersCollection::getShiftDyFlavBin( std::string const & flav ){
  std::vector<std::string>::const_iterator found =
    find( shiftFlavors_.begin(), shiftFlavors_.end(), flav );
  if ( found != shiftFlavors_.end() ) {
    return (found - shiftFlavors_.begin() + 1)+ (shiftDY+1) * 100;
  }
  else{
    throw cms::Exception("InvalidInput") <<
    "************** Can't find ShiftSection: "<<flav<<std::endl;
  }
  return 0;
}
MEtXYcorrectParametersCollection::key_type
MEtXYcorrectParametersCollection::getShiftTTJetsFlavBin( std::string const & flav ){
  std::vector<std::string>::const_iterator found =
    find( shiftFlavors_.begin(), shiftFlavors_.end(), flav );
  if ( found != shiftFlavors_.end() ) {
    return (found - shiftFlavors_.begin() + 1)+ (shiftTTJets+1) * 100;
  }
  else{
    throw cms::Exception("InvalidInput") <<
    "************** Can't find shiftSection: "<<flav<<std::endl;
  }
  return 0;
}
MEtXYcorrectParametersCollection::key_type
MEtXYcorrectParametersCollection::getShiftWJetsFlavBin( std::string const & flav ){
  std::vector<std::string>::const_iterator found =
    find( shiftFlavors_.begin(), shiftFlavors_.end(), flav );
  if ( found != shiftFlavors_.end() ) {
    return (found - shiftFlavors_.begin() + 1)+ (shiftWJets+1) * 100;
  }
  else{
    throw cms::Exception("InvalidInput") <<
    "************** Can't find shiftSection: "<<flav<<std::endl;
  }
  return 0;
}
MEtXYcorrectParametersCollection::key_type
MEtXYcorrectParametersCollection::getShiftDataFlavBin( std::string const & flav ){
  std::vector<std::string>::const_iterator found =
    find( shiftFlavors_.begin(), shiftFlavors_.end(), flav );
  if ( found != shiftFlavors_.end() ) {
    return (found - shiftFlavors_.begin() + 1)+ (shiftData+1) * 100;
  }
  else{
    throw cms::Exception("InvalidInput") <<
    "************** Can't find shiftSection: "<<flav<<std::endl;
  }
  return 0;
}

bool MEtXYcorrectParametersCollection::isShiftMC( key_type k ) {
  return k == shiftMC ||
    (k > (shiftMC+1)*100 && k < (shiftMC + 2)*100 );
}
bool MEtXYcorrectParametersCollection::isShiftDY( key_type k ) {
  return k == shiftDY ||
    (k > (shiftDY+1)*100 && k < (shiftDY + 2)*100 );
}
bool MEtXYcorrectParametersCollection::isShiftTTJets( key_type k ) {
  return k == shiftTTJets ||
    (k > (shiftTTJets+1)*100 && k < (shiftTTJets + 2)*100 );
}
bool MEtXYcorrectParametersCollection::isShiftWJets( key_type k ) {
  return k == shiftWJets ||
    (k > (shiftWJets+1)*100 && k < (shiftWJets + 2)*100 );
}
bool MEtXYcorrectParametersCollection::isShiftData( key_type k ) {
  return k == shiftData ||
    (k > (shiftData+1)*100 && k < (shiftData + 2)*100 );
}


#include "FWCore/Utilities/interface/typelookup.h"
 
TYPELOOKUP_DATA_REG(MEtXYcorrectParameters);
TYPELOOKUP_DATA_REG(MEtXYcorrectParametersCollection);
