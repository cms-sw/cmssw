#include "CondFormats/EcalObjects/interface/EcalSRSettings.h"
#include "FWCore/Utilities/src/Exception.cc"

#include <sstream>
#include <iostream>
#include <stdlib.h>
#include <cassert>
#include <algorithm>

using namespace std;

static int tccNum[12][12] = {
  /* EE- */
  {  36, 19, 20, 21, 22, 23, 18,  1,  2,  3,  4,  5}, //SRP 1
  {  24, 25, 26, 27, 28, 29,  6,  7,  8,  9, 10, 11}, //SRP 2
  {  30, 31, 32, 33, 34, 35, 12, 13, 14, 15, 16, 17}, //SRP 3
  /* EB- */
  { 54, 37, 38, 39, 40, 41, -1, -1, -1, -1, -1, -1},  //SRP 4
  { 42, 43, 44, 45, 46, 47, -1, -1, -1, -1, -1, -1},  //SRP 5
  { 48, 49, 50, 51, 52, 53, -1, -1, -1, -1, -1, -1},  //SRP 6
  /* EB+ */
  { 72, 55, 56, 57, 58, 59, -1, -1, -1, -1, -1, -1},  //SRP 7
  { 60, 61, 62, 63, 64, 65, -1, -1, -1, -1, -1, -1},  //SRP 8
  { 66, 67, 68, 69, 70, 71, -1, -1, -1, -1, -1, -1},  //SRP 9
  /* EE+ */
  {  90, 73, 74, 75, 76, 77,108, 91, 92, 93, 94, 95}, //SRP 10
  {  78, 79, 80, 81, 82, 83, 96, 97, 98, 99,100,101}, //SRP 11
  {  84, 85, 86, 87, 88, 89,102,103,104,105,106,107}   //SRP 12
};

static int dccNum[12][12] = {
  {  1,  2,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1},  //SRP 1 
  {  4,  5,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1},  //SRP 2 
  {  7,  8,  9, -1, -1, -1, -1, -1, -1, -1, -1, -1},  //SRP 3 
  { 10, 11, 12, 13, 14, 15, -1, -1, -1, -1, -1, -1},  //SRP 4 
  { 16, 17, 18, 19, 20, 21, -1, -1, -1, -1, -1, -1},  //SRP 5 
  { 22, 23, 24, 25, 26, 27, -1, -1, -1, -1, -1, -1},  //SRP 6 
  { 28, 29, 30, 31, 32, 33, -1, -1, -1, -1, -1, -1},  //SRP 7 
  { 34, 35, 36, 37, 38, 39, -1, -1, -1, -1, -1, -1},  //SRP 8 
  { 40, 41, 42, 43, 44, 45, -1, -1, -1, -1, -1, -1},  //SRP 9 
  { 46, 47, 48, -1, -1, -1, -1, -1, -1, -1, -1, -1},  //SRP 10
  { 49, 50, 51, -1, -1, -1, -1, -1, -1, -1, -1, -1},  //SRP 11
  { 52, 53, 54, -1, -1, -1, -1, -1, -1, -1, -1, -1}  //SRP 12
};

EcalSRSettings::EcalSRSettings():
  ebDccAdcToGeV_(0.),
  eeDccAdcToGeV_(0.),
  bxGlobalOffset_(0),
  automaticMasks_(0),
  automaticSrpSelect_(0)
{}


void EcalSRSettings::importSrpConfigFile(std::istream& f, bool d){
  //initialize vectors:
  deltaEta_ = vector<int>(1,0);
  deltaPhi_ = vector<int>(1,0);
  actions_ = vector<int>(4, 0);
  tccMasksFromConfig_ = vector<short>(nTccs_, 0);
  srpMasksFromConfig_ = vector<vector<short> >(nSrps_, vector<short>(8, 0));
  dccMasks_ = vector<short>(nDccs_);
  srfMasks_ = vector<short>(nDccs_);
  substitutionSrfs_ = vector<vector<short> >(nSrps_, vector<short>(68,0));
  testerTccEmuSrpIds_ = vector<int>(nSrps_, 0);
  testerSrpEmuSrpIds_ = vector<int>(nSrps_, 0);
  testerDccTestSrpIds_ = vector<int>(nSrps_, 0);
  testerSrpTestSrpIds_ = vector<int>(nSrps_, 0);
  bxOffsets_ = vector<short>(nSrps_, 0);
  automaticMasks_ = 0;
  automaticSrpSelect_ = 0;
  
  //string line;
  int iLine = 0;
  int iValueSet = -1;
  const int nValueSets = 6*nSrps_+9;
  string line;
  stringstream sErr("");
  while(!f.eof() && sErr.str().empty()){
    getline(f, line);
    ++iLine;
    line = trim(line);
    if(line[0] == '#' || line.empty()){//comment line and empty line to ignore
      continue;
    } else{
      ++iValueSet;
    }
    if(iValueSet>=nValueSets) break;
    uint32_t value;
    string sValue;
    int pos = 0;
    int iCh = 0;
    int nChs[nValueSets] = {
      //TCC masks: 0-11
      12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
      //SRP masks: 12-23
      8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,
      //DCC masks: 24-35
      12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
      //SRF Masks: 36-47
      6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,
      //substitution SRFs: 48-59
      68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68,
      //Tester card to emulate or test: 60-71
      4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,
      //Bx offsets: 72
      12,
      //algo type: 73
      1,
      //action flags: 74
      4,
      //pattern file directory: 75
      1,
      //VME slots: 76
      12,
      //card types: 77
      12,
      //config Mode
      1,
      //VME Interface card
      1,
      //Spy Mode
      12,
    };

    while(((sValue = tokenize(line, " \t", pos))!=string(""))
	  && (iCh<nChs[iValueSet]) && sErr.str().empty()){
      value = strtoul(sValue.c_str(), 0, 0);
      const int iSrp = iValueSet%nSrps_;
      if(iValueSet<12){//TCC
	assert((unsigned)iSrp < sizeof(tccNum) / sizeof(tccNum[0]));
	assert((unsigned)iCh < sizeof(tccNum[0]) / sizeof(tccNum[0][0]));
	int tcc = tccNum[iSrp][iCh];
	if(tcc>=0) {
	  if(d) cout << "tccMasksFromConfig_[" << tcc << "] <- "
		     << value << "\n";
	  tccMasksFromConfig_[tcc-1] = value;
	}
      } else if(iValueSet<24){//SRP-SRP
        if(d) cout << "srpMasks_[" << iSrp << "][" << iCh << "] <- "
		   << value << "\n";
	srpMasksFromConfig_[iSrp][iCh] = value;
      } else if(iValueSet<36) {//DCC output
	assert((unsigned)iSrp < sizeof(dccNum) / sizeof(dccNum[0]));
	assert((unsigned)iCh < sizeof(dccNum[0]) / sizeof(dccNum[0][0]));
	int dcc = dccNum[iSrp][iCh];
	if(dcc > 0){
	  assert((unsigned)(dcc-1) < dccMasks_.size());
	  if(d) cout << "dccMasks_[" << (dcc-1) << "] <- "
		     << value << "\n";
	  dccMasks_[dcc-1] = value;
	}
      } else if(iValueSet<48){//SRF masks
	assert((unsigned)iSrp < sizeof(dccNum) / sizeof(dccNum[0]));
	assert((unsigned)iCh < sizeof(dccNum[0]) / sizeof(dccNum[0][0]));
	int dcc = dccNum[iSrp][iCh];
	if(dcc > 0){
	  if(d) cout << "srfMasks_[" << (dcc-1) << "] <- "
		     << value << "\n";
	  assert((unsigned)(dcc-1) < srfMasks_.size());
	  srfMasks_[dcc-1] = value;
	}
      } else if(iValueSet<60){//substiution SRFs
	assert((unsigned)iSrp < substitutionSrfs_.size());
	assert((unsigned)iCh <  substitutionSrfs_[0].size());
	if(d) cout << "substitutionMasks_[" << iSrp << "][" << iCh << "] <- "
		   << value << "\n";
	substitutionSrfs_[iSrp][iCh] = value;
      } else if(iValueSet<72){//Tester card config
	switch(iCh){
	case 0:
	  assert((unsigned)iSrp < testerTccEmuSrpIds_.size());
	  if(d) cout << "testerTccEmuSrpIds_[" << iSrp << "] <- "
		     << value << "\n";
	  testerTccEmuSrpIds_[iSrp] = value;
	  break;
	case 1:
	  assert((unsigned)iSrp < testerSrpEmuSrpIds_.size());
	  if(d) cout << "testerSrpEmuSrpIds_[" << iSrp << "] <- "
		     << value << "\n";
	  testerSrpEmuSrpIds_[iSrp] = value;
	  break;
	case 2:
	  assert((unsigned)iSrp < testerDccTestSrpIds_.size());
	  if(d) cout << "testerDccTestSrpIds_[" << iSrp << "] <- "
		     << value << "\n";
	  testerDccTestSrpIds_[iSrp] = value;
	  break;
	case 3:
	  assert((unsigned)iSrp < testerSrpTestSrpIds_.size());
	  if(d) cout << "testerSrpTestSrpIds_[" << iSrp << "] <- "
		     << value << "\n";
	  testerSrpTestSrpIds_[iSrp] = value;
	  break;
	default:
	  sErr << "Syntax error in SRP system configuration "
	       << " line " << iLine << ".";
	}
      } else if(iValueSet<73){//bx offsets
	assert((unsigned)iCh < bxOffsets_.size());
	if(d) cout << "bxOffset_[" << iCh << "] <- "
		   << value << "\n";
	bxOffsets_[iCh] = value;
      } else if(iValueSet<74){//algo type
	int algo = value;
	switch(algo){
	case 0:
	  deltaEta_[0] = deltaPhi_[0] = 1;
	  break;
	case 1:
	  deltaEta_[0] = deltaPhi_[0]  = 2;
	  break;
	default:
	  throw cms::Exception("OutOfRange") << "Value of parameter algo ," << algo
					     << ", is invalid. Valid values are 0 and 1.";
	}
	if(d) cout << "deltaEta_[0] <- " << deltaEta_[0] << "\t"
		   << "deltaPhi_[0] <- " << deltaPhi_[0] << "\n";
      } else if(iValueSet<75){//action flags
	assert((unsigned)iCh < actions_.size());
	if(d) cout << "actions_[" << iCh << "] <- "
		   << value << "\n";
	actions_[iCh] = value;
      } else if(iValueSet<76){//pattern file directory
// 	emuDir_ = sValue;
// 	if(d) cout << "emuDir_ <= "
// 		   << value << "\n";
      } else if(iValueSet<77){//VME slots
// 	slotIds_[iCh] = value;
// 	if(d) cout << "slotIds_[" << iCh << "] <= "
// 		   << value << "\n";
      } else if(iValueSet<78){//card types
// 	cardTypes_[iCh] = sValue[0];
// 	if(d) cout << "cardTypes_[" << iCh << "] <= "
// 		   << value << "\n";
      } else if (iValueSet<79){//config mode
	//TODO validity check on value
// 	configMode_ = (ConfigMode)value;
// 	if(d) cout << "config mode <= " << value << "\n";
      } else if (iValueSet<80){//VME I/F
	//TODO validity check on value
	//	vmeInterface_ = (Vme::type_t)value;
	//if(d) cout << "Vme Interface code <= " << value << "\n";
      } else if (iValueSet<81){//Spy Mode
	//TODO validity check on value
	//	spyMode_[iCh] = value & 0x3;
	//	if(d) cout << "Spy mode <= " << value << "\n";
      } else{//should never be reached!
	assert(false);
      }
      ++iCh;
    }
    if(iCh!=nChs[iValueSet]){//error
      sErr << "Syntax error in imported SRP system configuration file "
	/*<< filename <<*/ " line " << iLine << ".";
    }
  }
  if(sErr.str().empty() && iValueSet!=(nValueSets-1)){//error
    sErr << "Syntax Error in imported SRP system configuration file "
      /*<< filename <<*/ " line " << iLine << ".";
  }
  if(sErr.str().size()!=0) throw cms::Exception("SyntaxError") << sErr.str();
}

double EcalSRSettings::normalizeWeights(int hwWeight){
  //Fix sign bit in case only the 12 least significant bits of hwWeight were set
  //(hardware reprensentation uses only 12 bits)
  if(hwWeight & (1<<11)) hwWeight |= ~0xEFF;
  return hwWeight/1024.;
}

string EcalSRSettings::tokenize(const string& s, const string& delim, int& pos){
  if(pos<0) return "";
  int pos0 = pos;
  int len = s.size();
  //eats delimeters at beginning of the string
  while(pos0<len && find(delim.begin(), delim.end(), s[pos0])!=delim.end()){
    ++pos0;
  }
  if(pos0==len) return "";
  pos = s.find_first_of(delim, pos0);
  return s.substr(pos0, (pos>0?pos:len)-pos0);
}

std::string EcalSRSettings::trim(std::string s){
  std::string::size_type pos0 = s.find_first_not_of(" \t");
  if(pos0==string::npos){
    pos0=0;
  }
  string::size_type pos1 = s.find_last_not_of(" \t") + 1;
  if(pos1==string::npos){
    pos1 = pos0;
  }
  return s.substr(pos0, pos1-pos0);
}

#define SR_PRINT(a) o << #a ": " << val.a ## _ << "\n";
#define SR_VPRINT(a) o << #a ; \
  if(val.a ## _.size()) o << "[0.." << (val.a ## _.size() - 1) << "]:";	\
  else o << "[]: <empty>"; \
  for(size_t i = 0; i < val.a ## _.size(); ++i) o << "\t" << val.a ## _[i]; \
  o << "\n";
#define SR_VVPRINT(a) \
  if(val.a ## _.size() == 0) o << #a "[][]: <empty>\n"; \
  for(size_t i = 0; i < val.a ## _.size(); ++i){ \
    o << #a "[" << i << "]"; \
    if(val.a ## _.size()) o << "[0.." << (val.a ## _[i].size() -1 ) << "]:"; \
    else o << "[]: <empty>"; \
    for(size_t j = 0; j < val.a ## _[i].size(); ++j) o << "\t" << val.a ## _[i][j]; \
    o << "\n";								\
  }


void EcalSRSettings::importParameterSet(const edm::ParameterSet& ps){
  deltaPhi_.resize(1);
  deltaPhi_[0] = ps.getParameter<int >("deltaPhi");
  deltaEta_.resize(1);
  deltaEta_[0] = ps.getParameter<int >("deltaEta");
  ecalDccZs1stSample_.resize(1);
  ecalDccZs1stSample_[0] = ps.getParameter<int >("ecalDccZs1stSample");
  ebDccAdcToGeV_ = ps.getParameter<double >("ebDccAdcToGeV");
  eeDccAdcToGeV_ = ps.getParameter<double >("eeDccAdcToGeV");
  dccNormalizedWeights_.resize(1);
  dccNormalizedWeights_[0] = ps.getParameter<std::vector<double>  >("dccNormalizedWeights");
  symetricZS_.resize(1);
  symetricZS_[0] = ps.getParameter<bool >("symetricZS");
  srpLowInterestChannelZS_.resize(2);
  const int eb = 0;
  const int ee = 1;
  srpLowInterestChannelZS_[eb] = ps.getParameter<double >("srpBarrelLowInterestChannelZS");
  srpLowInterestChannelZS_[ee] = ps.getParameter<double >("srpEndcapLowInterestChannelZS");
  srpHighInterestChannelZS_.resize(2);
  srpHighInterestChannelZS_[eb] = ps.getParameter<double >("srpBarrelHighInterestChannelZS");
  srpHighInterestChannelZS_[ee] = ps.getParameter<double >("srpEndcapHighInterestChannelZS");
  //trigPrimBypass_.resize(1);
  //trigPrimBypass_[0] = ps.getParameter<bool >("trigPrimBypass");
  //trigPrimBypassMode_.resize(1);
  //trigPrimBypassMode_[0] = ps.getParameter<int >("trigPrimBypassMode");
  //trigPrimBypassLTH_.resize(1);
  //trigPrimBypassLTH_[0] = ps.getParameter<double >("trigPrimBypassLTH");
  //trigPrimBypassHTH_.resize(1);
  //trigPrimBypassHTH_[0] = ps.getParameter<double >("trigPrimBypassHTH");
  //trigPrimBypassWithPeakFinder_.resize(1);
  //trigPrimBypassWithPeakFinder_[0] = ps.getParameter<bool >("trigPrimBypassWithPeakFinder");
  //defaultTtf_.resize(1);
  //defaultTtf_[0] = ps.getParameter<int >("defaultTtf");
  actions_ = ps.getParameter<std::vector<int> >("actions");
}

std::ostream& operator<< (std::ostream& o, const EcalSRSettings& val){
  o << "# Neighbour eta range, neighborhood: (2*deltaEta+1)*(2*deltaPhi+1)\n"
    "# In the vector contains:\n"
    "#   - 1 element, then value applies to whole ECAL\n"
    "#   - 2 elements, then element 0 applies to EB, element 1 to EE\n"
    "#   - 12 elements, then element i applied to SRP (i+1)\n"
    "# SRP emulation (see SimCalorimetry/EcalSelectiveReadoutProcuders) supports\n"
    "# only 1 element mode.\n";
  SR_VPRINT(deltaEta);
    
  o << "\n# Neighbouring eta range, neighborhood: (2*deltaEta+1)*(2*deltaPhi+1)\n"
    "# If the vector contains...\n"
    "#   ... 1 element, then value applies to whole ECAL\n"
    "#   ... 2 elements, then element 0 applies to EB, element 1 to EE\n"
    "#   ... 12 elements, then element i applied to SRP (i+1)\n"
    "# If the vector contains...\n"
    "#   ... 1 element, then value applies to whole ECAL\n"
    "#   ... 2 elements, then element 0 applies to EB, element 1 to EE\n"
    "#   ... 12 elements, then element i applied to SRP (i+1)\n"
    "# SRP emulation (see SimCalorimetry/EcalSelectiveReadoutProcuders) supports\n"
    "# only the single-element mode.\n";
  SR_VPRINT(deltaPhi);
    
  o << "\n# Index of time sample (staring from 1) the first DCC weights is implied\n"
    "# If the vector contains:\n"
    "#   ... 1 element, then value applies to whole ECAL\n"
    "#   ... 2 elements, then element 0 applies to EB, element 1 to EE\n"
    "#   ... 54 elements, then element i applied to DCC (i+1) (FED ID 651+i)\n"
    "# SRP emulation (see SimCalorimetry/EcalSelectiveReadoutProcuders) supports\n"
    "# only the single-element mode.\n";
  SR_VPRINT(ecalDccZs1stSample);

  o << "\n# ADC to GeV conversion factor used in ZS filter for EB\n";
  SR_PRINT(ebDccAdcToGeV);
  
  o << "\n# ADC to GeV conversion factor used in ZS filter for EE\n";
  SR_PRINT(eeDccAdcToGeV);

  o << "\n# DCC ZS FIR weights: weights are rounded in such way that in Hw\n"
    "# representation (weigth*1024 rounded to nearest integer) the sum is null:\n"
    "# Each element is a vector of 6 values, the 6 weights\n"
    "# If the vector contains...\n"
    "#   ... 1 element, then the weight set applies to whole ECAL\n"
    "#   ... 2 elements, then element 0 applies to EB, element 1 to EE\n"
    "#   ... 54 elements, then element i applied to DCC (i+1) (FED ID 651+i)\n";
  SR_VVPRINT(dccNormalizedWeights);
  
  o << "\n# Switch to use a symetric zero suppression (cut on absolute value). For\n"
    "# studies only, for time being it is not supported by the hardware.\n"
    "# having troubles for vector<bool> with coral (3.8.0pre1), using vector<int> instead,\n"
    "# 0 means false, a value different than 0 means true.\n"
    "# If the vector contains...\n"
    "#   ... 1 element, then the weight set applies to whole ECAL\n"
    "#   ... 2 elements, then element 0 applies to EB, element 1 to EE\n"
    "#   ... 54 elements, then element i applied to DCC (i+1) (FED ID 651+i)\n"
    "#   ... 75848 elements, then:\n"
    "#          for i < 61200, element i applies to EB crystal with denseIndex i\n"
    "#                         (see EBDetId::denseIndex())\n"
    "#          for i >= 61200, element i applies to EE crystal with denseIndex (i+61200)\n"
    "#                         (see EBDetId::denseIndex())\n"
    "# SRP emulation supports only 1 element mode. Hardware does not support\n"
    "# the symetric ZS, so symetricZS = 0 for real data.\n";
  SR_VPRINT(symetricZS);

  o << "\n# ZS energy threshold in GeV to apply to low interest channels of barrel\n"
    "# If the vector contains...\n"
    "#   ... 1 element, then the weight set applies to whole ECAL\n"
    "#   ... 2 elements, then element 0 applies to EB, element 1 to EE\n"
    "#   ... 54 elements, then element i applied to DCC (i+1) (FED ID 651+i)\n"
    "# SRP emulation supports only the 2-element mode.\n"
    "# Corresponds to srpBarrelLowInterestChannelZS and srpEndcapLowInterestChannelZS\n"
    "# of python configuration file parameters\n";
  SR_VPRINT(srpLowInterestChannelZS);
    
  o << "\n# ZS energy threshold in GeV to apply to high interest channels of endcap\n"
    "# If the vector contains...\n"
    "#   ... 1 element, then the weight set applies to whole ECAL\n"
    "#   ... 2 elements, then element 0 applies to EB, element 1 to EE\n"
    "#   ... 54 elements, then element i applied to DCC (i+1) (FED ID 651+i)\n"
    "# SRP emulation supports only the 2-element mode.\n"
    "# Corresponds to srpBarrelLowInterestChannelZS and srpEndcapLowInterestChannelZS\n"
    "# of python configuration file parameters\n";
  SR_VPRINT(srpHighInterestChannelZS);
  
  //  o << "\n# Switch to run w/o trigger primitive. For debug use only\n"
  //  "# having troubles for vector<bool> with coral (3.8.0pre1), using vector<int> instead\n"
  //  "# Parameter only relevant for emulation. For real data, must be contains 1 element with\n"
  //  "# value 0.\n"
  //  "#   ... 1 element, then the weight set applies to whole ECAL\n"
  //  "#   ... 2 elements, then element 0 applies to EB, element 1 to EE\n"
  //  "#   ... 54 elements, then element i applied to DCC (i+1) (FED ID 651+i)\n"
  //  "# SRP emulation supports only the single-element mode.\n";
  //  SR_VPRINT(trigPrimBypass);\n"
  //  
  //  o << "\n# Mode selection for "# Trig bypass" mode\n"
  //  "# 0: TT thresholds applied on sum of crystal Et's\n"
  //  "# 1: TT thresholds applies on compressed Et from Trigger primitive\n"
  //  "# @see trigPrimByPass switch\n"
  //  "# Parameter only relevant for \n";
  //  SR_VPRINT(trigPrimBypassMode);
  // 
  //  o << "\n# for debug mode only:\n";
  //  SR_VPRINT( trigPrimBypassLTH);
  //
  //  o << "\n# for debug mode only:\n";
  //  SR_VPRINT(trigPrimBypassHTH);
  //
  //  o << "\n# for debug mode only\n"
  //  "# having troubles for vector<bool> with coral (3.8.0pre1), using vector<int> instead\n";
  //  SR_VPRINT( trigPrimBypassWithPeakFinder);
  //
  //  o << "\n# Trigger Tower Flag to use when a flag is not found from the input\n"
  //  "# Trigger Primitive collection. Must be one of the following values:\n"
  //  "# 0: low interest, 1: mid interest, 3: high interest\n"
  //  "# 4: forced low interest, 5: forced mid interest, 7: forced high interest\n";
  //  SR_VPRINT(defaultTtf);\n"

  o << "\n# SR->action flag map. 4 elements\n"
    "# action_[i]: action for flag value i\n";
  SR_VPRINT(actions);

  o << "\n# Masks for TTC inputs of SRP cards\n"
    "# One element per TCC, that is 108 elements: element i applies to TCC (i+1)\n";
  SR_VPRINT(tccMasksFromConfig);

  o << "\n# Masks for SRP-SRP inputs of SRP cards\n"
    "# One element per SRP, that is 12 elements: element i applies to SRP (i+1)\n"
    "# indices: [iSrp][iCh]\n";
  SR_VVPRINT(srpMasksFromConfig);

  o << "\n# Masks for DCC output of SRP cards\n"
    "# One element per DCC, that is 54 elements: element i applies to DCC (i+1)\n";
  SR_VPRINT(dccMasks);

  o << "\n# Mask to enable pattern test. Typical value: 0.\n"
    "# One element per SRP, that is 12 elements: element i applies to SRP (i+1)\n";
  SR_VPRINT(srfMasks);

  o << "\n# Substitution flags used in patterm mode\n"
    "# indices: [iSrp][iFlag]\n";
  SR_VVPRINT(substitutionSrfs);
  
  o << "\n# Tester mode configuration\n";
  SR_VPRINT(testerTccEmuSrpIds);
  SR_VPRINT(testerSrpEmuSrpIds);
  SR_VPRINT(testerDccTestSrpIds);
  SR_VPRINT(testerSrpTestSrpIds);
  //@}
  
  o << "\n# Per SRP card bunch crossing counter offset.\n"
    "# This offset is added to the bxGlobalOffset\n";
  SR_VPRINT(bxOffsets);

  o << "\n# SRP system bunch crossing counter offset.\n"
    "# For each card the bxOffset[i]\n"
    "# is added to this one.\n";
  SR_PRINT(bxGlobalOffset);

  o << "\n# Switch for automatic channel masking. 0: disabled; 1: enabled. Standard  configuration: 1.\n"
    "# When enabled, if a FED is excluded from the run, the corresponding TCC inputs is automatically\n"
    "# masked (overwrites the tccInputMasks).\n";
  SR_PRINT(automaticMasks);
  
  o << "\n# Switch for automatic SRP card selection. 0: disabled; 1 : enabled..\n"
    "# When enabled, if all the FEDs corresponding to a given SRP is excluded from the run,\n"
    "# Then the corresponding SRP card is automatically excluded.\n";
  SR_PRINT(automaticSrpSelect);

  return o;
}

void EcalSRSettings::checkValidity(bool forEmulator) const{
  if(forEmulator){
    if(this->dccNormalizedWeights_.size() != 1){
      throw cms::Exception("Configuration") << "Selective readout emulator, EcalSelectiveReadout, supports only single set of ZS weights. "
      "while the configuration contains " << this->dccNormalizedWeights_.size() << " set(s)\n";
    }
  }
  
  if(this->dccNormalizedWeights_.size() != 1
     && this->dccNormalizedWeights_.size() != 2
     && this->dccNormalizedWeights_.size() != 54
     && this->dccNormalizedWeights_.size() != 75848){
    throw cms::Exception("Configuration") << "Invalid number of DCC weight set (" << this->dccNormalizedWeights_.size()
					  << ") in condition object EcalSRSetting::dccNormalizedWeights_. "
					  << "Valid counts are: 1 (single set), 2 (EB and EE), 54 (one per DCC) and 75848 "
      "(one per crystal)\n";
  }
  
  if(this->dccNormalizedWeights_.size() != this->ecalDccZs1stSample_.size()){
    throw cms::Exception("Configuration") << "Inconsistency between number of weigth sets ("
					  << this->dccNormalizedWeights_.size() << ") and "
					  << "number of ecalDccZs1Sample values ("
					  << this->ecalDccZs1stSample_.size() << ").";
  }  
}
