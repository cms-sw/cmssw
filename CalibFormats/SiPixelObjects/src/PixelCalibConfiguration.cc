//
// This class is a first attempt at writing a configuration
// object that will perform a calibration loop.
//
//
//

#include "CalibFormats/SiPixelObjects/interface/PixelCalibConfiguration.h"
#include "CalibFormats/SiPixelObjects/interface/PixelTimeFormatter.h"
#include "CalibFormats/SiPixelObjects/interface/PixelDACNames.h"
//#include "PixelUtilities/PixelTestStandUtilities/include/PixelTimer.h"
#include <fstream>
#include <iostream>
#include <ios>
#include <cassert>
#include <cstdlib>
#include <algorithm>

using namespace pos;
using namespace std;

#define BPIX

PixelCalibConfiguration::PixelCalibConfiguration(std::vector<std::vector<std::string> >& tableMat)
    : PixelCalibBase(), PixelConfigBase("", "", "") {
  std::string mthn = "[PixelCalibConfiguration::PixelCalibConfiguration()]\t    ";
  std::map<std::string, int> colM;
  std::vector<std::string> colNames;
  /**

  EXTENSION_TABLE_NAME: PIXEL_CALIB_CLOB (VIEW: CONF_KEY_PIXEL_CALIB_V)
  
  CONFIG_KEY				    NOT NULL VARCHAR2(80)
  KEY_TYPE				    NOT NULL VARCHAR2(80)
  KEY_ALIAS				    NOT NULL VARCHAR2(80)
  VERSION					     VARCHAR2(40)
  KIND_OF_COND  			    NOT NULL VARCHAR2(40)
  CALIB_TYPE					     VARCHAR2(200)
  CALIB_OBJ_DATA_FILE			    NOT NULL VARCHAR2(200)
  CALIB_OBJ_DATA_CLOB			    NOT NULL CLOB
  */

  colNames.push_back("CONFIG_KEY");
  colNames.push_back("KEY_TYPE");
  colNames.push_back("KEY_ALIAS");
  colNames.push_back("VERSION");
  colNames.push_back("KIND_OF_COND");
  colNames.push_back("CALIB_TYPE");
  colNames.push_back("CALIB_OBJ_DATA_FILE");
  colNames.push_back("CALIB_OBJ_DATA_CLOB");

  for (unsigned int c = 0; c < tableMat[0].size(); c++) {
    for (unsigned int n = 0; n < colNames.size(); n++) {
      if (tableMat[0][c] == colNames[n]) {
        colM[colNames[n]] = c;
        break;
      }
    }
  }  //end for
  for (unsigned int n = 0; n < colNames.size(); n++) {
    if (colM.find(colNames[n]) == colM.end()) {
      std::cerr << mthn << "Couldn't find in the database the column with name " << colNames[n] << std::endl;
      assert(0);
    }
  }

  _bufferData = true;

  std::istringstream in;
  in.str(tableMat[1][colM["CALIB_OBJ_DATA_CLOB"]]);

  std::string tmp;

  in >> tmp;

  if (tmp == "Mode:") {
    in >> mode_;
    //    std::cout << __LINE__ << "]\t" << mthn << "mode=" << mode_ << std::endl;
    in >> tmp;
  } else {
    mode_ = "FEDChannelOffsetPixel";
    std::cout << __LINE__ << "]\t" << mthn << "mode not set, is this an old file? " << std::endl;
    assert(0);
  }

  singleROC_ = false;

  if (tmp == "SingleROC") {
    singleROC_ = true;
    in >> tmp;
  }

  // Read in parameters, if any.
  if (tmp == "Parameters:") {
    in >> tmp;
    while (tmp != "Rows:") {
      assert(!in.eof());
      std::string paramName = tmp;
      in >> tmp;  // tmp contains the parameter value
      parameters_[paramName] = tmp;
      in >> tmp;  // tmp contains the next parameter's name, or "Rows:"
    }
  }

  assert(tmp == "Rows:");

  in >> tmp;

  std::vector<unsigned int> rows;
  while (tmp != "Cols:") {
    if (tmp == "|") {
      rows_.push_back(rows);
      rows.clear();
    } else {
      if (tmp != "*") {
        rows.push_back(atoi(tmp.c_str()));
      }
    }
    in >> tmp;
  }
  rows_.push_back(rows);
  rows.clear();

  in >> tmp;

  std::vector<unsigned int> cols;
  while ((tmp != "VcalLow:") && (tmp != "VcalHigh:") && (tmp != "Vcal:") && (tmp != "VcalHigh") && (tmp != "VcalLow")) {
    if (tmp == "|") {
      cols_.push_back(cols);
      cols.clear();
    } else {
      if (tmp != "*") {
        cols.push_back(atoi(tmp.c_str()));
      }
    }
    in >> tmp;
  }
  cols_.push_back(cols);
  cols.clear();

  highVCalRange_ = true;

  if (tmp == "VcalLow") {
    highVCalRange_ = false;
    in >> tmp;
  }

  if (tmp == "VcalHigh") {
    highVCalRange_ = true;
    in >> tmp;
  }

  if (tmp == "VcalLow:") {
    highVCalRange_ = false;
  }

  if ((tmp == "VcalLow:") || (tmp == "VcalHigh:") || (tmp == "Vcal:")) {
    unsigned int first, last, step;
    in >> first >> last >> step;
    unsigned int index = 1;
    if (!dacs_.empty()) {
      index = dacs_.back().index() * dacs_.back().getNPoints();
    }
    in >> tmp;
    bool mix = false;
    if (tmp == "mix") {
      mix = true;
      in >> tmp;
    }
    PixelDACScanRange dacrange(pos::k_DACName_Vcal, first, last, step, index, mix);
    dacs_.push_back(dacrange);
  } else {
    //in >> tmp;
    while (tmp == "Scan:" || tmp == "ScanValues:") {
      if (tmp == "ScanValues:") {
        std::string dacname;
        in >> dacname;
        vector<unsigned int> values;
        int val;
        in >> val;
        while (val != -1) {
          values.push_back(val);
          in >> val;
        }
        unsigned int index = 1;
        if (!dacs_.empty()) {
          index = dacs_.back().index() * dacs_.back().getNPoints();
        }
        PixelDACScanRange dacrange(dacname, values, index, false);
        dacs_.push_back(dacrange);
        in >> tmp;
      } else {
        std::string dacname;
        in >> dacname;
        unsigned int first, last, step;
        in >> first >> last >> step;
        unsigned int index = 1;
        if (!dacs_.empty()) {
          index = dacs_.back().index() * dacs_.back().getNPoints();
        }
        in >> tmp;
        bool mix = false;
        if (tmp == "mix") {
          mix = true;
          in >> tmp;
        }
        PixelDACScanRange dacrange(dacname, first, last, step, index, mix);
        dacs_.push_back(dacrange);
      }
    }

    while ((tmp == "Set:") || (tmp == "SetRelative:")) {
      string name;
      in >> name;
      int val;
      in >> val;
      unsigned int index = 1;
      if (!dacs_.empty())
        index = dacs_.back().index() * dacs_.back().getNPoints();
      PixelDACScanRange dacrange(name, val, val, 1, index, false);
      if (tmp == "SetRelative:") {
        dacrange.setRelative();
      }
      dacs_.push_back(dacrange);
      in >> tmp;
    }
  }

  assert(tmp == "Repeat:");

  in >> ntrigger_;

  in >> tmp;

  usesROCList_ = false;
  bool buildROCListNow = false;
  if (tmp == "Rocs:") {
    buildROCListNow = true;
    usesROCList_ = true;
  } else {
    assert(tmp == "ToCalibrate:");
    buildROCListNow = false;
  }

  while (!in.eof()) {
    tmp = "";
    in >> tmp;

    // added by F.Blekman to be able to deal with POS245 style calib.dat files in CMSSW
    // these files use the syntax:
    // Rocs:
    // all

    if (tmp == "all" || tmp == "+" || tmp == "-") {
      buildROCListNow = false;
    }
    // end of addition by F.B.

    if (tmp.empty())
      continue;
    rocListInstructions_.push_back(tmp);
  }

  rocAndModuleListsBuilt_ = false;
  if (buildROCListNow) {
    std::set<PixelROCName> rocSet;
    for (std::vector<std::string>::iterator rocListInstructions_itr = rocListInstructions_.begin();
         rocListInstructions_itr != rocListInstructions_.end();
         ++rocListInstructions_itr) {
      PixelROCName rocname(*rocListInstructions_itr);
      rocSet.insert(rocname);
    }
    buildROCAndModuleListsFromROCSet(rocSet);
  }

  objectsDependingOnTheNameTranslationBuilt_ = false;

  // Added by Dario as a temporary patch for Debbie (this will disappear in the future)
  calibFileContent_ = in.str();
  // End of temporary patch

  return;
}

PixelCalibConfiguration::PixelCalibConfiguration(std::string filename) : PixelCalibBase(), PixelConfigBase("", "", "") {
  std::string mthn = "[PixelCalibConfiguration::PixelCalibConfiguration()]\t    ";

  _bufferData = true;

  std::ifstream in(filename.c_str());

  if (!in.good()) {
    std::cout << __LINE__ << "]\t" << mthn << "Could not open:" << filename << std::endl;
    assert(0);
  } else {
    std::cout << __LINE__ << "]\t" << mthn << "Opened:" << filename << std::endl;
  }

  std::string tmp;

  in >> tmp;

  if (tmp == "Mode:") {
    in >> mode_;
    std::cout << __LINE__ << "]\t" << mthn << "PixelCalibConfiguration mode=" << mode_ << std::endl;
    in >> tmp;
  } else {
    mode_ = "FEDChannelOffsetPixel";
    std::cout << __LINE__ << "]\t" << mthn << "PixelCalibConfiguration mode not set, is this an old file? " << __LINE__
              << "]\t" << std::endl;
    assert(0);
  }

  singleROC_ = false;

  if (tmp == "SingleROC") {
    singleROC_ = true;
    in >> tmp;
  }

  // Read in parameters, if any.
  if (tmp == "Parameters:") {
    in >> tmp;
    while (tmp != "Rows:") {
      assert(!in.eof());
      std::string paramName = tmp;
      in >> tmp;  // tmp contains the parameter value
      parameters_[paramName] = tmp;
      in >> tmp;  // tmp contains the next parameter's name, or "Rows:"
    }
  }

  assert(tmp == "Rows:");

  in >> tmp;

  std::vector<unsigned int> rows;
  while (tmp != "Cols:") {
    if (tmp == "|") {
      rows_.push_back(rows);
      rows.clear();
    } else {
      if (tmp != "*") {
        rows.push_back(atoi(tmp.c_str()));
      }
    }
    in >> tmp;
  }
  rows_.push_back(rows);
  rows.clear();

  in >> tmp;

  std::vector<unsigned int> cols;
  while ((tmp != "VcalLow:") && (tmp != "VcalHigh:") && (tmp != "Vcal:") && (tmp != "VcalHigh") && (tmp != "VcalLow")) {
    if (tmp == "|") {
      cols_.push_back(cols);
      cols.clear();
    } else {
      if (tmp != "*") {
        cols.push_back(atoi(tmp.c_str()));
      }
    }
    in >> tmp;
  }
  cols_.push_back(cols);
  cols.clear();

  highVCalRange_ = true;

  if (tmp == "VcalLow") {
    highVCalRange_ = false;
    in >> tmp;
  }

  if (tmp == "VcalHigh") {
    highVCalRange_ = true;
    in >> tmp;
  }

  if (tmp == "VcalLow:") {
    highVCalRange_ = false;
  }

  if ((tmp == "VcalLow:") || (tmp == "VcalHigh:") || (tmp == "Vcal:")) {
    unsigned int first, last, step;
    in >> first >> last >> step;
    unsigned int index = 1;
    if (!dacs_.empty()) {
      index = dacs_.back().index() * dacs_.back().getNPoints();
    }
    in >> tmp;
    bool mix = false;
    if (tmp == "mix") {
      mix = true;
      in >> tmp;
    }
    PixelDACScanRange dacrange(pos::k_DACName_Vcal, first, last, step, index, mix);
    dacs_.push_back(dacrange);
  } else {
    //in >> tmp;
    while (tmp == "Scan:" || tmp == "ScanValues:") {
      if (tmp == "ScanValues:") {
        std::string dacname;
        in >> dacname;
        vector<unsigned int> values;
        int val;
        in >> val;
        while (val != -1) {
          values.push_back(val);
          in >> val;
        }
        unsigned int index = 1;
        if (!dacs_.empty()) {
          index = dacs_.back().index() * dacs_.back().getNPoints();
        }
        PixelDACScanRange dacrange(dacname, values, index, false);
        dacs_.push_back(dacrange);
        in >> tmp;
      } else {
        std::string dacname;
        in >> dacname;
        unsigned int first, last, step;
        in >> first >> last >> step;
        unsigned int index = 1;
        if (!dacs_.empty()) {
          index = dacs_.back().index() * dacs_.back().getNPoints();
        }
        in >> tmp;
        bool mix = false;
        if (tmp == "mix") {
          mix = true;
          in >> tmp;
        }
        PixelDACScanRange dacrange(dacname, first, last, step, index, mix);
        dacs_.push_back(dacrange);
      }
    }

    while ((tmp == "Set:") || (tmp == "SetRelative:")) {
      string name;
      in >> name;
      int val;
      in >> val;
      unsigned int absval = std::abs(val);
      unsigned int index = 1;
      if (!dacs_.empty())
        index = dacs_.back().index() * dacs_.back().getNPoints();
      PixelDACScanRange dacrange(name, absval, absval, 1, index, false);
      if (tmp == "SetRelative:") {
        dacrange.setRelative();
        if (val < 0) {
          dacrange.setNegative();
        }
      }
      dacs_.push_back(dacrange);
      in >> tmp;
    }
  }

  assert(tmp == "Repeat:");

  in >> ntrigger_;

  in >> tmp;

  usesROCList_ = false;
  bool buildROCListNow = false;
  if (tmp == "Rocs:") {
    buildROCListNow = true;
    usesROCList_ = true;
  } else {
    assert(tmp == "ToCalibrate:");
    buildROCListNow = false;
  }

  while (!in.eof()) {
    tmp = "";
    in >> tmp;

    // added by F.Blekman to be able to deal with POS245 style calib.dat files in CMSSW
    // these files use the syntax:
    // Rocs:
    // all

    if (tmp == "all" || tmp == "+" || tmp == "-") {
      buildROCListNow = false;
    }
    // end of addition by F.B.

    if (tmp.empty())
      continue;
    rocListInstructions_.push_back(tmp);
  }

  in.close();

  rocAndModuleListsBuilt_ = false;
  if (buildROCListNow) {
    std::set<PixelROCName> rocSet;
    for (std::vector<std::string>::iterator rocListInstructions_itr = rocListInstructions_.begin();
         rocListInstructions_itr != rocListInstructions_.end();
         ++rocListInstructions_itr) {
      PixelROCName rocname(*rocListInstructions_itr);
      rocSet.insert(rocname);
    }
    buildROCAndModuleListsFromROCSet(rocSet);
  }

  objectsDependingOnTheNameTranslationBuilt_ = false;

  // Added by Dario as a temporary patch for Debbie (this will disappear in the future)
  std::ifstream inTmp(filename.c_str());
  calibFileContent_ = "";
  while (!inTmp.eof()) {
    std::string tmpString;
    getline(inTmp, tmpString);
    calibFileContent_ += tmpString + "\n";
    //cout << __LINE__ << "]\t" << "[PixelCalibConfiguration::~PixelCalibConfiguration()]\t\t" << calibFileContent_ << endl ;
  }
  inTmp.close();
  // End of temporary patch

  return;
}

PixelCalibConfiguration::~PixelCalibConfiguration() {}

void PixelCalibConfiguration::buildROCAndModuleLists(const PixelNameTranslation* translation,
                                                     const PixelDetectorConfig* detconfig) {
  assert(translation != nullptr);
  assert(detconfig != nullptr);

  if (rocAndModuleListsBuilt_) {
    buildObjectsDependingOnTheNameTranslation(translation);
    return;
  }

  // Build the ROC set from the instructions.
  std::set<PixelROCName> rocSet;
  bool addNext = true;
  const map<PixelROCName, PixelROCStatus>& iroclist = detconfig->getROCsList();
  for (std::vector<std::string>::iterator rocListInstructions_itr = rocListInstructions_.begin();
       rocListInstructions_itr != rocListInstructions_.end();
       ++rocListInstructions_itr) {
    std::string instruction = *rocListInstructions_itr;

    if (instruction == "+") {
      addNext = true;
      continue;
    }
    if (instruction == "-") {
      addNext = false;
      continue;
    }

    if (instruction == "all") {
      if (addNext)  // add all ROCs in the configuration
      {
        const std::vector<PixelModuleName>& moduleList = detconfig->getModuleList();
        for (std::vector<PixelModuleName>::const_iterator moduleList_itr = moduleList.begin();
             moduleList_itr != moduleList.end();
             ++moduleList_itr) {
          std::vector<PixelROCName> ROCsOnThisModule = translation->getROCsFromModule(*moduleList_itr);
          for (std::vector<PixelROCName>::const_iterator ROCsOnThisModule_itr = ROCsOnThisModule.begin();
               ROCsOnThisModule_itr != ROCsOnThisModule.end();
               ++ROCsOnThisModule_itr) {
            map<PixelROCName, PixelROCStatus>::const_iterator it = iroclist.find(*ROCsOnThisModule_itr);
            assert(it != iroclist.end());
            PixelROCStatus istatus = it->second;
            if (!istatus.get(PixelROCStatus::noAnalogSignal))
              rocSet.insert(*ROCsOnThisModule_itr);
          }
        }
      } else  // remove all ROCs
      {
        rocSet.clear();
      }
      addNext = true;
      continue;
    }

    // Assume it's a ROC or module name.
    PixelModuleName modulename(instruction);

    // Skip if this module (or the module this ROC is on) isn't in the detector config.
    if (!(detconfig->containsModule(modulename))) {
      addNext = true;
      continue;
    }

    if (modulename.modulename() == instruction)  // it's a module
    {
      std::vector<PixelROCName> ROCsOnThisModule = translation->getROCsFromModule(modulename);
      for (std::vector<PixelROCName>::iterator ROCsOnThisModule_itr = ROCsOnThisModule.begin();
           ROCsOnThisModule_itr != ROCsOnThisModule.end();
           ++ROCsOnThisModule_itr) {
        if (addNext) {
          map<PixelROCName, PixelROCStatus>::const_iterator it = iroclist.find(*ROCsOnThisModule_itr);
          assert(it != iroclist.end());
          PixelROCStatus istatus = it->second;
          if (!istatus.get(PixelROCStatus::noAnalogSignal))
            rocSet.insert(*ROCsOnThisModule_itr);
        } else
          rocSet.erase(*ROCsOnThisModule_itr);
      }
      addNext = true;
      continue;
    } else  // it's a ROC
    {
      PixelROCName rocname(instruction);
      if (addNext) {
        // Only add this ROC if it's in the configuration.
        bool foundIt = false;
        std::list<const PixelROCName*> allROCs = translation->getROCs();
        for (std::list<const PixelROCName*>::iterator allROCs_itr = allROCs.begin(); allROCs_itr != allROCs.end();
             ++allROCs_itr) {
          if ((*(*allROCs_itr)) == rocname) {
            foundIt = true;
            break;
          }
        }
        if (foundIt) {
          map<PixelROCName, PixelROCStatus>::const_iterator it = iroclist.find(rocname);
          assert(it != iroclist.end());
          PixelROCStatus istatus = it->second;
          if (!istatus.get(PixelROCStatus::noAnalogSignal))
            rocSet.insert(rocname);
        }
      } else {
        rocSet.erase(rocname);
      }
      addNext = true;
      continue;
    }

    // should never get here
    assert(0);
  }
  // done building ROC set

  buildROCAndModuleListsFromROCSet(rocSet);

  buildObjectsDependingOnTheNameTranslation(translation);
}

void PixelCalibConfiguration::buildROCAndModuleListsFromROCSet(const std::set<PixelROCName>& rocSet) {
  assert(!rocAndModuleListsBuilt_);

  std::string mthn = "[PixelCalibConfiguration::buildROCAndModuleListsFromROCSet()]    ";
  // Build the ROC list from the ROC set.
  for (std::set<PixelROCName>::iterator rocSet_itr = rocSet.begin(); rocSet_itr != rocSet.end(); ++rocSet_itr) {
    rocs_.push_back(*rocSet_itr);
  }

  //t.stop();
  //cout << "buildROCAndModuleListsFromROCSet 1 time="<<t.tottime()<<endl;
  //t.start();

  // Build the module set from the ROC set.
  std::map<PixelModuleName, unsigned int> countROC;
  for (std::set<PixelROCName>::iterator rocSet_itr = rocSet.begin(); rocSet_itr != rocSet.end(); ++rocSet_itr) {
    //t1.start();
    PixelModuleName modulename(*rocSet_itr);
    //t1.stop();
    //t2.start();
    modules_.insert(modulename);
    countROC[modulename]++;
    //t2.stop();
  }

  //t.stop();
  //cout << "buildROCAndModuleListsFromROCSet 2 time="<<t.tottime()<<endl;
  //cout << "buildROCAndModuleListsFromROCSet t1="<<t1.tottime()<<endl;
  //cout << "buildROCAndModuleListsFromROCSet t2="<<t2.tottime()<<endl;
  //t.start();

  // Test printout.
  /*cout << "\nROC list:\n";
    for ( std::vector<PixelROCName>::iterator rocs_itr = rocs_.begin(); 
    rocs_itr != rocs_.end(); 
    rocs_itr++ ){
    cout << rocs_itr->rocname() << "\n";
    }    cout << "\nModule list:\n";
    for ( std::set<PixelModuleName>::iterator modules_itr = modules_.begin(); modules_itr != modules_.end(); modules_itr++ )
	{
	cout << modules_itr->modulename() << "\n";
	}
	cout << "\n";*/

  // Determine max ROCs on a module for singleROC mode.
  nROC_ = 1;
  if (singleROC_) {
    unsigned maxROCs = 0;
    for (std::map<PixelModuleName, unsigned int>::iterator imodule = countROC.begin(); imodule != countROC.end();
         ++imodule) {
      if (imodule->second > maxROCs)
        maxROCs = imodule->second;
    }
    nROC_ = maxROCs;

    std::cout << __LINE__ << "]\t" << mthn << "Max ROCs on a module=" << nROC_ << std::endl;
  }

  for (unsigned int irocs = 0; irocs < rocs_.size(); irocs++) {
    old_irows.push_back(-1);
    old_icols.push_back(-1);
  }

  rocAndModuleListsBuilt_ = true;
}

void PixelCalibConfiguration::buildObjectsDependingOnTheNameTranslation(const PixelNameTranslation* aNameTranslation) {
  assert(!objectsDependingOnTheNameTranslationBuilt_);
  assert(rocAndModuleListsBuilt_);
  assert(aNameTranslation != nullptr);

  // Build the channel list.
  assert(channels_.empty());
  for (std::vector<PixelROCName>::const_iterator rocs_itr = rocs_.begin(); rocs_itr != rocs_.end(); ++rocs_itr) {
    channels_.insert(aNameTranslation->getChannelForROC(*rocs_itr));
  }

  // Build the maps from ROC to ROC number.

  assert(ROCNumberOnChannelAmongThoseCalibrated_.empty() && numROCsCalibratedOnChannel_.empty());

  std::set<PixelROCName> tempROCs;

  for (std::vector<PixelROCName>::const_iterator it = rocs_.begin(); it != rocs_.end(); ++it) {
    tempROCs.insert(*it);
  }

  for (std::set<PixelChannel>::const_iterator channels_itr = channels_.begin(); channels_itr != channels_.end();
       ++channels_itr) {
    std::vector<PixelROCName> rocsOnChannel = aNameTranslation->getROCsFromChannel(*channels_itr);

    std::set<PixelROCName> foundROCs;

    for (std::vector<PixelROCName>::const_iterator rocsOnChannel_itr = rocsOnChannel.begin();
         rocsOnChannel_itr != rocsOnChannel.end();
         ++rocsOnChannel_itr) {
      if (tempROCs.find(*rocsOnChannel_itr) != tempROCs.end()) {
        ROCNumberOnChannelAmongThoseCalibrated_[*rocsOnChannel_itr] = foundROCs.size();
        foundROCs.insert(*rocsOnChannel_itr);
      }
    }

    for (std::set<PixelROCName>::const_iterator foundROCs_itr = foundROCs.begin(); foundROCs_itr != foundROCs.end();
         ++foundROCs_itr) {
      numROCsCalibratedOnChannel_[*foundROCs_itr] = foundROCs.size();
    }
  }

  objectsDependingOnTheNameTranslationBuilt_ = true;
}

unsigned int PixelCalibConfiguration::iScan(std::string dac) const {
  for (unsigned int i = 0; i < dacs_.size(); i++) {
    if (dac == dacs_[i].name())
      return i;
  }

  std::cout << __LINE__ << "]\t[PixelCalibConfiguration::iScan()]\t\t    could not find dac=" << dac << std::endl;

  assert(0);

  return 0;
}

unsigned int PixelCalibConfiguration::scanROC(unsigned int state) const {
  assert(state < nConfigurations());

  unsigned int i_ROC = state / (cols_.size() * rows_.size() * nScanPoints());

  return i_ROC;
}

unsigned int PixelCalibConfiguration::scanValue(unsigned int iscan,
                                                unsigned int state,
                                                unsigned int ROCNumber,
                                                unsigned int ROCsOnChannel) const {
  unsigned int i_threshold = scanCounter(iscan, state);

  // Spread the DAC values on the different ROCs uniformly across the scan range.
  if (dacs_[iscan].mixValuesAcrossROCs())
    i_threshold = (i_threshold + (nScanPoints(iscan) * ROCNumber) / ROCsOnChannel) % nScanPoints(iscan);

  unsigned int threshold = dacs_[iscan].value(i_threshold);

  //assert(threshold==dacs_[iscan].first()+i_threshold*dacs_[iscan].step());

  return threshold;
}

bool PixelCalibConfiguration::scanningROCForState(PixelROCName roc, unsigned int state) const {
  if (!singleROC_)
    return true;
  return scanROC(state) == ROCNumberOnChannelAmongThoseCalibrated(roc);
}

unsigned int PixelCalibConfiguration::scanValue(unsigned int iscan, unsigned int state, PixelROCName roc) const {
  unsigned int ROCNumber = ROCNumberOnChannelAmongThoseCalibrated(roc);
  unsigned int ROCsOnChannel = numROCsCalibratedOnChannel(roc);

  return scanValue(iscan, state, ROCNumber, ROCsOnChannel);
}

unsigned int PixelCalibConfiguration::ROCNumberOnChannelAmongThoseCalibrated(PixelROCName roc) const {
  assert(objectsDependingOnTheNameTranslationBuilt_);
  std::map<PixelROCName, unsigned int>::const_iterator foundROC = ROCNumberOnChannelAmongThoseCalibrated_.find(roc);
  assert(foundROC != ROCNumberOnChannelAmongThoseCalibrated_.end());
  return foundROC->second;
}

unsigned int PixelCalibConfiguration::numROCsCalibratedOnChannel(PixelROCName roc) const {
  assert(objectsDependingOnTheNameTranslationBuilt_);
  std::map<PixelROCName, unsigned int>::const_iterator foundROC = numROCsCalibratedOnChannel_.find(roc);
  assert(foundROC != numROCsCalibratedOnChannel_.end());
  return foundROC->second;
}

unsigned int PixelCalibConfiguration::scanCounter(unsigned int iscan, unsigned int state) const {
  assert(state < nConfigurations());

  unsigned int i_scan = state % nScanPoints();

  for (unsigned int i = 0; i < iscan; i++) {
    i_scan /= nScanPoints(i);
  }

  unsigned int i_threshold = i_scan % nScanPoints(iscan);

  return i_threshold;
}

unsigned int PixelCalibConfiguration::rowCounter(unsigned int state) const {
  unsigned int i_row =
      (state - scanROC(state) * cols_.size() * rows_.size() * nScanPoints()) / (cols_.size() * nScanPoints());
  assert(i_row < rows_.size());
  return i_row;
}

unsigned int PixelCalibConfiguration::colCounter(unsigned int state) const {
  unsigned int i_col = (state - scanROC(state) * cols_.size() * rows_.size() * nScanPoints() -
                        rowCounter(state) * cols_.size() * nScanPoints()) /
                       (nScanPoints());
  assert(i_col < cols_.size());
  return i_col;
}

void PixelCalibConfiguration::nextFECState(std::map<unsigned int, PixelFECConfigInterface*>& pixelFECs,
                                           PixelDetectorConfig* detconfig,
                                           PixelNameTranslation* trans,
                                           std::map<pos::PixelModuleName, pos::PixelMaskBase*>* masks,
                                           std::map<pos::PixelModuleName, pos::PixelTrimBase*>* trims,
                                           std::map<pos::PixelModuleName, pos::PixelDACSettings*>* dacs,

                                           unsigned int state) const {
  std::string mthn = "[PixelCalibConfiguration::nextFECState()]\t\t    ";
  std::string modeName = parameterValue("ScanMode");

  int mode = -1;

  if (modeName == "maskAllPixel")
    mode = 0;
  if (modeName == "useAllPixel" || modeName.empty())
    mode = 1;
  if (modeName == "default")
    mode = 2;

  static bool first = true;

  if (first) {
    cout << __LINE__ << "]\t" << mthn << "mode=" << mode << endl;
    first = false;
  }

  if (mode == -1) {
    cout << __LINE__ << "]\t" << mthn << "ScanMode=" << modeName << " not understood." << endl;
    ::abort();
  }

  if (rocInfo_.empty()) {
    //here we will do some initialization...
    for (unsigned int i = 0; i < rocs_.size(); i++) {
      const PixelHdwAddress* hdwadd = trans->getHdwAddress(rocs_[i]);
      PixelROCInfo rocInfo;
      rocInfo.use_ = true;
      //FIXME This is very inefficient
      PixelModuleName module(rocs_[i].rocname());

      std::map<pos::PixelModuleName, pos::PixelMaskBase*>::const_iterator foundMask = masks->find(module);
      if (foundMask == masks->end()) {
        rocInfo.use_ = false;
        rocInfo_.push_back(rocInfo);
        continue;
      }

      rocInfo.hdwadd_ = hdwadd;
      rocInfo.trims_ = (*trims)[module]->getTrimBits(rocs_[i]);
      rocInfo.masks_ = (*masks)[module]->getMaskBits(rocs_[i]);

#ifdef BPIX
      const PixelChannel channel = trans->getChannelForROC(rocs_[i]);
      string tbmChannel = channel.TBMChannelString();
      //cout<<" tbm channel "<<tbmChannel<<endl;
      rocInfo.tbmChannel_ = tbmChannel;
#endif

      std::map<std::string, unsigned int> defaultDACValues;
      (*dacs)[PixelModuleName(rocs_[i].rocname())]->getDACSettings(rocs_[i])->getDACs(defaultDACValues);

      for (std::vector<PixelDACScanRange>::const_iterator dacs_itr = dacs_.begin(); dacs_itr != dacs_.end();
           ++dacs_itr) {
        std::map<std::string, unsigned int>::const_iterator foundThisDAC = defaultDACValues.find(dacs_itr->name());
        assert(foundThisDAC != defaultDACValues.end());

        pair<unsigned int, unsigned int> dacchannelAndValue(dacs_itr->dacchannel(), foundThisDAC->second);

        rocInfo.defaultDACs_.push_back(dacchannelAndValue);
      }
      rocInfo_.push_back(rocInfo);
    }
  }

  assert(rocs_.size() == rocInfo_.size());

  bool changedWBC = false;

  //std::map<unsigned int, PixelFECConfigInterface*>::iterator iPixelFEC=pixelFECs.begin();
  //for(;iPixelFEC!=pixelFECs.end();++iPixelFEC){
  //iPixelFEC->second->fecDebug(1);
  //}

  //unsigned long version=0;
  //pixelFEC->getversion(&version);
  //std::cout<<"mfec firmware version:"<<version<<std::endl;

  assert(rocAndModuleListsBuilt_);

  assert(state < nConfigurations());

  // Which set of rows we're on.
  unsigned int i_row = rowCounter(state);

  // Which set of columns we're on.
  unsigned int i_col = colCounter(state);

  // Whether we're beginning a new scan over the DACs after changing which ROC or which pixel pattern.
  unsigned int first_scan = true;
  for (unsigned int i = 0; i < dacs_.size(); i++) {
    if (scanCounter(i, state) != 0)
      first_scan = false;
  }

  // Disable all pixels at the beginning.
  if (state == 0 && (mode == 0 || mode == 1)) {
    for (unsigned int i = 0; i < rocs_.size(); i++) {
      if (!rocInfo_[i].use_)
        continue;

      PixelHdwAddress theROC = *rocInfo_[i].hdwadd_;
      PixelROCTrimBits* rocTrims = rocInfo_[i].trims_;

      //Turn off all pixels
      disablePixels(pixelFECs[theROC.fecnumber()], rocTrims, theROC);
    }

    // reset
    //cout<<"finish init"<<endl;
    //sleep(1);
    //pixelFEC->injectrstroc(7,1);
    // or do    pixelFEC->rocreset(theROC.mfec(),
    // 		       theROC.mfecchannel(),
    // 		       14,                    //FIXME hardcode for Channel A
    // 		       theROC.hubaddress());
  }

  // When a scan is complete for a given ROC or pixel pattern, reset the DACs to default values and disable the previously-enabled pixels.
  if (first_scan && state != 0 && mode != 2) {
    unsigned int previousState = state - 1;

    unsigned int i_row_previous = rowCounter(previousState);

    unsigned int i_col_previous = colCounter(previousState);

    for (unsigned int i = 0; i < rocs_.size(); i++) {
      if (!rocInfo_[i].use_)
        continue;

      PixelHdwAddress theROC = *rocInfo_[i].hdwadd_;

      if (!scanningROCForState(rocs_[i], previousState))
        continue;

      // Set the DACs back to their default values when we're done with a scan.
      for (unsigned int j = 0; j < dacs_.size(); j++) {
        //Try to not reprogram DACs as often..
        if (state != 0) {
          if (scanCounter(dacs_[j].name(), state) == scanCounter(dacs_[j].name(), state - 1)) {
            continue;
          }
        }

        pixelFECs[theROC.fecnumber()]->progdac(theROC.mfec(),
                                               theROC.mfecchannel(),
                                               theROC.hubaddress(),
                                               theROC.portaddress(),
                                               theROC.rocid(),
                                               rocInfo_[i].defaultDACs_[j].first,
                                               rocInfo_[i].defaultDACs_[j].second,
                                               _bufferData);

        if (dacs_[j].dacchannel() == k_DACAddress_WBC) {
          changedWBC = true;
          //cout << "Changed WBC 1"<<endl;
        }
      }

      PixelROCTrimBits* rocTrims = rocInfo_[i].trims_;

      disablePixels(pixelFECs[theROC.fecnumber()], i_row_previous, i_col_previous, rocTrims, theROC);
    }
  }

  // Set each ROC with the new settings for this state.
  for (unsigned int i = 0; i < rocs_.size(); i++) {
    if (!rocInfo_[i].use_)
      continue;

    PixelHdwAddress theROC = *rocInfo_[i].hdwadd_;

    // Skip this ROC if we're in SingleROC mode and we're not on this ROC number.
    if (!scanningROCForState(rocs_[i], state))
      continue;

    //	std::cout << "Will call progdac for vcal:"<< vcal << std::endl;

    // Program all the DAC values.
    for (unsigned int ii = 0; ii < dacs_.size(); ii++) {
      //Try to not reprogram DACs as often..
      if (state != 0) {
        if (scanCounter(dacs_[ii].name(), state) == scanCounter(dacs_[ii].name(), state - 1)) {
          continue;
        }
      }

      int dacvalue = scanValue(ii, state, rocs_[i]);

      //cout << "dacname ii:"<<dacs_[ii].name()<<" "<<ii<<endl;

      if (dacs_[ii].relative()) {
        //We have to find the default DAC setting so that we can
        //add the offset as we are in relative mode.

        if (dacs_[ii].negative())
          dacvalue = -dacvalue;

        dacvalue += rocInfo_[i].defaultDACs_[ii].second;
        //cout << "[PixelCalibConfiguration::nextFECState] ROC="<<rocs_[i]
        //     << " dac="<<dacs_[ii].name()<<" new value="<<dacvalue<<endl;
      }

      pixelFECs[theROC.fecnumber()]->progdac(theROC.mfec(),
                                             theROC.mfecchannel(),
                                             theROC.hubaddress(),
                                             theROC.portaddress(),
                                             theROC.rocid(),
                                             rocInfo_[i].defaultDACs_[ii].first,
                                             dacvalue,
                                             _bufferData);

      if (dacs_[ii].dacchannel() == k_DACAddress_WBC) {
        changedWBC = true;
        //cout << "Changed WBC 2"<<endl;
      }
    }

    // At the beginning of a scan, set the pixel pattern.
    if (first_scan) {
      // Set masks and trims.
      if (mode != 2) {
        PixelROCMaskBits* rocMasks = rocInfo_[i].masks_;
        PixelROCTrimBits* rocTrims = rocInfo_[i].trims_;

        if (mode == 1)
          rocMasks = nullptr;

        //std::cout << "Will enable pixels!" <<std::endl;
        enablePixels(pixelFECs[theROC.fecnumber()], i_row, i_col, rocMasks, rocTrims, theROC);
      }

      // Set high or low Vcal range.

      if (state == 0) {
        PixelModuleName module(rocs_[i].rocname());

        unsigned int roccontrolword = (*dacs)[module]->getDACSettings(rocs_[i])->getControlRegister();

        //range is controlled here by one bit, but rest must match config
        //bit 0 on/off= 20/40 MHz speed; bit 1 on/off=disabled/enable; bit 3=Vcal range

        if (highVCalRange_)
          roccontrolword |= 0x4;  //turn range bit on
        else
          roccontrolword &= 0xfb;  //turn range bit off

        pixelFECs[theROC.fecnumber()]->progdac(theROC.mfec(),
                                               theROC.mfecchannel(),
                                               theROC.hubaddress(),
                                               theROC.portaddress(),
                                               theROC.rocid(),
                                               0xfd,
                                               roccontrolword,
                                               _bufferData);
      }

      // Clear all pixels before setting the pixel pattern.
      pixelFECs[theROC.fecnumber()]->clrcal(
          theROC.mfec(), theROC.mfecchannel(), theROC.hubaddress(), theROC.portaddress(), theROC.rocid(), _bufferData);

      // Program the pixel pattern.
      unsigned int nrow = rows_[i_row].size();
      unsigned int ncol = cols_[i_col].size();
      unsigned int nmax = std::max(nrow, ncol);
      if (nrow == 0 || ncol == 0)
        nmax = 0;
      for (unsigned int n = 0; n < nmax; n++) {
        unsigned int irow = n;
        unsigned int icol = n;
        if (irow >= nrow)
          irow = nrow - 1;
        if (icol >= ncol)
          icol = ncol - 1;
        unsigned int row = rows_[i_row][irow];
        unsigned int col = cols_[i_col][icol];

        pixelFECs[theROC.fecnumber()]->calpix(theROC.mfec(),
                                              theROC.mfecchannel(),
                                              theROC.hubaddress(),
                                              theROC.portaddress(),
                                              theROC.rocid(),
                                              col,
                                              row,
                                              1,
                                              _bufferData);
      }

    }  // end of instructions for the beginning of a scan
  }    // end of loop over ROCs

  if (_bufferData) {
    std::map<unsigned int, PixelFECConfigInterface*>::iterator iPixelFEC = pixelFECs.begin();
    for (; iPixelFEC != pixelFECs.end(); ++iPixelFEC) {
      iPixelFEC->second->qbufsend();
    }
  }

  if (changedWBC) {
    for (unsigned int i = 0; i < rocs_.size(); i++) {
      if (!rocInfo_[i].use_)
        continue;

      PixelHdwAddress theROC = *rocInfo_[i].hdwadd_;

      int tbmRegister = 14;  // define for TBM-A
#ifdef BPIX
      string tbmChannel = rocInfo_[i].tbmChannel_;  // get TBM channel
      if (tbmChannel == "B")
        tbmRegister = 15;  // for TBM=B
#endif

      pixelFECs[theROC.fecnumber()]->rocreset(theROC.mfec(), theROC.mfecchannel(), tbmRegister, theROC.hubaddress());
    }  // for rocs
  }

  return;
}

// FIXME This code breaks if it is called more than once with different crate numbers!
std::vector<std::pair<unsigned int, std::vector<unsigned int> > >& PixelCalibConfiguration::fedCardsAndChannels(
    unsigned int crate,
    PixelNameTranslation* translation,
    PixelFEDConfig* fedconfig,
    PixelDetectorConfig* detconfig) const {
  assert(rocAndModuleListsBuilt_);

  assert(!rocs_.empty());

  for (unsigned int i = 0; i < rocs_.size(); i++) {
    PixelModuleName module(rocs_[i].rocname());
    if (!detconfig->containsModule(module))
      continue;
    const PixelHdwAddress* hdw = translation->getHdwAddress(rocs_[i]);
    assert(hdw != nullptr);
    //std::cout << "ROC, fednumber:"<<rocs_[i]<<" "<<hdw->fednumber()
    //	  << std::endl;
    //first check if fed associated with the roc is in the right crate
    if (fedconfig->crateFromFEDNumber(hdw->fednumber()) != crate)
      continue;
    //next look if we have already found fed number
    unsigned int index = fedCardsAndChannels_.size();
    for (unsigned int j = 0; j < fedCardsAndChannels_.size(); j++) {
      if (fedCardsAndChannels_[j].first == hdw->fednumber()) {
        index = j;
        break;
      }
    }
    //If we didn't find the fedcard we will create it
    if (index == fedCardsAndChannels_.size()) {
      std::vector<unsigned int> tmp;
      tmp.push_back(hdw->fedchannel());
      std::pair<unsigned int, std::vector<unsigned int> > tmp2(hdw->fednumber(), tmp);
      fedCardsAndChannels_.push_back(tmp2);
      continue;
    }
    //Now look and see if the channel has been added
    std::vector<unsigned int>& channels = fedCardsAndChannels_[index].second;
    bool found = false;
    for (unsigned int k = 0; k < channels.size(); k++) {
      if (channels[k] == hdw->fedchannel()) {
        found = true;
        break;
      }
    }
    if (found)
      continue;
    channels.push_back(hdw->fedchannel());
  }

  return fedCardsAndChannels_;
}

std::map<unsigned int, std::set<unsigned int> > PixelCalibConfiguration::getFEDsAndChannels(
    PixelNameTranslation* translation) {
  assert(rocAndModuleListsBuilt_);

  std::map<unsigned int, std::set<unsigned int> > fedsChannels;
  assert(!rocs_.empty());
  std::vector<PixelROCName>::iterator iroc = rocs_.begin();

  for (; iroc != rocs_.end(); ++iroc) {
    const PixelHdwAddress* roc_hdwaddress = translation->getHdwAddress(*iroc);
    unsigned int fednumber = roc_hdwaddress->fednumber();
    unsigned int fedchannel = roc_hdwaddress->fedchannel();
    fedsChannels[fednumber].insert(fedchannel);
  }

  return fedsChannels;
}

std::set<unsigned int> PixelCalibConfiguration::getFEDCrates(const PixelNameTranslation* translation,
                                                             const PixelFEDConfig* fedconfig) const {
  assert(rocAndModuleListsBuilt_);

  std::set<unsigned int> fedcrates;
  assert(!modules_.empty());
  std::set<PixelModuleName>::iterator imodule = modules_.begin();

  for (; imodule != modules_.end(); ++imodule) {
    std::set<PixelChannel> channelsOnThisModule = translation->getChannelsOnModule(*imodule);
    for (std::set<PixelChannel>::const_iterator channelsOnThisModule_itr = channelsOnThisModule.begin();
         channelsOnThisModule_itr != channelsOnThisModule.end();
         ++channelsOnThisModule_itr) {
      const PixelHdwAddress& channel_hdwaddress = translation->getHdwAddress(*channelsOnThisModule_itr);
      unsigned int fednumber = channel_hdwaddress.fednumber();
      fedcrates.insert(fedconfig->crateFromFEDNumber(fednumber));
    }
  }

  return fedcrates;
}

std::set<unsigned int> PixelCalibConfiguration::getFECCrates(const PixelNameTranslation* translation,
                                                             const PixelFECConfig* fecconfig) const {
  assert(rocAndModuleListsBuilt_);

  std::set<unsigned int> feccrates;
  assert(!modules_.empty());
  std::set<PixelModuleName>::iterator imodule = modules_.begin();

  for (; imodule != modules_.end(); ++imodule) {
    std::set<PixelChannel> channelsOnThisModule = translation->getChannelsOnModule(*imodule);
    for (std::set<PixelChannel>::const_iterator channelsOnThisModule_itr = channelsOnThisModule.begin();
         channelsOnThisModule_itr != channelsOnThisModule.end();
         ++channelsOnThisModule_itr) {
      const PixelHdwAddress& channel_hdwaddress = translation->getHdwAddress(*channelsOnThisModule_itr);
      unsigned int fecnumber = channel_hdwaddress.fecnumber();
      feccrates.insert(fecconfig->crateFromFECNumber(fecnumber));
    }
  }

  return feccrates;
}

std::set<unsigned int> PixelCalibConfiguration::getTKFECCrates(
    const PixelPortcardMap* portcardmap,
    const std::map<std::string, PixelPortCardConfig*>& mapNamePortCard,
    const PixelTKFECConfig* tkfecconfig) const {
  assert(rocAndModuleListsBuilt_);

  std::set<unsigned int> tkfeccrates;
  assert(!modules_.empty());
  std::set<PixelModuleName>::iterator imodule = modules_.begin();

  for (; imodule != modules_.end(); ++imodule) {
    // implement this by module --(PixelPortcardMap)-> port card(s) --(PixelPortCardConfig)-> FEC # --(PixelFECConfig theTKFECConfiguration_)-> crate
    const std::set<std::string> portCards = portcardmap->portcards(*imodule);
    for (std::set<std::string>::const_iterator portCards_itr = portCards.begin(); portCards_itr != portCards.end();
         ++portCards_itr) {
      const std::string& portcardname = *portCards_itr;
      std::map<std::string, PixelPortCardConfig*>::const_iterator portcardconfig_itr =
          mapNamePortCard.find(portcardname);
      assert(portcardconfig_itr != mapNamePortCard.end());
      PixelPortCardConfig* portcardconfig = portcardconfig_itr->second;
      std::string TKFECID = portcardconfig->getTKFECID();
      tkfeccrates.insert(tkfecconfig->crateFromTKFECID(TKFECID));
    }
  }

  return tkfeccrates;
}

std::ostream& pos::operator<<(std::ostream& s, const PixelCalibConfiguration& calib) {
  if (!calib.parameters_.empty()) {
    s << "Parameters:" << std::endl;
    for (std::map<std::string, std::string>::const_iterator paramItr = calib.parameters_.begin();
         paramItr != calib.parameters_.end();
         ++paramItr) {
      s << paramItr->first << " " << paramItr->second << std::endl;
    }
  }

  s << "Rows:" << std::endl;
  for (unsigned int i = 0; i < calib.rows_.size(); i++) {
    for (unsigned int j = 0; j < calib.rows_[i].size(); j++) {
      s << calib.rows_[i][j] << " " << std::endl;
    }
    s << "|" << std::endl;
  }

  s << "Cols:" << std::endl;
  for (unsigned int i = 0; i < calib.cols_.size(); i++) {
    for (unsigned int j = 0; j < calib.cols_[i].size(); j++) {
      s << calib.cols_[i][j] << " " << std::endl;
    }
    s << "|" << std::endl;
  }

  s << "Vcal:" << std::endl;

  //s << calib.vcal_<<std::endl;

  s << "Vcthr:" << std::endl;

  s << calib.dacs_[0].first() << " " << calib.dacs_[0].last() << " " << calib.dacs_[0].step() << std::endl;

  s << "CalDel:" << std::endl;

  s << calib.dacs_[1].first() << " " << calib.dacs_[0].last() << " " << calib.dacs_[1].step() << std::endl;

  s << "Repeat:" << std::endl;

  s << calib.ntrigger_ << std::endl;

  return s;
}

void PixelCalibConfiguration::enablePixels(PixelFECConfigInterface* pixelFEC,
                                           unsigned int irows,
                                           unsigned int icols,
                                           pos::PixelROCMaskBits* masks,
                                           pos::PixelROCTrimBits* trims,
                                           const PixelHdwAddress& theROC) const {
  for (unsigned int irow = 0; irow < rows_[irows].size(); irow++) {
    for (unsigned int icol = 0; icol < cols_[icols].size(); icol++) {
      /*	    std::cout << "Will turn on pixel col="
		    <<cols_[icols][icol]
		    <<" row="<<rows_[irows][irow]<<std::endl;
      */
      unsigned int bits = trims->trim(cols_[icols][icol], rows_[irows][irow]);

      //if masks==0 always enable pixel
      if (masks == nullptr || masks->mask(cols_[icols][icol], rows_[irows][irow]))
        bits |= 0x80;

      pixelFEC->progpix(theROC.mfec(),
                        theROC.mfecchannel(),
                        theROC.hubaddress(),
                        theROC.portaddress(),
                        theROC.rocid(),
                        cols_[icols][icol],
                        rows_[irows][irow],
                        bits,
                        _bufferData);
    }
  }
}

void PixelCalibConfiguration::disablePixels(PixelFECConfigInterface* pixelFEC,
                                            unsigned int irows,
                                            unsigned int icols,
                                            pos::PixelROCTrimBits* trims,
                                            const PixelHdwAddress& theROC) const {
  for (unsigned int irow = 0; irow < rows_[irows].size(); irow++) {
    for (unsigned int icol = 0; icol < cols_[icols].size(); icol++) {
      /*		std::cout << "Will turn off pixel col="
			  <<cols_[old_icols][icol]
			  <<" row="<<rows_[old_irows][irow]<<std::endl;
	      */
      unsigned int bits = trims->trim(cols_[icols][icol], rows_[irows][irow]);
      pixelFEC->progpix(theROC.mfec(),
                        theROC.mfecchannel(),
                        theROC.hubaddress(),
                        theROC.portaddress(),
                        theROC.rocid(),
                        cols_[icols][icol],
                        rows_[irows][irow],
                        bits,
                        _bufferData);
    }
  }
}

void PixelCalibConfiguration::disablePixels(PixelFECConfigInterface* pixelFEC,
                                            pos::PixelROCTrimBits* trims,
                                            const PixelHdwAddress& theROC) const {
  //cout<<" disable ROC "<<theROC.hubaddress()<<" "<<theROC.rocid()<<endl;
  //FIXME This should be done with more efficient commands!
  for (unsigned int row = 0; row < 80; row++) {
    for (unsigned int col = 0; col < 52; col++) {
      unsigned int bits = trims->trim(col, row);
      pixelFEC->progpix(theROC.mfec(),
                        theROC.mfecchannel(),
                        theROC.hubaddress(),
                        theROC.portaddress(),
                        theROC.rocid(),
                        col,
                        row,
                        bits,
                        _bufferData);
    }
  }
}

std::string PixelCalibConfiguration::parameterValue(std::string parameterName) const {
  std::map<std::string, std::string>::const_iterator itr = parameters_.find(parameterName);
  if (itr == parameters_.end())  // parameterName is not in the list
  {
    return "";
  } else {
    return itr->second;
  }
}

void PixelCalibConfiguration::writeASCII(std::string dir) const {
  //FIXME this is not tested for all the use cases...

  if (!dir.empty())
    dir += "/";
  std::string filename = dir + "calib.dat";
  std::ofstream out(filename.c_str());

  out << "Mode: " << mode_ << endl;
  if (singleROC_)
    out << "SingleROC" << endl;
  if (!parameters_.empty()) {
    out << "Parameters:" << endl;
    std::map<std::string, std::string>::const_iterator it = parameters_.begin();
    for (; it != parameters_.end(); ++it) {
      out << it->first << "        " << it->second << endl;
    }
  }
  out << "Rows:" << endl;
  for (unsigned int i = 0; i < rows_.size(); i++) {
    for (unsigned int j = 0; j < rows_[i].size(); j++) {
      out << rows_[i][j] << " ";
    }
    if (i != rows_.size() - 1)
      out << "|";
    out << endl;
  }
  out << "Cols:" << endl;
  for (unsigned int i = 0; i < cols_.size(); i++) {
    for (unsigned int j = 0; j < cols_[i].size(); j++) {
      out << cols_[i][j] << " ";
    }
    if (i != cols_.size() - 1)
      out << "|";
    out << endl;
  }

  if (highVCalRange_) {
    out << "VcalHigh" << endl;
  } else {
    out << "VcalLow" << endl;
  }

  for (unsigned int i = 0; i < dacs_.size(); i++) {
    if (dacs_[i].uniformSteps()) {
      if (dacs_[i].first() != dacs_[i].last()) {
        out << "Scan: " << dacs_[i].name() << " ";
        out << dacs_[i].first() << " ";
        out << dacs_[i].last() << " ";
        out << dacs_[i].step() << endl;
      } else {
        out << "Set: " << dacs_[i].name() << " ";
        out << dacs_[i].first() << endl;
      }
    } else {
      out << "ScanValues: " << dacs_[i].name() << " ";
      for (unsigned int ival = 0; ival < dacs_[i].getNPoints(); ival++) {
        out << dacs_[i].value(ival) << " ";
      }
      out << " -1" << endl;
    }
  }

  out << "Repeat:" << endl;
  out << ntrigger_ << endl;

  if (usesROCList_) {
    out << "Rocs:" << endl;
  } else {
    out << "ToCalibrate:" << endl;
  }
  for (unsigned int i = 0; i < rocListInstructions_.size(); i++) {
    out << rocListInstructions_[i] << endl;
    if (rocListInstructions_[i] == "+" || rocListInstructions_[i] == "-") {
      out << " ";
    } else {
      out << endl;
    }
  }

  out.close();
}

unsigned int PixelCalibConfiguration::maxNumHitsPerROC() const {
  unsigned int returnValue = 0;
  for (std::vector<std::vector<unsigned int> >::const_iterator rows_itr = rows_.begin(); rows_itr != rows_.end();
       ++rows_itr) {
    for (std::vector<std::vector<unsigned int> >::const_iterator cols_itr = cols_.begin(); cols_itr != cols_.end();
         ++cols_itr) {
      unsigned int theSize = rows_itr->size() * cols_itr->size();
      returnValue = max(returnValue, theSize);
    }
  }
  return returnValue;
}

std::set<std::pair<unsigned int, unsigned int> > PixelCalibConfiguration::pixelsWithHits(unsigned int state) const {
  std::set<std::pair<unsigned int, unsigned int> > pixels;
  //                  column #      row #

  for (std::vector<unsigned int>::const_iterator col_itr = cols_[colCounter(state)].begin();
       col_itr != cols_[colCounter(state)].end();
       ++col_itr) {
    for (std::vector<unsigned int>::const_iterator row_itr = rows_[rowCounter(state)].begin();
         row_itr != rows_[rowCounter(state)].end();
         ++row_itr) {
      pixels.insert(std::pair<unsigned int, unsigned int>(*col_itr, *row_itr));
    }
  }

  return pixels;
}

bool PixelCalibConfiguration::containsScan(std::string name) const {
  for (unsigned int i = 0; i < numberOfScanVariables(); i++) {
    if (scanName(i) == name) {
      return true;
    }
  }
  return false;
}

//=============================================================================================
void PixelCalibConfiguration::writeXMLHeader(pos::PixelConfigKey key,
                                             int version,
                                             std::string path,
                                             std::ofstream* outstream,
                                             std::ofstream* out1stream,
                                             std::ofstream* out2stream) const {
  std::string mthn = "[PixelCalibConfiguration::writeXMLHeader()]\t\t    ";
  std::stringstream maskFullPath;

  writeASCII(path);

  maskFullPath << path << "/PixelCalib_Test_" << PixelTimeFormatter::getmSecTime() << ".xml";
  std::cout << __LINE__ << "]\t" << mthn << "Writing to: " << maskFullPath.str() << std::endl;

  outstream->open(maskFullPath.str().c_str());

  *outstream << "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>" << std::endl;
  *outstream << "<ROOT xmlns:xsi='http://www.w3.org/2001/XMLSchema-instance'>" << std::endl;
  *outstream << "" << std::endl;
  *outstream << " <!-- " << mthn << "-->" << std::endl;
  *outstream << "" << std::endl;
  *outstream << " <HEADER>" << std::endl;
  *outstream << "  <TYPE>" << std::endl;
  *outstream << "   <EXTENSION_TABLE_NAME>PIXEL_CALIB_CLOB</EXTENSION_TABLE_NAME>" << std::endl;
  *outstream << "   <NAME>Calibration Object Clob</NAME>" << std::endl;
  *outstream << "  </TYPE>" << std::endl;
  *outstream << "  <RUN>" << std::endl;
  *outstream << "   <RUN_TYPE>Gain Calibration</RUN_TYPE>" << std::endl;
  *outstream << "   <RUN_NUMBER>1</RUN_NUMBER>" << std::endl;
  *outstream << "   <RUN_BEGIN_TIMESTAMP>" << PixelTimeFormatter::getTime() << "</RUN_BEGIN_TIMESTAMP>" << std::endl;
  *outstream << "   <LOCATION>CERN P5</LOCATION>" << std::endl;
  *outstream << "  </RUN>" << std::endl;
  *outstream << " </HEADER>" << std::endl;
  *outstream << "" << std::endl;
  *outstream << " <DATA_SET>" << std::endl;
  *outstream << "" << std::endl;
  *outstream << "  <VERSION>" << version << "</VERSION>" << std::endl;
  *outstream << "  <COMMENT_DESCRIPTION>" << getComment() << "</COMMENT_DESCRIPTION>" << std::endl;
  *outstream << "  <CREATED_BY_USER>" << getAuthor() << "</CREATED_BY_USER>" << std::endl;
  *outstream << "" << std::endl;
  *outstream << "  <PART>" << std::endl;
  *outstream << "   <NAME_LABEL>CMS-PIXEL-ROOT</NAME_LABEL>" << std::endl;
  *outstream << "   <KIND_OF_PART>Detector ROOT</KIND_OF_PART>" << std::endl;
  *outstream << "  </PART>" << std::endl;
}

//=============================================================================================
void PixelCalibConfiguration::writeXML(std::ofstream* outstream,
                                       std::ofstream* out1stream,
                                       std::ofstream* out2stream) const {
  std::string mthn = "[PixelCalibConfiguration::writeXML()]\t\t    ";

  *outstream << " " << std::endl;
  *outstream << "  <DATA>" << std::endl;
  *outstream << "   <CALIB_OBJ_DATA_FILE>./calib.dat</CALIB_OBJ_DATA_FILE>" << std::endl;
  *outstream << "   <CALIB_TYPE>calib</CALIB_TYPE>" << std::endl;
  *outstream << "  </DATA>" << std::endl;
  *outstream << " " << std::endl;
}

//=============================================================================================
void PixelCalibConfiguration::writeXMLTrailer(std::ofstream* outstream,
                                              std::ofstream* out1stream,
                                              std::ofstream* out2stream) const {
  std::string mthn = "[PixelCalibConfiguration::writeXMLTrailer()]\t\t    ";

  *outstream << " </DATA_SET>" << std::endl;
  *outstream << "</ROOT>" << std::endl;

  outstream->close();
  std::cout << __LINE__ << "]\t" << mthn << "Data written " << std::endl;
}
