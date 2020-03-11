#include "DQM/SiPixelMonitorClient/interface/SiPixelContinuousPalette.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelUtility.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include <cstdlib>

using namespace std;
//
// Get a list of MEs in a folder
//
int SiPixelUtility::getMEList(string name, vector<string> &values) {
  values.clear();
  string prefix_str = name.substr(0, (name.find(":")));
  prefix_str += "/";
  string temp_str = name.substr(name.find(":") + 1);
  split(temp_str, values, ",");
  for (vector<string>::iterator it = values.begin(); it != values.end(); it++)
    (*it).insert(0, prefix_str);
  return values.size();
}
//
// Get a list of MEs in a folder and the path name
//
int SiPixelUtility::getMEList(string name, string &dir_path, vector<string> &values) {
  values.clear();
  dir_path = name.substr(0, (name.find(":")));
  dir_path += "/";
  string temp_str = name.substr(name.find(":") + 1);
  split(temp_str, values, ",");
  return values.size();
}

// Check if the requested ME exists in a folder
bool SiPixelUtility::checkME(string name, string me_name, string &full_path) {
  if (name.find(name) == string::npos)
    return false;
  string prefix_str = name.substr(0, (name.find(":")));
  prefix_str += "/";
  string temp_str = name.substr(name.find(":") + 1);
  vector<string> values;
  split(temp_str, values, ",");
  for (vector<string>::iterator it = values.begin(); it != values.end(); it++) {
    if ((*it).find(me_name) != string::npos) {
      full_path = prefix_str + (*it);
      return true;
    }
  }
  return false;
}
//
// -- Split a given string into a number of strings using given
//    delimiters and fill a vector with splitted strings
//
void SiPixelUtility::split(const string &str, vector<string> &tokens, const string &delimiters) {
  // Skip delimiters at beginning.
  string::size_type lastPos = str.find_first_not_of(delimiters, 0);

  // Find first "non-delimiter".
  string::size_type pos = str.find_first_of(delimiters, lastPos);

  while (string::npos != pos || string::npos != lastPos) {
    // Found a token, add it to the vector.
    tokens.push_back(str.substr(lastPos, pos - lastPos));

    // Skip delimiters.  Note the "not_of"
    lastPos = str.find_first_not_of(delimiters, pos);

    // Find next "non-delimiter"
    pos = str.find_first_of(delimiters, lastPos);
  }
}
//
// -- Get Color code from Status
//
void SiPixelUtility::getStatusColor(int status, int &rval, int &gval, int &bval) {
  if (status == dqm::qstatus::STATUS_OK) {
    rval = 0;
    gval = 255;
    bval = 0;
  } else if (status == dqm::qstatus::WARNING) {
    rval = 255;
    gval = 255;
    bval = 0;
  } else if (status == dqm::qstatus::ERROR) {
    rval = 255;
    gval = 0;
    bval = 0;
  } else if (status == dqm::qstatus::OTHER) {
    rval = 255;
    gval = 150;
    bval = 0;
  } else {
    rval = 0;
    gval = 0;
    bval = 255;
  }
}
//
// -- Get Color code from Status
//
void SiPixelUtility::getStatusColor(int status, int &icol, string &tag) {
  if (status == dqm::qstatus::STATUS_OK) {
    tag = "Ok";
    icol = 3;
  } else if (status == dqm::qstatus::WARNING) {
    tag = "Warning";
    icol = 5;
  } else if (status == dqm::qstatus::ERROR) {
    tag = "Error";
    icol = 2;
  } else if (status == dqm::qstatus::OTHER) {
    tag = "Other";
    icol = 1;
  } else {
    tag = " ";
    icol = 1;
  }
}
//
// -- Get Color code from Status
//
void SiPixelUtility::getStatusColor(double status, int &rval, int &gval, int &bval) {
  rval = SiPixelContinuousPalette::r[(int)(status * 100)];
  gval = SiPixelContinuousPalette::g[(int)(status * 100)];
  bval = SiPixelContinuousPalette::b[(int)(status * 100)];
}
//
// -- Get Status of Monitor Element
//
int SiPixelUtility::getStatus(MonitorElement *me) {
  int status = 0;
  if (me->getQReports().empty()) {
    status = 0;
  } else if (me->hasError()) {
    status = dqm::qstatus::ERROR;
  } else if (me->hasWarning()) {
    status = dqm::qstatus::WARNING;
  } else if (me->hasOtherReport()) {
    status = dqm::qstatus::OTHER;
  } else {
    status = dqm::qstatus::STATUS_OK;
  }
  return status;
}

vector<string> SiPixelUtility::getQTestNameList(MonitorElement *me) {
  vector<string> qtestNameList;
  return qtestNameList;
}

int SiPixelUtility::computeErrorCode(int status) {
  int code = 0;
  switch (status) {
    case dqm::qstatus::INSUF_STAT:
      code = 1;
      break;
    case dqm::qstatus::WARNING:
      code = 2;
      break;
    case dqm::qstatus::ERROR:
      code = 3;
      break;
  }  // end switch

  return code;
}

int SiPixelUtility::computeHistoBin(string &module_path) {
  int module_bin = 0;

  int module = 0;
  int shell = 0;
  int layer = 0;
  int ladder = 0;
  int halfcylinder = 0;
  int disk = 0;
  int blade = 0;
  int panel = 0;

  int nbinShell = 192;
  int nbinLayer = 0;
  int nbinLadder = 4;

  int nbinHalfcylinder = 168;
  int nbinDisk = 84;
  int nbinBlade = 7;
  int nbinPanel = 0;

  vector<string> subDirVector;
  SiPixelUtility::split(module_path, subDirVector, "/");

  for (vector<string>::const_iterator it = subDirVector.begin(); it != subDirVector.end(); it++) {
    if ((*it).find("Collector") != string::npos ||
        //(*it).find("Collated") != string::npos ||
        (*it).find("FU") != string::npos || (*it).find("Pixel") != string::npos ||
        (*it).find("Barrel") != string::npos || (*it).find("Endcap") != string::npos)
      continue;

    if ((*it).find("Module") != string::npos) {
      module = atoi((*it).substr((*it).find("_") + 1).c_str());
    }

    if ((*it).find("Shell") != string::npos) {
      if ((*it).find("mI") != string::npos)
        shell = 1;
      if ((*it).find("mO") != string::npos)
        shell = 2;
      if ((*it).find("pI") != string::npos)
        shell = 3;
      if ((*it).find("pO") != string::npos)
        shell = 4;
    }
    if ((*it).find("Layer") != string::npos) {
      layer = atoi((*it).substr((*it).find("_") + 1).c_str());
      if (layer == 1) {
        nbinLayer = 0;
      }
      if (layer == 2) {
        nbinLayer = 40;
      }
      if (layer == 3) {
        nbinLayer = 40 + 64;
      }
    }
    if ((*it).find("Ladder") != string::npos) {
      ladder = atoi((*it).substr((*it).find("_") + 1, 2).c_str());
    }
    if ((*it).find("HalfCylinder") != string::npos) {
      if ((*it).find("mI") != string::npos)
        halfcylinder = 1;
      if ((*it).find("mO") != string::npos)
        halfcylinder = 2;
      if ((*it).find("pI") != string::npos)
        halfcylinder = 3;
      if ((*it).find("pO") != string::npos)
        halfcylinder = 4;
    }
    if ((*it).find("Disk") != string::npos) {
      disk = atoi((*it).substr((*it).find("_") + 1).c_str());
    }
    if ((*it).find("Blade") != string::npos) {
      blade = atoi((*it).substr((*it).find("_") + 1, 2).c_str());
    }
    if ((*it).find("Panel") != string::npos) {
      panel = atoi((*it).substr((*it).find("_") + 1).c_str());
      if (panel == 1)
        nbinPanel = 0;
      if (panel == 2)
        nbinPanel = 4;
    }
  }
  if (module_path.find("Barrel") != string::npos) {
    module_bin = module + (ladder - 1) * nbinLadder + nbinLayer + (shell - 1) * nbinShell;
  }
  if (module_path.find("Endcap") != string::npos) {
    module_bin = module + (panel - 1) * nbinPanel + (blade - 1) * nbinBlade + (disk - 1) * nbinDisk +
                 (halfcylinder - 1) * nbinHalfcylinder;
  }

  return module_bin;

  //  cout << "leaving SiPixelInformationExtractor::computeHistoBin" << endl;
}

void SiPixelUtility::fillPaveText(TPaveText *pave, const map<string, pair<int, double>> &messages) {
  TText *sourceCodeOnCanvas;
  for (map<string, pair<int, double>>::const_iterator it = messages.begin(); it != messages.end(); it++) {
    string message = it->first;
    int color = (it->second).first;
    double size = (it->second).second;
    sourceCodeOnCanvas = pave->AddText(message.c_str());
    sourceCodeOnCanvas->SetTextColor(color);
    sourceCodeOnCanvas->SetTextSize(size);
    sourceCodeOnCanvas->SetTextFont(112);
  }
}

map<string, string> SiPixelUtility::sourceCodeMap() {
  map<string, string> sourceCode;
  for (int iSource = 0; iSource < 5; iSource++) {
    string type;
    string code;
    switch (iSource) {
      case 0:
        type = "RAW";
        code = "1    ";
        break;
      case 1:
        type = "DIG";
        code = "10   ";
        break;
      case 2:
        type = "CLU";
        code = "100  ";
        break;
      case 3:
        type = "TRK";
        code = "1000 ";
        break;
      case 4:
        type = "REC";
        code = "10000";
        break;
    }  // end of switch
    sourceCode[type] = code;
  }
  return sourceCode;
}

void SiPixelUtility::createStatusLegendMessages(map<string, pair<int, double>> &messages) {
  for (int iStatus = 1; iStatus < 5; iStatus++) {
    pair<int, double> color_size;
    int color = 1;
    double size = 0.03;
    string code;
    string type;
    color_size.second = size;
    switch (iStatus) {
      case 1:
        code = "1";
        type = "INSUF_STAT";
        color = kBlue;
        break;
      case 2:
        code = "2";
        type = "WARNING(S)";
        color = kYellow;
        break;
      case 3:
        code = "3";
        type = "ERROR(S)  ";
        color = kRed;
        break;
      case 4:
        code = "4";
        type = "ERRORS    ";
        color = kMagenta;
        break;
    }  // end of switch
    string messageString = code + ": " + type;
    color_size.first = color;
    messages[messageString] = color_size;
  }
}

//------------------------------------------------------------------------------
//
// -- Set Drawing Option
//
void SiPixelUtility::setDrawingOption(TH1 *hist, float xlow, float xhigh) {
  if (!hist)
    return;

  TAxis *xa = hist->GetXaxis();
  TAxis *ya = hist->GetYaxis();

  xa->SetTitleOffset(0.7);
  xa->SetTitleSize(0.06);
  xa->SetLabelSize(0.04);
  //  xa->SetLabelColor(0);

  ya->SetTitleOffset(0.7);
  ya->SetTitleSize(0.06);

  if (xlow != -1 && xhigh != -1.0) {
    xa->SetRangeUser(xlow, xhigh);
  }
}
