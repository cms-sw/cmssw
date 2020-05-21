#include "CalibFormats/SiPixelObjects/interface/PixelFEDTestDAC.h"
#include "CalibFormats/SiPixelObjects/interface/PixelTimeFormatter.h"
#include <cstring>
#include <cassert>
#include <map>
#include <sstream>

using namespace std;

using namespace pos;

PixelFEDTestDAC::PixelFEDTestDAC(std::vector<std::vector<std::string> > &tableMat) {
  std::string mthn = "[PixelFEDTestDAC::PixelFEDTestDAC()]\t\t\t    ";
  const unsigned long int UB = 200;
  const unsigned long int B = 500;
  const unsigned long int offset = 0;
  vector<unsigned int> pulseTrain(256), pixelDCol(1), pixelPxl(2), pixelTBMHeader(3), pixelTBMTrailer(3);
  unsigned int DCol, LorR, start = 15;
  std::string line;
  std::string::size_type loc1, loc2, loc3, loc4;
  unsigned long int npos = std::string::npos;
  int i;

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
      std::cerr << __LINE__ << "]\t" << mthn << "Couldn't find in the database the column with name " << colNames[n]
                << std::endl;
      assert(0);
    }
  }

  std::istringstream fin;
  fin.str(tableMat[1][colM["CALIB_OBJ_DATA_CLOB"]]);

  // Initialise the pulseTrain to offset+black
  for (unsigned int i = 0; i < pulseTrain.size(); ++i) {
    pulseTrain[i] = offset + B;
  }

  i = start;

  getline(fin, line);
  mode_ = line;
  assert(mode_ == "EmulatedPhysics" || mode_ == "FEDBaselineWithTestDACs" || mode_ == "FEDAddressLevelWithTestDACs");

  while (!fin.eof()) {
    getline(fin, line);

    if (line.find("TBMHeader") != npos) {
      loc1 = line.find('(');
      if (loc1 == npos) {
        cout << __LINE__ << "]\t" << mthn << "'(' not found after TBMHeader.\n";
        break;
      }
      loc2 = line.find(')', loc1 + 1);
      if (loc2 == npos) {
        cout << __LINE__ << "]\t" << mthn << "')' not found after TBMHeader.\n";
        break;
      }
      int TBMHeader = atoi(line.substr(loc1 + 1, loc2 - loc1 - 1).c_str());

      pulseTrain[i] = UB;
      ++i;
      pulseTrain[i] = UB;
      ++i;
      pulseTrain[i] = UB;
      ++i;
      pulseTrain[i] = B;
      ++i;

      pixelTBMHeader = decimalToBaseX(TBMHeader, 4, 4);

      pulseTrain[i] = levelEncoder(pixelTBMHeader[3]);
      ++i;
      pulseTrain[i] = levelEncoder(pixelTBMHeader[2]);
      ++i;
      pulseTrain[i] = levelEncoder(pixelTBMHeader[1]);
      ++i;
      pulseTrain[i] = levelEncoder(pixelTBMHeader[0]);
      ++i;
    } else if (line.find("ROCHeader") != std::string::npos) {
      loc1 = line.find('(');
      if (loc1 == npos) {
        cout << __LINE__ << "]\t" << mthn << "'(' not found after ROCHeader.\n";
        break;
      }
      loc2 = line.find(')', loc1 + 1);
      if (loc2 == npos) {
        cout << __LINE__ << "]\t" << mthn << "')' not found after ROCHeader.\n";
        break;
      }
      int LastDAC = atoi(line.substr(loc1 + 1, loc2 - loc1 - 1).c_str());

      std::cout << "--------------" << std::endl;

      pulseTrain[i] = UB;
      ++i;
      pulseTrain[i] = B;
      ++i;
      pulseTrain[i] = levelEncoder(LastDAC);
      ++i;
    } else if (line.find("PixelHit") != std::string::npos) {
      loc1 = line.find('(');
      if (loc1 == npos) {
        cout << __LINE__ << "]\t" << mthn << "'(' not found after PixelHit.\n";
        break;
      }
      loc2 = line.find(',', loc1 + 1);
      if (loc2 == npos) {
        cout << __LINE__ << "]\t" << mthn << "',' not found after the first argument of PixelHit.\n";
        break;
      }
      loc3 = line.find(',', loc2 + 1);
      if (loc3 == npos) {
        cout << __LINE__ << "]\t" << mthn << "'.' not found after the second argument of PixelHit.\n";
        break;
      }
      loc4 = line.find(')', loc3 + 1);
      if (loc4 == npos) {
        cout << __LINE__ << "]\t" << mthn << "')' not found after the third argument of PixelHit.\n";
        break;
      }
      int column = atoi(line.substr(loc1 + 1, loc2 - loc1 - 1).c_str());
      int row = atoi(line.substr(loc2 + 1, loc3 - loc2 - 1).c_str());
      int charge = atoi(line.substr(loc3 + 1, loc4 - loc3 - 1).c_str());

      DCol = int(column / 2);
      LorR = int(column - DCol * 2);
      pixelDCol = decimalToBaseX(DCol, 6, 2);
      pixelPxl = decimalToBaseX((80 - row) * 2 + LorR, 6, 3);

      std::cout << "Pxl = " << pixelPxl[2] << pixelPxl[1] << pixelPxl[0] << ", DCol= " << pixelDCol[1] << pixelDCol[0]
                << std::endl;

      pulseTrain[i] = levelEncoder(pixelDCol[1]);
      ++i;
      pulseTrain[i] = levelEncoder(pixelDCol[0]);
      ++i;
      pulseTrain[i] = levelEncoder(pixelPxl[2]);
      ++i;
      pulseTrain[i] = levelEncoder(pixelPxl[1]);
      ++i;
      pulseTrain[i] = levelEncoder(pixelPxl[0]);
      ++i;
      pulseTrain[i] = charge;
      ++i;

    } else if (line.find("TBMTrailer") != std::string::npos) {
      loc1 = line.find('(');
      if (loc1 == npos) {
        cout << __LINE__ << "]\t" << mthn << "'(' not found after TBMTrailer.\n";
        break;
      }
      loc2 = line.find(')', loc1 + 1);
      if (loc2 == npos) {
        cout << __LINE__ << "]\t" << mthn << "')' not found after TBMTrailer.\n";
        break;
      }
      int TBMTrailer = atoi(line.substr(loc1 + 1, loc2 - loc1 - 1).c_str());

      pulseTrain[i] = UB;
      ++i;
      pulseTrain[i] = UB;
      ++i;
      pulseTrain[i] = B;
      ++i;
      pulseTrain[i] = B;
      ++i;

      pixelTBMTrailer = decimalToBaseX(TBMTrailer, 4, 4);
      pulseTrain[i] = levelEncoder(pixelTBMTrailer[3]);
      ++i;
      pulseTrain[i] = levelEncoder(pixelTBMTrailer[2]);
      ++i;
      pulseTrain[i] = levelEncoder(pixelTBMTrailer[1]);
      ++i;
      pulseTrain[i] = levelEncoder(pixelTBMTrailer[0]);
      ++i;
    }
  }
  //   fin.close();
  dacs_ = pulseTrain;
}

PixelFEDTestDAC::PixelFEDTestDAC(std::string filename) {
  std::string mthn = "[PixelFEDTestDAC::PixelFEDTestDAC()]\t\t\t\t    ";
  const unsigned long int UB = 200;
  const unsigned long int B = 500;
  const unsigned long int offset = 0;
  vector<unsigned int> pulseTrain(256), pixelDCol(1), pixelPxl(2), pixelTBMHeader(3), pixelTBMTrailer(3);
  unsigned int DCol, LorR, start = 15;
  std::string line;
  std::string::size_type loc1, loc2, loc3, loc4;
  unsigned long int npos = std::string::npos;
  int i;

  // Initialise the pulseTrain to offset+black
  for (unsigned int i = 0; i < pulseTrain.size(); ++i) {
    pulseTrain[i] = offset + B;
  }

  ifstream fin(filename.c_str());

  i = start;

  getline(fin, line);
  mode_ = line;
  assert(mode_ == "EmulatedPhysics" || mode_ == "FEDBaselineWithTestDACs" || mode_ == "FEDAddressLevelWithTestDACs");

  while (!fin.eof()) {
    getline(fin, line);

    if (line.find("TBMHeader") != npos) {
      loc1 = line.find('(');
      if (loc1 == npos) {
        cout << __LINE__ << "]\t" << mthn << "'(' not found after TBMHeader.\n";
        break;
      }
      loc2 = line.find(')', loc1 + 1);
      if (loc2 == npos) {
        cout << __LINE__ << "]\t" << mthn << "')' not found after TBMHeader.\n";
        break;
      }
      int TBMHeader = atoi(line.substr(loc1 + 1, loc2 - loc1 - 1).c_str());

      pulseTrain[i] = UB;
      ++i;
      pulseTrain[i] = UB;
      ++i;
      pulseTrain[i] = UB;
      ++i;
      pulseTrain[i] = B;
      ++i;

      pixelTBMHeader = decimalToBaseX(TBMHeader, 4, 4);

      pulseTrain[i] = levelEncoder(pixelTBMHeader[3]);
      ++i;
      pulseTrain[i] = levelEncoder(pixelTBMHeader[2]);
      ++i;
      pulseTrain[i] = levelEncoder(pixelTBMHeader[1]);
      ++i;
      pulseTrain[i] = levelEncoder(pixelTBMHeader[0]);
      ++i;
    } else if (line.find("ROCHeader") != std::string::npos) {
      loc1 = line.find('(');
      if (loc1 == npos) {
        cout << __LINE__ << "]\t" << mthn << "'(' not found after ROCHeader.\n";
        break;
      }
      loc2 = line.find(')', loc1 + 1);
      if (loc2 == npos) {
        cout << __LINE__ << "]\t" << mthn << "')' not found after ROCHeader.\n";
        break;
      }
      int LastDAC = atoi(line.substr(loc1 + 1, loc2 - loc1 - 1).c_str());

      std::cout << "--------------" << std::endl;

      pulseTrain[i] = UB;
      ++i;
      pulseTrain[i] = B;
      ++i;
      pulseTrain[i] = levelEncoder(LastDAC);
      ++i;
    } else if (line.find("PixelHit") != std::string::npos) {
      loc1 = line.find('(');
      if (loc1 == npos) {
        cout << __LINE__ << "]\t" << mthn << "'(' not found after PixelHit.\n";
        break;
      }
      loc2 = line.find(',', loc1 + 1);
      if (loc2 == npos) {
        cout << __LINE__ << "]\t" << mthn << "',' not found after the first argument of PixelHit.\n";
        break;
      }
      loc3 = line.find(',', loc2 + 1);
      if (loc3 == npos) {
        cout << __LINE__ << "]\t" << mthn << "'.' not found after the second argument of PixelHit.\n";
        break;
      }
      loc4 = line.find(')', loc3 + 1);
      if (loc4 == npos) {
        cout << __LINE__ << "]\t" << mthn << "')' not found after the third argument of PixelHit.\n";
        break;
      }
      int column = atoi(line.substr(loc1 + 1, loc2 - loc1 - 1).c_str());
      int row = atoi(line.substr(loc2 + 1, loc3 - loc2 - 1).c_str());
      int charge = atoi(line.substr(loc3 + 1, loc4 - loc3 - 1).c_str());

      DCol = int(column / 2);
      LorR = int(column - DCol * 2);
      pixelDCol = decimalToBaseX(DCol, 6, 2);
      pixelPxl = decimalToBaseX((80 - row) * 2 + LorR, 6, 3);

      std::cout << "Pxl = " << pixelPxl[2] << pixelPxl[1] << pixelPxl[0] << ", DCol= " << pixelDCol[1] << pixelDCol[0]
                << std::endl;

      pulseTrain[i] = levelEncoder(pixelDCol[1]);
      ++i;
      pulseTrain[i] = levelEncoder(pixelDCol[0]);
      ++i;
      pulseTrain[i] = levelEncoder(pixelPxl[2]);
      ++i;
      pulseTrain[i] = levelEncoder(pixelPxl[1]);
      ++i;
      pulseTrain[i] = levelEncoder(pixelPxl[0]);
      ++i;
      pulseTrain[i] = charge;
      ++i;

    } else if (line.find("TBMTrailer") != std::string::npos) {
      loc1 = line.find('(');
      if (loc1 == npos) {
        cout << __LINE__ << "]\t" << mthn << "'(' not found after TBMTrailer.\n";
        break;
      }
      loc2 = line.find(')', loc1 + 1);
      if (loc2 == npos) {
        cout << __LINE__ << "]\t" << mthn << "')' not found after TBMTrailer.\n";
        break;
      }
      int TBMTrailer = atoi(line.substr(loc1 + 1, loc2 - loc1 - 1).c_str());

      pulseTrain[i] = UB;
      ++i;
      pulseTrain[i] = UB;
      ++i;
      pulseTrain[i] = B;
      ++i;
      pulseTrain[i] = B;
      ++i;

      pixelTBMTrailer = decimalToBaseX(TBMTrailer, 4, 4);
      pulseTrain[i] = levelEncoder(pixelTBMTrailer[3]);
      ++i;
      pulseTrain[i] = levelEncoder(pixelTBMTrailer[2]);
      ++i;
      pulseTrain[i] = levelEncoder(pixelTBMTrailer[1]);
      ++i;
      pulseTrain[i] = levelEncoder(pixelTBMTrailer[0]);
      ++i;
    }
  }
  fin.close();
  dacs_ = pulseTrain;
}

unsigned int PixelFEDTestDAC::levelEncoder(int level) {
  unsigned int pulse;

  switch (level) {
    case 0:
      pulse = 450;
      break;
    case 1:
      pulse = 500;
      break;
    case 2:
      pulse = 550;
      break;
    case 3:
      pulse = 600;
      break;
    case 4:
      pulse = 650;
      break;
    case 5:
      pulse = 700;
      break;
    default:
      assert(0);
      break;
  }

  return pulse;
}

vector<unsigned int> PixelFEDTestDAC::decimalToBaseX(unsigned int a, unsigned int x, unsigned int length) {
  vector<unsigned int> ans(100, 0);
  int i = 0;

  while (a > 0) {
    ans[i] = a % x;
    //ans.push_back(a%x);
    a = a / x;
    i += 1;
  }

  if (length > 0)
    ans.resize(length);
  else
    ans.resize(i);

  return ans;
}

//=============================================================================================
void PixelFEDTestDAC::writeXMLHeader(pos::PixelConfigKey key,
                                     int version,
                                     std::string path,
                                     std::ofstream *outstream,
                                     std::ofstream *out1stream,
                                     std::ofstream *out2stream) const {
  std::string mthn = "[PixelFEDTestDAC::writeXMLHeader()]\t\t\t    ";
  std::stringstream maskFullPath;

  //  writeASCII(path) ;

  maskFullPath << path << "/PixelCalib_Test_" << PixelTimeFormatter::getmSecTime() << ".xml";
  std::cout << __LINE__ << "]\t" << mthn << "Writing to: " << maskFullPath.str() << std::endl;

  outstream->open(maskFullPath.str().c_str());

  *outstream << "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>" << std::endl;
  *outstream << "<ROOT xmlns:xsi='http://www.w3.org/2001/XMLSchema-instance'>" << std::endl;
  *outstream << "" << std::endl;
  *outstream << " <HEADER>" << std::endl;
  *outstream << "  <TYPE>" << std::endl;
  *outstream << "   <EXTENSION_TABLE_NAME>PIXEL_CALIB_CLOB</EXTENSION_TABLE_NAME>" << std::endl;
  *outstream << "   <NAME>Calibration Object Clob</NAME>" << std::endl;
  *outstream << "  </TYPE>" << std::endl;
  *outstream << "  <RUN>" << std::endl;
  *outstream << "   <RUN_TYPE>PixelFEDTestDAC</RUN_TYPE>" << std::endl;
  *outstream << "   <RUN_NUMBER>1</RUN_NUMBER>" << std::endl;
  *outstream << "   <RUN_BEGIN_TIMESTAMP>" << PixelTimeFormatter::getTime() << "</RUN_BEGIN_TIMESTAMP>" << std::endl;
  *outstream << "   <LOCATION>CERN P5</LOCATION>" << std::endl;
  *outstream << "  </RUN>" << std::endl;
  *outstream << " </HEADER>" << std::endl;
  *outstream << "" << std::endl;
  *outstream << " <DATA_SET>" << std::endl;
  *outstream << "" << std::endl;
  *outstream << "  <VERSION>" << version << "</VERSION>" << std::endl;
  *outstream << "  <COMMENT_DESCRIPTION>No comment defined: this class does NOT inherit from "
                "PixelCalibBase</COMMENT_DESCRIPTION>"
             << std::endl;
  *outstream << "  <CREATED_BY_USER>Unknown user</CREATED_BY_USER>" << std::endl;
  *outstream << "" << std::endl;
  *outstream << "  <PART>" << std::endl;
  *outstream << "   <NAME_LABEL>CMS-PIXEL-ROOT</NAME_LABEL>" << std::endl;
  *outstream << "   <KIND_OF_PART>Detector ROOT</KIND_OF_PART>" << std::endl;
  *outstream << "  </PART>" << std::endl;
}

//=============================================================================================
void PixelFEDTestDAC::writeXML(std::ofstream *outstream, std::ofstream *out1stream, std::ofstream *out2stream) const {
  std::string mthn = "[PixelFEDTestDAC::writeXML()]\t\t\t    ";

  *outstream << " " << std::endl;
  *outstream << "  <DATA>" << std::endl;
  *outstream << "   <CALIB_OBJ_DATA_FILE>./fedtestdac.dat</CALIB_OBJ_DATA_FILE>" << std::endl;
  *outstream << "   <CALIB_TYPE>fedtestdac</CALIB_TYPE>" << std::endl;
  *outstream << "  </DATA>" << std::endl;
  *outstream << " " << std::endl;
}

//=============================================================================================
void PixelFEDTestDAC::writeXMLTrailer(std::ofstream *outstream,
                                      std::ofstream *out1stream,
                                      std::ofstream *out2stream) const {
  std::string mthn = "[PixelFEDTestDAC::writeXMLTrailer()]\t\t\t    ";

  *outstream << " </DATA_SET>" << std::endl;
  *outstream << "</ROOT>" << std::endl;

  outstream->close();
  std::cout << __LINE__ << "]\t" << mthn << "Data written " << std::endl;
}
