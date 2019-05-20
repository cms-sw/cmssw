//
// This class provide a base class for the
// pixel mask data for the pixel FEC configuration
// This is a pure interface (abstract class) that
// needs to have an implementation.
//
// All applications should just use this
// interface and not care about the specific
// implementation
//
//
#include <sstream>
#include "CalibFormats/SiPixelObjects/interface/PixelModuleName.h"
#include "CalibFormats/SiPixelObjects/interface/PixelMaskAllPixels.h"
#include "CalibFormats/SiPixelObjects/interface/PixelTimeFormatter.h"
#include "CalibFormats/SiPixelObjects/interface/PixelBase64.h"
#include <fstream>
#include <map>
#include <iostream>
#include <cassert>
#include <stdexcept>

using namespace pos;
using namespace std;

//================================================================================================================
PixelMaskAllPixels::PixelMaskAllPixels(std::vector<std::vector<std::string> > &tableMat) : PixelMaskBase("", "", "") {
  std::string mthn = "[PixelMaskAllPixels::PixelMaskAllPixels()]\t\t    ";
  //std::cout << __LINE__ << "]\t" << mthn << "Table Size in const: " << tableMat.size() << std::endl;

  std::vector<std::string> ins = tableMat[0];
  std::map<std::string, int> colM;
  std::vector<std::string> colNames;

  /*
 EXTENSION_TABLE_NAME: ROC_MASKS (VIEW: CONF_KEY_ROC_MASKS_V)

 CONFIG_KEY				   NOT NULL VARCHAR2(80)
 KEY_TYPE				   NOT NULL VARCHAR2(80)
 KEY_ALIAS				   NOT NULL VARCHAR2(80)
 VERSION					    VARCHAR2(40)
 KIND_OF_COND				   NOT NULL VARCHAR2(40)
 ROC_NAME				   NOT NULL VARCHAR2(200)
 KILL_MASK				   NOT NULL VARCHAR2(4000) colNames.push_back("CONFIG_KEY_ID" );
*/
  colNames.push_back("CONFIG_KEY");
  colNames.push_back("KEY_TYPE");
  colNames.push_back("KEY_ALIAS");
  colNames.push_back("VERSION");
  colNames.push_back("KIND_OF_COND");
  colNames.push_back("ROC_NAME");
  colNames.push_back("KILL_MASK");

  for (unsigned int c = 0; c < ins.size(); c++) {
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
  maskbits_.clear();
  for (unsigned int r = 1; r < tableMat.size(); r++) {  //Goes to every row of the Matrix
    std::string currentRocName = tableMat[r][colM["ROC_NAME"]];
    PixelROCName rocid(currentRocName);
    PixelROCMaskBits tmp;
    tmp.read(rocid,
             base64_decode(tableMat[r][colM["KILL_MASK"]]));  // decode back from specially base64-encoded data for XML
    maskbits_.push_back(tmp);
  }  //end for r
}

//================================================================================================================
// modified by MR on 18-04-2008 10:02:00
PixelMaskAllPixels::PixelMaskAllPixels() : PixelMaskBase("", "", "") { ; }

//================================================================================================================
void PixelMaskAllPixels::addROCMaskBits(const PixelROCMaskBits &bits) { maskbits_.push_back(bits); }

//================================================================================================================
PixelMaskAllPixels::PixelMaskAllPixels(std::string filename) : PixelMaskBase("", "", "") {
  std::string mthn = "[PixelMaskAllPixels::PixelMaskAllPixels()]\t\t    ";

  if (filename[filename.size() - 1] == 't') {
    std::ifstream in(filename.c_str());

    if (!in.good()) {
      std::cout << __LINE__ << "]\t" << mthn << "Could not open: " << filename << std::endl;
      throw std::runtime_error("Failed to open file " + filename);
    }

    std::string tag;
    in >> tag;

    maskbits_.clear();

    while (!in.eof()) {
      PixelROCName rocid(in);

      PixelROCMaskBits tmp;

      tmp.read(rocid, in);

      maskbits_.push_back(tmp);

      in >> tag;
    }

    in.close();

  } else {
    std::ifstream in(filename.c_str(), std::ios::binary);

    char nchar;

    in.read(&nchar, 1);

    //in >> nchar;

    std::string s1;

    //wrote these lines of code without ref. needs to be fixed
    for (int i = 0; i < nchar; i++) {
      char c;
      in >> c;
      s1.push_back(c);
    }

    //std::cout << __LINE__ << "]\t" << mthn << "READ ROC name: "  << s1    << std::endl;

    maskbits_.clear();

    while (!in.eof()) {
      //std::cout << __LINE__ << "]\t" << mthn << "read s1: "    << s1    << std::endl;

      PixelROCName rocid(s1);

      //std::cout << __LINE__ << "]\t" << mthn << "read rocid: " << rocid << std::endl;

      PixelROCMaskBits tmp;

      tmp.readBinary(rocid, in);

      maskbits_.push_back(tmp);

      in.read(&nchar, 1);

      s1.clear();

      if (in.eof())
        continue;

      //std::cout << __LINE__ << "]\t" << mthn << "Will read: " << (int)nchar << " characters." <<std::endl;

      //wrote these lines of code without ref. needs to be fixed
      for (int i = 0; i < nchar; i++) {
        char c;
        in >> c;
        //std::cout << " " <<c;
        s1.push_back(c);
      }
      //std::cout << std::endl;
    }

    in.close();
  }

  //std::cout << __LINE__ << "]\t" << mthn << "Read maskbits for " << maskbits_.size() << " ROCs" << std::endl;
}

//================================================================================================================
const PixelROCMaskBits &PixelMaskAllPixels::getMaskBits(int ROCId) const { return maskbits_[ROCId]; }

//================================================================================================================
PixelROCMaskBits *PixelMaskAllPixels::getMaskBits(PixelROCName name) {
  for (unsigned int i = 0; i < maskbits_.size(); i++) {
    if (maskbits_[i].name() == name)
      return &(maskbits_[i]);
  }

  return nullptr;
}

//================================================================================================================
void PixelMaskAllPixels::writeBinary(std::string filename) const {
  std::ofstream out(filename.c_str(), std::ios::binary);

  for (unsigned int i = 0; i < maskbits_.size(); i++) {
    maskbits_[i].writeBinary(out);
  }
}

//================================================================================================================
void PixelMaskAllPixels::writeASCII(std::string dir) const {
  if (!dir.empty())
    dir += "/";
  PixelModuleName module(maskbits_[0].name().rocname());
  std::string filename = dir + "ROC_Masks_module_" + module.modulename() + ".dat";

  std::ofstream out(filename.c_str());

  for (unsigned int i = 0; i < maskbits_.size(); i++) {
    maskbits_[i].writeASCII(out);
  }
}

//=============================================================================================
void PixelMaskAllPixels::writeXMLHeader(pos::PixelConfigKey key,
                                        int version,
                                        std::string path,
                                        std::ofstream *outstream,
                                        std::ofstream *out1stream,
                                        std::ofstream *out2stream) const {
  std::string mthn = "[PixelMaskAllPixels::writeXMLHeader()]\t\t\t    ";
  std::stringstream maskFullPath;

  maskFullPath << path << "/Pixel_RocMasks_" << PixelTimeFormatter::getmSecTime() << ".xml";
  std::cout << __LINE__ << "]\t" << mthn << "Writing to: " << maskFullPath.str() << std::endl;

  outstream->open(maskFullPath.str().c_str());

  *outstream << "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>" << std::endl;
  *outstream << "<ROOT xmlns:xsi='http://www.w3.org/2001/XMLSchema-instance'>" << std::endl;
  *outstream << "" << std::endl;
  *outstream << " <HEADER>" << std::endl;
  *outstream << "  <TYPE>" << std::endl;
  *outstream << "   <EXTENSION_TABLE_NAME>ROC_MASKS</EXTENSION_TABLE_NAME>" << std::endl;
  *outstream << "   <NAME>ROC Mask Bits</NAME>" << std::endl;
  *outstream << "  </TYPE>" << std::endl;
  *outstream << "  <RUN>" << std::endl;
  *outstream << "   <RUN_TYPE>ROC Mask Bits</RUN_TYPE>" << std::endl;
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
  *outstream << "  " << std::endl;
}
//=============================================================================================
void PixelMaskAllPixels::writeXML(std::ofstream *outstream,
                                  std::ofstream *out1stream,
                                  std::ofstream *out2stream) const {
  std::string mthn = "[PixelMaskAllPixels::writeXML()]\t\t\t    ";

  for (unsigned int i = 0; i < maskbits_.size(); i++) {
    maskbits_[i].writeXML(outstream);
  }
}
//=============================================================================================
void PixelMaskAllPixels::writeXMLTrailer(std::ofstream *outstream,
                                         std::ofstream *out1stream,
                                         std::ofstream *out2stream) const {
  std::string mthn = "[PixelMaskAllPixels::writeXMLTrailer()]\t\t\t    ";

  *outstream << "  " << std::endl;
  *outstream << " </DATA_SET>" << std::endl;
  *outstream << "</ROOT>" << std::endl;

  outstream->close();
  std::cout << __LINE__ << "]\t" << mthn << "Data written " << std::endl;
}
