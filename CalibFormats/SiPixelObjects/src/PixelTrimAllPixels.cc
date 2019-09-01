//
// This class provide a base class for the
// pixel trim data for the pixel FEC configuration
// This is a pure interface (abstract class) that
// needs to have an implementation.
//
// Need to figure out what is 'VMEcommand' below!
//
// All applications should just use this
// interface and not care about the specific
// implementation.
//

#include <sstream>
#include <iostream>
#include <ios>
#include <cassert>
#include <stdexcept>
#include "CalibFormats/SiPixelObjects/interface/PixelTrimAllPixels.h"
#include "CalibFormats/SiPixelObjects/interface/PixelTimeFormatter.h"
#include "CalibFormats/SiPixelObjects/interface/PixelBase64.h"

using namespace pos;

PixelTrimAllPixels::PixelTrimAllPixels(std::vector<std::vector<std::string> > &tableMat) : PixelTrimBase("", "", "") {
  std::string mthn = "]\t[PixelTrimAllPixels::PixelTrimAllPixels()]\t\t    ";
  std::stringstream currentRocName;
  std::map<std::string, int> colM;
  std::vector<std::string> colNames;
  /**
       EXTENSION_TABLE_NAME: ROC_TRIMS (VIEW: CONF_KEY_ROC_TRIMS_V)

       CONFIG_KEY				 NOT NULL VARCHAR2(80)
       KEY_TYPE 				 NOT NULL VARCHAR2(80)
       KEY_ALIAS				 NOT NULL VARCHAR2(80)
       VERSION  					  VARCHAR2(40)
       KIND_OF_COND				 NOT NULL VARCHAR2(40)
       ROC_NAME 				 NOT NULL VARCHAR2(200)
       TRIM_BITS				 NOT NULL VARCHAR2(4000)
    */

  colNames.push_back("CONFIG_KEY");
  colNames.push_back("KEY_TYPE");
  colNames.push_back("KEY_ALIAS");
  colNames.push_back("VERSION");
  colNames.push_back("KIND_OF_COND");
  colNames.push_back("ROC_NAME");
  colNames.push_back("TRIM_BITS");

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
      std::cerr << __LINE__ << mthn << "Couldn't find in the database the column with name " << colNames[n]
                << std::endl;
      assert(0);
    }
  }

  //unsigned char *bits ;        /// supose to be " unsigned  char bits[tableMat[1][colM["TRIM_BLOB"]].size()] ;  "
  //char c[2080];
  std::string bits;
  trimbits_.clear();
  for (unsigned int r = 1; r < tableMat.size(); r++)  //Goes to every row of the Matrix
  {
    PixelROCName rocid(tableMat[r][colM["ROC_NAME"]]);
    PixelROCTrimBits tmp;
    tmp.read(rocid, base64_decode(tableMat[r][colM["TRIM_BITS"]]));
    trimbits_.push_back(tmp);
  }  //end for r
     //std::cout<<trimbits_.size()<<std::endl;
}  //end contructor with databasa table

PixelTrimAllPixels::PixelTrimAllPixels(std::string filename) : PixelTrimBase("", "", "") {
  if (filename[filename.size() - 1] == 't') {
    std::ifstream in(filename.c_str());

    if (!in.good())
      throw std::runtime_error("Failed to open file " + filename);
    //	std::cout << "filename =" << filename << std::endl;

    std::string s1;
    in >> s1;

    trimbits_.clear();

    while (in.good()) {
      std::string s2;
      in >> s2;

      //	    std::cout << "PixelTrimAllPixels::PixelTrimAllPixels read s1:"<<s1<< " s2:" << s2 << std::endl;

      assert(s1 == "ROC:");

      PixelROCName rocid(s2);

      //std::cout << "PixelTrimAllPixels::PixelTrimAllPixels read rocid:"<<rocid<<std::endl;

      PixelROCTrimBits tmp;

      tmp.read(rocid, in);

      trimbits_.push_back(tmp);

      in >> s1;
    }

    in.close();

  } else {
    std::ifstream in(filename.c_str(), std::ios::binary);
    if (!in.good())
      throw std::runtime_error("Failed to open file " + filename);

    char nchar;

    in.read(&nchar, 1);

    std::string s1;

    //wrote these lines of code without ref. needs to be fixed
    for (int i = 0; i < nchar; i++) {
      char c;
      in >> c;
      s1.push_back(c);
    }

    //std::cout << "READ ROC name:"<<s1<<std::endl;

    trimbits_.clear();

    while (!in.eof()) {
      //std::cout << "PixelTrimAllPixels::PixelTrimAllPixels read s1:"<<s1<<std::endl;

      PixelROCName rocid(s1);

      //std::cout << "PixelTrimAllPixels::PixelTrimAllPixels read rocid:"<<rocid<<std::endl;

      PixelROCTrimBits tmp;

      tmp.readBinary(rocid, in);

      trimbits_.push_back(tmp);

      in.read(&nchar, 1);

      s1.clear();

      if (in.eof())
        continue;

      //wrote these lines of code without ref. needs to be fixed
      for (int i = 0; i < nchar; i++) {
        char c;
        in >> c;
        s1.push_back(c);
      }
    }

    in.close();
  }

  //std::cout << "Read trimbits for "<<trimbits_.size()<<" ROCs"<<std::endl;
}

//std::string PixelTrimAllPixels::getConfigCommand(PixelMaskBase& pixelMask){
//
//  std::string s;
//  return s;
//
//}

PixelROCTrimBits PixelTrimAllPixels::getTrimBits(int ROCId) const { return trimbits_[ROCId]; }

PixelROCTrimBits *PixelTrimAllPixels::getTrimBits(PixelROCName name) {
  for (unsigned int i = 0; i < trimbits_.size(); i++) {
    if (trimbits_[i].name() == name)
      return &(trimbits_[i]);
  }

  return nullptr;
}

void PixelTrimAllPixels::generateConfiguration(PixelFECConfigInterface *pixelFEC,
                                               PixelNameTranslation *trans,
                                               const PixelMaskBase &pixelMask) const {
  for (unsigned int i = 0; i < trimbits_.size(); i++) {
    std::vector<unsigned char> trimAndMasks(4160);

    const PixelROCMaskBits &maskbits = pixelMask.getMaskBits(i);

    for (unsigned int col = 0; col < 52; col++) {
      for (unsigned int row = 0; row < 80; row++) {
        unsigned char tmp = trimbits_[i].trim(col, row);
        if (maskbits.mask(col, row) != 0)
          tmp |= 0x80;
        trimAndMasks[col * 80 + row] = tmp;
      }
    }

    // the slow way, one pixel at a time
    //pixelFEC->setMaskAndTrimAll(*(trans->getHdwAddress(trimbits_[i].name())),trimAndMasks);
    // the fast way, a full roc in column mode (& block xfer)
    const PixelHdwAddress *theROC = trans->getHdwAddress(trimbits_[i].name());
    pixelFEC->roctrimload(theROC->mfec(),
                          theROC->mfecchannel(),
                          theROC->hubaddress(),
                          theROC->portaddress(),
                          theROC->rocid(),
                          trimAndMasks);
  }
}

void PixelTrimAllPixels::writeBinary(std::string filename) const {
  std::ofstream out(filename.c_str(), std::ios::binary);

  for (unsigned int i = 0; i < trimbits_.size(); i++) {
    trimbits_[i].writeBinary(out);
  }
}

void PixelTrimAllPixels::writeASCII(std::string dir) const {
  if (!dir.empty())
    dir += "/";
  PixelModuleName module(trimbits_[0].name().rocname());
  std::string filename = dir + "ROC_Trims_module_" + module.modulename() + ".dat";

  std::ofstream out(filename.c_str());

  for (unsigned int i = 0; i < trimbits_.size(); i++) {
    trimbits_[i].writeASCII(out);
  }
}
//=============================================================================================
void PixelTrimAllPixels::writeXMLHeader(pos::PixelConfigKey key,
                                        int version,
                                        std::string path,
                                        std::ofstream *outstream,
                                        std::ofstream *out1stream,
                                        std::ofstream *out2stream) const {
  std::string mthn = "[PixelTrimAllPixels::writeXMLHeader()]\t\t\t    ";
  std::stringstream maskFullPath;

  maskFullPath << path << "/Pixel_RocTrims_" << PixelTimeFormatter::getmSecTime() << ".xml";
  std::cout << mthn << "Writing to: " << maskFullPath.str() << std::endl;

  outstream->open(maskFullPath.str().c_str());

  *outstream << "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>" << std::endl;
  *outstream << "<ROOT xmlns:xsi='http://www.w3.org/2001/XMLSchema-instance'>" << std::endl;
  *outstream << "" << std::endl;
  *outstream << " <HEADER>" << std::endl;
  *outstream << "  <TYPE>" << std::endl;
  *outstream << "   <EXTENSION_TABLE_NAME>ROC_TRIMS</EXTENSION_TABLE_NAME>" << std::endl;
  *outstream << "   <NAME>ROC Trim Bits</NAME>" << std::endl;
  *outstream << "  </TYPE>" << std::endl;
  *outstream << "  <RUN>" << std::endl;
  *outstream << "   <RUN_TYPE>ROC Trim Bits</RUN_TYPE>" << std::endl;
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
  *outstream << "" << std::endl;
}

//=============================================================================================
void PixelTrimAllPixels::writeXML(std::ofstream *outstream,
                                  std::ofstream *out1stream,
                                  std::ofstream *out2stream) const {
  std::string mthn = "[PixelTrimAllPixels::writeXML()]\t\t\t    ";

  for (unsigned int i = 0; i < trimbits_.size(); i++) {
    trimbits_[i].writeXML(outstream);
  }
}

//=============================================================================================
void PixelTrimAllPixels::writeXMLTrailer(std::ofstream *outstream,
                                         std::ofstream *out1stream,
                                         std::ofstream *out2stream) const {
  std::string mthn = "[PixelTrimAllPixels::writeXMLTrailer()]\t\t\t    ";

  *outstream << " </DATA_SET>" << std::endl;
  *outstream << "</ROOT>" << std::endl;

  outstream->close();
  std::cout << mthn << "Data written " << std::endl;
}
