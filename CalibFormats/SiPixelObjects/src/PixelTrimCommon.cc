//
// This class provide an implementation for
// pixel trim data where all pixels have the
// same settings.
//
//
// All applications should just use this
// interface and not care about the specific
// implementation
//

#include <iostream>
#include <ios>
#include "CalibFormats/SiPixelObjects/interface/PixelTrimCommon.h"
#include "CalibFormats/SiPixelObjects/interface/PixelROCName.h"

using namespace pos;

PixelTrimCommon::PixelTrimCommon(std::string filename) : PixelTrimBase("", "", "") {
  if (filename[filename.size() - 1] == 't') {
    std::ifstream in(filename.c_str());

    std::string s1;
    in >> s1;

    trimbits_.clear();

    while (!in.eof()) {
      //std::cout << "PixelTrimCommon::PixelTrimCommon read s1:"<<s1<<std::endl;

      PixelROCName rocid(in);

      //std::cout << "PixelTrimCommon::PixelTrimCommon read rocid:"<<rocid<<std::endl;

      unsigned int trimbits;

      in >> trimbits;

      trimbits_.push_back(trimbits);

      in >> s1;
    }

    in.close();

  } else {
    std::ifstream in(filename.c_str(), std::ios::binary);

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
      //std::cout << "PixelTrimCommon::PixelTrimCommon read s1:"<<s1<<std::endl;

      PixelROCName rocid(s1);

      //std::cout << "PixelTrimCommon::PixelTrimCommon read rocid:"<<rocid<<std::endl;

      unsigned int trimbits;

      in >> trimbits;

      trimbits_.push_back(trimbits);

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

//std::string PixelTrimCommon::getConfigCommand(PixelMaskBase& pixelMask){
//
//  std::string s;
//  return s;
//
//}

//PixelROCTrimBits PixelTrimCommon::getTrimBits(int ROCId) const {
//
//  return trimbits_[ROCId];
//
//}

void PixelTrimCommon::generateConfiguration(PixelFECConfigInterface* pixelFEC,
                                            PixelNameTranslation* trans,
                                            const PixelMaskBase& pixelMask) const {
  for (unsigned int i = 0; i < trimbits_.size(); i++) {
    std::vector<unsigned char> trimAndMasks(4160);

    const PixelROCMaskBits& maskbits = pixelMask.getMaskBits(i);

    for (unsigned int col = 0; col < 52; col++) {
      for (unsigned int row = 0; row < 80; row++) {
        unsigned char tmp = trimbits_[i];
        if (maskbits.mask(col, row) != 0)
          tmp |= 0x80;
        trimAndMasks[col * 80 + row] = tmp;
      }
    }

    pixelFEC->setMaskAndTrimAll(*(trans->getHdwAddress(rocname_[i])), trimAndMasks);
  }
}

void PixelTrimCommon::writeBinary(std::string filename) const {
  std::ofstream out(filename.c_str(), std::ios::binary);

  for (unsigned int i = 0; i < trimbits_.size(); i++) {
    assert(0);
    //trimbits_[i].writeBinary(out);
  }
}

void PixelTrimCommon::writeASCII(std::string filename) const {
  std::ofstream out(filename.c_str());

  for (unsigned int i = 0; i < trimbits_.size(); i++) {
    assert(0);
    //trimbits_[i].writeASCII(out);
  }
}
