//
// This class stores the name and related
// hardware mapings for a ROC
//

#include "CalibFormats/SiPixelObjects/interface/PixelROCName.h"
#include <string>
#include <iostream>
#include <sstream>
#include <cctype>
#include <cassert>
#include <cstdlib>

using namespace std;
using namespace pos;

PixelROCName::PixelROCName() : id_(0) {}

PixelROCName::PixelROCName(std::string rocname) { parsename(rocname); }

void PixelROCName::setIdFPix(char np, char LR, int disk, int blade, int panel, int plaquet, int roc) {
  std::string mthn = "[PixelROCName::setIdFPix()]\t\t\t\t    ";
  id_ = 0;

  //std::cout << __LINE__ << "]\t" << mthn << "subdet: " << subdet << std::endl;
  //std::cout << __LINE__ << "]\t" << mthn << "np    : " << np     << std::endl;
  //std::cout << __LINE__ << "]\t" << mthn << "LR    : " << LR     << std::endl;
  //std::cout << __LINE__ << "]\t" << mthn << "disk  : " << disk   << std::endl;

  assert(roc >= 0 && roc < 10);

  if (np == 'p')
    id_ = (id_ | 0x40000000);
  //std::cout<< __LINE__ << "]\t" << mthn <<"2 id_="<<std::hex<<id_<<std::dec<<std::endl;
  if (LR == 'I')
    id_ = (id_ | 0x20000000);
  //std::cout<< __LINE__ << "]\t" << mthn <<"3 id_="<<std::hex<<id_<<std::dec<<std::endl;
  id_ = (id_ | (disk << 12));
  //std::cout<< __LINE__ << "]\t" << mthn <<"4 id_="<<std::hex<<id_<<std::dec<<std::endl;
  id_ = (id_ | (blade << 7));
  //std::cout<< __LINE__ << "]\t" << mthn <<"5 id_="<<std::hex<<id_<<std::dec<<std::endl;
  id_ = (id_ | ((panel - 1) << 6));
  //std::cout<< __LINE__ << "]\t" << mthn <<"6 id_="<<std::hex<<id_<<std::dec<<std::endl;
  id_ = (id_ | ((plaquet - 1) << 4));
  //std::cout<< __LINE__ << "]\t" << mthn <<"7 id_="<<std::hex<<id_<<std::dec<<std::endl;
  id_ = (id_ | roc);

  //std::cout<< __LINE__ << "]\t" << mthn <<"final id_="<<std::hex<<id_<<std::dec<<std::endl;
}

void PixelROCName::setIdBPix(char np, char LR, int sec, int layer, int ladder, char HF, int module, int roc) {
  id_ = 0;

  //std::cout<< __LINE__ << "]\t" << mthn <<"BPix ladder:"<<ladder<<std::endl;
  //std::cout<< __LINE__ << "]\t" << mthn <<"np  : " << np   << std::endl;
  //std::cout<< __LINE__ << "]\t" << mthn <<"LR  : " << LR   << std::endl;
  //std::cout<< __LINE__ << "]\t" << mthn <<"disk: " << disk << std::endl;

  assert(roc >= 0 && roc < 16);

  id_ = 0x80000000;

  if (np == 'p')
    id_ = (id_ | 0x40000000);
  //std::cout<< __LINE__ << "]\t" << mthn <<"2 id_="<<std::hex<<id_<<std::dec<<std::endl;
  if (LR == 'I')
    id_ = (id_ | 0x20000000);
  //std::cout<< __LINE__ << "]\t" << mthn <<"3 id_="<<std::hex<<id_<<std::dec<<std::endl;
  id_ = (id_ | ((sec - 1) << 14));
  //std::cout<< __LINE__ << "]\t" << mthn <<"4 id_="<<std::hex<<id_<<std::dec<<std::endl;
  if (HF == 'F')
    id_ = (id_ | 0x00000800);

  id_ = (id_ | (layer << 12));
  //std::cout<< __LINE__ << "]\t" << mthn <<"5 id_="<<std::hex<<id_<<std::dec<<std::endl;
  id_ = (id_ | (ladder << 6));
  //std::cout<< __LINE__ << "]\t" << mthn <<"6 id_="<<std::hex<<id_<<std::dec<<std::endl;
  id_ = (id_ | ((module - 1) << 4));
  //std::cout<< __LINE__ << "]\t" << mthn <<"7 id_="<<std::hex<<id_<<std::dec<<std::endl;
  id_ = (id_ | roc);

  //std::cout<< __LINE__ << "]\t" << mthn <<"final id_="<<std::hex<<id_<<std::dec<<std::endl;
}

void PixelROCName::check(bool check, const string& name) {
  static std::string mthn = "[PixelROCName::check()]\t\t\t\t    ";

  if (check)
    return;

  cout << __LINE__ << "]\t" << mthn << "ERROR tried to parse string:'" << name;
  cout << "' as a ROC name. Will terminate." << endl;

  ::abort();
}

void PixelROCName::parsename(std::string name) {
  //
  // The name should be on the format
  //
  // FPix_BpR_D1_BLD1_PNL1_PLQ1_ROC1
  //

  //    std::cout << "[PixelROCName::parsename()]\t\tROC name:"<<name<<std::endl;

  check(name[0] == 'F' || name[0] == 'B', name);

  if (name[0] == 'F') {
    check(name[0] == 'F', name);
    check(name[1] == 'P', name);
    check(name[2] == 'i', name);
    check(name[3] == 'x', name);
    check(name[4] == '_', name);
    check(name[5] == 'B', name);
    check((name[6] == 'm') || (name[6] == 'p'), name);
    char np = name[6];
    check((name[7] == 'I') || (name[7] == 'O'), name);
    char LR = name[7];
    check(name[8] == '_', name);
    check(name[9] == 'D', name);
    char digit[2] = {0, 0};
    digit[0] = name[10];
    int disk = atoi(digit);
    check(name[11] == '_', name);
    check(name[12] == 'B', name);
    check(name[13] == 'L', name);
    check(name[14] == 'D', name);
    check(std::isdigit(name[15]), name);
    digit[0] = name[15];
    int bld = atoi(digit);
    unsigned int offset = 0;
    if (std::isdigit(name[16])) {
      digit[0] = name[16];
      bld = 10 * bld + atoi(digit);
      offset++;
    }
    check(name[16 + offset] == '_', name);
    check(name[17 + offset] == 'P', name);
    check(name[18 + offset] == 'N', name);
    check(name[19 + offset] == 'L', name);
    check(std::isdigit(name[20 + offset]), name);
    digit[0] = name[20 + offset];
    int pnl = atoi(digit);
    check(name[21 + offset] == '_', name);
    check(name[22 + offset] == 'P', name);
    check(name[23 + offset] == 'L', name);
    check(name[24 + offset] == 'Q', name);
    check(std::isdigit(name[25 + offset]), name);
    digit[0] = name[25 + offset];
    int plq = atoi(digit);
    check(name[26 + offset] == '_', name);
    check(name[27 + offset] == 'R', name);
    check(name[28 + offset] == 'O', name);
    check(name[29 + offset] == 'C', name);
    check(std::isdigit(name[30 + offset]), name);
    digit[0] = name[30 + offset];
    int roc = atoi(digit);
    if (name.size() == 32 + offset) {
      digit[0] = name[31 + offset];
      roc = roc * 10 + atoi(digit);
    }

    setIdFPix(np, LR, disk, bld, pnl, plq, roc);
  } else {
    check(name[0] == 'B', name);
    check(name[1] == 'P', name);
    check(name[2] == 'i', name);
    check(name[3] == 'x', name);
    check(name[4] == '_', name);
    check(name[5] == 'B', name);
    check((name[6] == 'm') || (name[6] == 'p'), name);
    char np = name[6];
    check((name[7] == 'I') || (name[7] == 'O'), name);
    char LR = name[7];
    check(name[8] == '_', name);
    check(name[9] == 'S', name);
    check(name[10] == 'E', name);
    check(name[11] == 'C', name);
    char digit[2] = {0, 0};
    digit[0] = name[12];
    int sec = atoi(digit);
    check(name[13] == '_', name);
    check(name[14] == 'L', name);
    check(name[15] == 'Y', name);
    check(name[16] == 'R', name);
    check(std::isdigit(name[17]), name);
    digit[0] = name[17];
    int layer = atoi(digit);
    check(name[18] == '_', name);
    check(name[19] == 'L', name);
    check(name[20] == 'D', name);
    check(name[21] == 'R', name);
    check(std::isdigit(name[22]), name);
    digit[0] = name[22];
    int ladder = atoi(digit);
    unsigned int offset = 0;
    if (std::isdigit(name[23])) {
      offset++;
      digit[0] = name[22 + offset];
      ladder = 10 * ladder + atoi(digit);
    }
    check(name[23 + offset] == 'H' || name[23 + offset] == 'F', name);
    char HF = name[23 + offset];
    check(name[24 + offset] == '_', name);
    check(name[25 + offset] == 'M', name);
    check(name[26 + offset] == 'O', name);
    check(name[27 + offset] == 'D', name);
    check(std::isdigit(name[28 + offset]), name);
    digit[0] = name[28 + offset];
    int module = atoi(digit);
    check(name[29 + offset] == '_', name);
    check(name[30 + offset] == 'R', name);
    check(name[31 + offset] == 'O', name);
    check(name[32 + offset] == 'C', name);
    check(std::isdigit(name[33 + offset]), name);
    digit[0] = name[33 + offset];
    int roc = atoi(digit);
    if (name.size() == 35 + offset) {
      digit[0] = name[34 + offset];
      roc = roc * 10 + atoi(digit);
    }

    setIdBPix(np, LR, sec, layer, ladder, HF, module, roc);
  }
}

PixelROCName::PixelROCName(std::ifstream& s) {
  std::string tmp;

  s >> tmp;

  parsename(tmp);
}

std::string PixelROCName::rocname() const {
  std::string s;

  std::ostringstream s1;

  if (detsub() == 'F') {
    s1 << "FPix";
    s1 << "_B";
    s1 << mp();
    s1 << IO();
    s1 << "_D";
    s1 << disk();
    s1 << "_BLD";
    s1 << blade();
    s1 << "_PNL";
    s1 << panel();
    s1 << "_PLQ";
    s1 << plaquet();
    s1 << "_ROC";
    s1 << roc();

    assert(roc() >= 0 && roc() <= 10);
  } else {
    s1 << "BPix";
    s1 << "_B";
    s1 << mp();
    s1 << IO();
    s1 << "_SEC";
    s1 << sec();
    s1 << "_LYR";
    s1 << layer();
    s1 << "_LDR";
    s1 << ladder();
    s1 << HF();
    s1 << "_MOD";
    s1 << module();
    s1 << "_ROC";
    s1 << roc();

    assert(roc() >= 0 && roc() <= 15);
  }

  s = s1.str();

  return s;
}

std::ostream& pos::operator<<(std::ostream& s, const PixelROCName& pixelroc) {
  // FPix_BpR_D1_BLD1_PNL1_PLQ1_ROC1

  s << pixelroc.rocname();

  return s;
}

const PixelROCName& PixelROCName::operator=(const PixelROCName& aROC) {
  id_ = aROC.id_;

  return *this;
}
