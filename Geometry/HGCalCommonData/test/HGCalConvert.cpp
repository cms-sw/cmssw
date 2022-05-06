//////////////////////////////////////////////////////////////////////////////
//
//  Contains 3 macros to extract from the ASCII files from Philippe, Chris
//  and Katya to make part of the xml file to be used to define the
//  geometry of the HGCal detector. There are 3 versions for the silicon
//  part for the format versions V14|V15, version V16 and verion V17
//
//  HGCalConvert 0/2 infile outfile1 outfile2 outfile3 modeGlobal debug
//               0 for the V14/V15 formats; 1 for the V16 format;
//               2 for the V17 format
//  infile   (const char*)   Input file from Philippe conatining layer #,
//                           size, depth, x, y position, orientation, u, v
//  outfile1 (const char*)   Output fle for the EE part
//  outfile2 (const char*)   Output fle for the HE part (full silicon)
//  outfile3 (const char*)   Output fle for the HE part (mixed layers)
//  modeGlobal (int)         Flag to create parts to be inserted in the
//                           global section (0) or to be inserted in
//                           the ddAlgorithm part (1)
//  debug     (int)          Three digit integer to set debug for each
//                           of the outputs
//
//  HGCalConvert 3 infile outfile1 outfile2 laymin cassette debug
//  infile   (const char*)   Input file from Katya (modified by Chris)
//                           containing layer #. ring #, start and end
//                           of ring radius, SiPM size, 4 hexadecimal
//                           words specifying if the phi location in
//                           120-degree sector is filled, scintillator
//                           type (c: cast, m: mould)
//  outfile1 (const char*)   Output file for the part to be inserted in
//                           the global section
//  outfile2 (const char*)   Output file for the part to be inserted in
//                           the ddAlgorithm part
//  laymin   (int)           First layer number of the HE part
//                           (28 for versions: V14, V15; 26 for V16, V17)
//  cassette (int)           Cassettes are used in geometry definition
//                           (0 if nonone, 1 if 12 cassettes are used)
//  debug    (int)           Two digit integer to set debug for each
//                           of the outputs
//
//////////////////////////////////////////////////////////////////////////////

#include "Geometry/HGCalCommonData/interface/HGCalProperty.h"
#include "Geometry/HGCalCommonData/interface/HGCalTileIndex.h"
#include "Geometry/HGCalCommonData/interface/HGCalTypes.h"
#include "Geometry/HGCalCommonData/interface/HGCalWaferIndex.h"
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

struct wafer {
  int thick, partial, orient, cassette;
  wafer(int t = 0, int p = 0, int o = 0, int c = 0) : thick(t), partial(p), orient(o), cassette(c){};
};

struct layerInfo {
  int layer, type;
  std::vector<double> deltaR;
  layerInfo(int ly, int ty, std::vector<double> dR) : layer(ly), type(ty), deltaR(dR){};
};

struct tile {
  double sipm;
  int type, hex[4];
  tile(double s = 0, int h1 = 0, int h2 = 0, int h3 = 0, int h4 = 0, int t = 0) : sipm(s), type(t) {
    hex[0] = h1;
    hex[1] = h2;
    hex[2] = h3;
    hex[3] = h4;
  };
};

struct tileZone {
  tileZone(int l0 = 0, int r0 = 0, int r1 = 0, int f0 = 0, int f1 = 0, int c0 = 0)
      : layer(l0), rmin(r0), rmax(r1), phimin(f0), phimax(f1), cassette(c0){};
  int layer, rmin, rmax, phimin, phimax, cassette;
};

std::vector<std::string> splitString(const std::string& fLine);

class ConvertSiliconV0 {
public:
  ConvertSiliconV0(int layMax1 = 28, int layMax2 = 36);
  void convert(const char* infile,
               const char* outfile1,
               const char* outfile2,
               const char* outfile3,
               int modeGlobal = 0,
               int debug = 0);

private:
  void writeSilicon(const char*, const std::map<int, wafer>&, const std::string&, const bool&, const bool&);

  const int layMax1_, layMax2_;
};

class ConvertSiliconV1 {
public:
  ConvertSiliconV1(int layMax1 = 26, int layMax2 = 33);
  void convert(const char* infile,
               const char* outfile1,
               const char* outfile2,
               const char* outfile3,
               int modeGlobal = 0,
               int debug = 0);

private:
  void writeSilicon(
      const char*, const std::vector<int>&, const std::map<int, wafer>&, const std::string&, const bool&, const bool&);

  const int layMax1_, layMax2_;
};

class ConvertSiliconV2 {
public:
  ConvertSiliconV2(unsigned int layMax1 = 26, unsigned int layMax2 = 33, unsigned int layMax3 = 47);
  void convert(const char* infile,
               const char* outfile1,
               const char* outfile2,
               const char* outfile3,
               int modeGlobal = 0,
               int debug = 0);

private:
  void writeSilicon(const char*,
                    const unsigned int,
                    const std::vector<layerInfo>&,
                    const std::map<int, wafer>&,
                    const std::string&,
                    const bool&,
                    const bool&);

  const unsigned int layMax1_, layMax2_, layMax3_;
};

class ConvertScintillator {
public:
  ConvertScintillator(int layMin = 26, int cassette = 0);
  void convert(const char*, const char*, const char*, int debug = 0);

private:
  void makeTitle(const char* outfile,
                 const std::map<int, tile>& module,
                 const std::map<int, std::pair<double, double> >& ringR,
                 int lmin,
                 int lmax,
                 bool debug);

  const int layMin_;
  const int cassette_;
};

int main(int argc, char* argv[]) {
  if (argc < 7) {
    std::cout << "Please give a minimum of 7 arguments \n"
              << "mode (0, 1:silicon; 2:scintillator)\n"
              << "input file name\n"
              << "output file name\n"
              << "for silicon 4 additional parameters:\n"
              << "  second output file name\n"
              << "  third output file name\n"
              << "  output mode (0: gobal; 1: local)\n"
              << "  debug (three digis to set debug for each output)\n"
              << "for scintillator 4 additional parameters after the first 3\n"
              << "  second output file name\n"
              << "  number of layers in the EE section: 28 or 26\n"
              << "  flag to utilize cassette partition or not\n"
              << "  debug flag\n"
              << std::endl;
    return 0;
  }

  int mode = std::atoi(argv[1]);
  if ((mode <= 2) && (argc < 7)) {
    std::cout << "Please give a minimum of 7 arguments \n"
              << "mode == " << mode << "\n"
              << "input file name\n"
              << "output file name\n"
              << "second output file name\n"
              << "third output file name\n"
              << "output mode (0: gobal; 1: local)\n"
              << "debug (three digis to set debug for each output)\n"
              << std::endl;
    return 0;
  }

  const char* infile = argv[2];
  if (mode <= 2) {
    const char* outfile1 = argv[3];
    const char* outfile2 = argv[4];
    const char* outfile3 = argv[5];
    int modeGlobal = atoi(argv[6]);
    int debug = atoi(argv[7]);
    std::cout << "Calls ConvertSilicon for i/p file " << infile << " o/p files " << outfile1 << ":" << outfile2 << ":"
              << outfile3 << " Mode " << modeGlobal << " Debug " << debug << std::endl;
    if (mode == 0) {
      ConvertSiliconV0 c1;
      c1.convert(infile, outfile1, outfile2, outfile3, modeGlobal, debug);
    } else if (mode == 1) {
      ConvertSiliconV1 c1;
      c1.convert(infile, outfile1, outfile2, outfile3, modeGlobal, debug);
    } else {
      ConvertSiliconV2 c1;
      c1.convert(infile, outfile1, outfile2, outfile3, modeGlobal, debug);
    }
  } else {
    const char* outfile1 = argv[3];
    const char* outfile2 = argv[4];
    int laymin = atoi(argv[5]);
    int cassette = atoi(argv[6]);
    int debug = atoi(argv[7]);
    std::cout << "Calls ConvertScintillator for i/p file " << infile << " o/p files " << outfile1 << ":" << outfile2
              << " Laymin " << laymin << " Cassette " << cassette << " Debug " << debug << std::endl;
    ConvertScintillator c1(laymin, cassette);
    c1.convert(infile, outfile1, outfile2, debug);
  }
  return 0;
}

std::vector<std::string> splitString(const std::string& fLine) {
  std::vector<std::string> result;
  int start = 0;
  bool empty = true;
  for (unsigned i = 0; i <= fLine.size(); i++) {
    if (fLine[i] == ' ' || i == fLine.size()) {
      if (!empty) {
        std::string item(fLine, start, i - start);
        result.push_back(item);
        empty = true;
      }
      start = i + 1;
    } else {
      if (empty)
        empty = false;
    }
  }
  return result;
}

ConvertSiliconV0::ConvertSiliconV0(int layMax1, int layMax2) : layMax1_(layMax1), layMax2_(layMax2) {}

void ConvertSiliconV0::convert(
    const char* infile, const char* outfile1, const char* outfile2, const char* outfile3, int modeGlobal, int debug) {
  std::ifstream fInput(infile);
  if (!fInput.good()) {
    std::cout << "Cannot open file " << infile << std::endl;
  } else {
    //First read in all records
    char buffer[1024];
    std::string thick[3] = {"120", "200", "300"};
    std::string partial[8] = {"F", "b", "g", "gm", "a", "d", "dm", "c"};
    std::map<int, wafer> module1, module2, module3;
    unsigned int all(0), comments(0), others(0), bad(0), good(0);
    bool global = (modeGlobal < 1);
    while (fInput.getline(buffer, 1024)) {
      ++all;
      if (debug % 10 > 1)
        std::cout << "[" << all << "] " << buffer << std::endl;
      if (buffer[0] == '#') {
        ++comments;
      } else {
        ++others;
        std::vector<std::string> items = splitString(std::string(buffer));
        if (items.size() != 8) {
          ++bad;
        } else {
          ++good;
          int layer = std::atoi(items[0].c_str());
          int waferU = std::atoi(items[6].c_str());
          int waferV = std::atoi(items[7].c_str());
          int thck = static_cast<int>(std::find(thick, thick + 3, items[2]) - thick);
          int part = static_cast<int>(std::find(partial, partial + 8, items[1]) - partial);
          int orient = std::atoi(items[5].c_str());
          wafer waf(thck, part, orient, 0);
          if (layer <= layMax1_) {
            int index = HGCalWaferIndex::waferIndex(layer, waferU, waferV, false);
            module1[index] = waf;
          } else if ((layer <= layMax2_) || global) {
            int index = HGCalWaferIndex::waferIndex(layer - layMax1_, waferU, waferV, false);
            module2[index] = waf;
          } else {
            int index = HGCalWaferIndex::waferIndex(layer - layMax1_, waferU, waferV, false);
            module3[index] = waf;
          }
        }
      }
    }
    fInput.close();
    std::cout << "Read " << all << " records with " << comments << " comments " << others
              << " non-comment records out of which " << good << ":" << module1.size() << ":" << module2.size() << ":"
              << module3.size() << " are good and " << bad << " are bad\n"
              << std::endl;
    //Now write separately for EE, HEsil and HEmix
    writeSilicon(outfile1, module1, "EE", global, (debug % 10 > 0));
    // Next HEsil part
    writeSilicon(outfile2, module2, "HE", global, ((debug / 10) % 10 > 0));
    // Finally HEmix part
    if (!global)
      writeSilicon(outfile3, module3, "HE", global, ((debug / 100) % 10 > 0));
  }
}

void ConvertSiliconV0::writeSilicon(const char* outfile,
                                    const std::map<int, wafer>& module,
                                    const std::string& tag,
                                    const bool& mode,
                                    const bool& debug) {
  char apost('"');
  unsigned int k1(0), k2(0);
  std::map<int, wafer>::const_iterator itr;
  std::string blank("  ");
  std::ofstream fOut(outfile);
  std::vector<int> layerStart;
  int layer(-1);
  if (mode) {
    blank = "  ";
    fOut << blank << "<Vector name=" << apost << "WaferIndex" << tag << apost << " type=" << apost << "numeric" << apost
         << " nEntries=" << apost << module.size() << apost << ">";
  } else {
    blank = "    ";
    fOut << blank << "<Vector name=" << apost << "WaferIndex" << apost << " type=" << apost << "numeric" << apost
         << " nEntries=" << apost << module.size() << apost << ">";
  }
  for (itr = module.begin(); itr != module.end(); ++itr) {
    std::string last = ((k1 + 1) == module.size()) ? " " : ",";
    if (k1 % 7 == 0)
      fOut << "\n  " << blank << std::setw(8) << itr->first << last;
    else
      fOut << std::setw(8) << itr->first << last;
    if (HGCalWaferIndex::waferLayer(itr->first) != layer) {
      layerStart.emplace_back(k1);
      layer = HGCalWaferIndex::waferLayer(itr->first);
    }
    ++k1;
    if (debug)
      std::cout << "Wafer " << HGCalWaferIndex::waferLayer(itr->first) << ":" << HGCalWaferIndex::waferU(itr->first)
                << ":" << HGCalWaferIndex::waferV(itr->first) << " T " << (itr->second).thick << " P "
                << (itr->second).partial << " O " << (itr->second).orient << std::endl;
  }
  fOut << "\n" << blank << "</Vector>\n";
  if (mode)
    fOut << blank << "<Vector name=" << apost << "WaferProperties" << tag << apost << " type=" << apost << "numeric"
         << apost << " nEntries=" << apost << module.size() << apost << ">";
  else
    fOut << blank << "<Vector name=" << apost << "WaferProperties" << apost << " type=" << apost << "numeric" << apost
         << " nEntries=" << apost << module.size() << apost << ">";
  for (itr = module.begin(); itr != module.end(); ++itr) {
    int property = HGCalProperty::waferProperty(
        (itr->second).thick, (itr->second).partial, (itr->second).orient, (itr->second).cassette);
    std::string last = ((k2 + 1) == module.size()) ? " " : ",";
    if (k2 % 10 == 0)
      fOut << "\n  " << blank << std::setw(5) << property << last;
    else
      fOut << std::setw(5) << property << last;
    ++k2;
  }
  fOut << "\n" << blank << "</Vector>\n";
  if (mode) {
    fOut << blank << "<Vector name=" << apost << "WaferLayerStart" << tag << apost << " type=" << apost << "numeric"
         << apost << " nEntries=" << apost << layerStart.size() << apost << ">";
  } else {
    fOut << blank << "<Vector name=" << apost << "WaferLayerStart" << apost << " type=" << apost << "numeric" << apost
         << " nEntries=" << apost << layerStart.size() << apost << ">";
  }
  for (unsigned k3 = 0; k3 < layerStart.size(); ++k3) {
    std::string last = ((k3 + 1) == layerStart.size()) ? " " : ",";
    if (k3 % 10 == 0)
      fOut << "\n  " << blank << std::setw(5) << layerStart[k3] << last;
    else
      fOut << std::setw(5) << layerStart[k3] << last;
  }
  fOut << "\n" << blank << "</Vector>\n";
  fOut.close();
}

ConvertSiliconV1::ConvertSiliconV1(int layMax1, int layMax2) : layMax1_(layMax1), layMax2_(layMax2) {}

void ConvertSiliconV1::convert(
    const char* infile, const char* outfile1, const char* outfile2, const char* outfile3, int modeGlobal, int debug) {
  std::ifstream fInput(infile);
  if (!fInput.good()) {
    std::cout << "Cannot open file " << infile << std::endl;
  } else {
    //First read in all records
    char buffer[1024];
    std::string thick[4] = {"h120", "l200", "l300", "h200"};
    int addType[4] = {HGCalTypes::WaferFineThin,
                      HGCalTypes::WaferCoarseThin,
                      HGCalTypes::WaferCoarseThick,
                      HGCalTypes::WaferFineThick};
    const int partTypeH[6] = {HGCalTypes::WaferFull,
                              HGCalTypes::WaferHalf2,
                              HGCalTypes::WaferChopTwoM,
                              HGCalTypes::WaferSemi2,
                              HGCalTypes::WaferSemi2,
                              HGCalTypes::WaferFive2};
    const int partTypeL[7] = {HGCalTypes::WaferFull,
                              HGCalTypes::WaferHalf,
                              HGCalTypes::WaferHalf,
                              HGCalTypes::WaferSemi,
                              HGCalTypes::WaferSemi,
                              HGCalTypes::WaferFive,
                              HGCalTypes::WaferThree};
    std::map<int, wafer> module1, module2, module3;
    unsigned int all(0), comments(0), others(0), bad(0), good(0);
    int layers(0);
    std::vector<int> layer1, layer2, layer3;
    bool global = (modeGlobal < 1);
    while (fInput.getline(buffer, 1024)) {
      ++all;
      if (debug % 10 > 1)
        std::cout << "[" << all << "] " << buffer << std::endl;
      if (buffer[0] == '#') {
        ++comments;
      } else {
        ++others;
        std::vector<std::string> items = splitString(std::string(buffer));
        if (others == 1) {
          layers = std::atoi(items[0].c_str());
        } else if (others == 2) {
          if (items.size() == static_cast<unsigned int>(layers)) {
            for (int k = 0; k < layers; ++k) {
              int ltype = std::atoi(items[k].c_str());
              if (k < layMax1_) {
                layer1.emplace_back(ltype);
              } else if ((k < layMax2_) || global) {
                layer2.emplace_back(ltype);
              } else {
                layer3.emplace_back(ltype);
              }
            }
          } else {
            ++bad;
          }
        } else if (items.size() != 8) {
          ++bad;
        } else {
          ++good;
          int layer = std::atoi(items[0].c_str());
          int waferU = std::atoi(items[6].c_str());
          int waferV = std::atoi(items[7].c_str());
          int thck = static_cast<int>(std::find(thick, thick + 4, items[2]) - thick);
          int part = std::atoi(items[1].c_str());
          if ((thck < 4) && (part >= 0)) {
            if ((addType[thck] == HGCalTypes::WaferFineThin) || (addType[thck] == HGCalTypes::WaferFineThick))
              part = partTypeH[part];
            else
              part = partTypeL[part];
          }
          int orient = std::atoi(items[5].c_str());
          wafer waf(thck, part, orient, 0);
          if (layer <= layMax1_) {
            int index = HGCalWaferIndex::waferIndex(layer, waferU, waferV, false);
            module1[index] = waf;
          } else if ((layer <= layMax2_) || global) {
            int index = HGCalWaferIndex::waferIndex(layer - layMax1_, waferU, waferV, false);
            module2[index] = waf;
          } else {
            int index = HGCalWaferIndex::waferIndex(layer - layMax1_, waferU, waferV, false);
            module3[index] = waf;
          }
        }
      }
    }
    fInput.close();
    std::cout << "Read " << all << " records with " << comments << " comments " << others
              << " non-comment records out of which " << good << ":" << module1.size() << ":" << module2.size() << ":"
              << module3.size() << " are good and " << bad << " are bad and with " << layers << " layers\n";
    std::cout << "\nThere are " << layer1.size() << " of types:" << std::endl;
    for (const auto& l : layer1)
      std::cout << " " << l;
    std::cout << "\nThere are " << layer2.size() << " of types:" << std::endl;
    for (const auto& l : layer2)
      std::cout << " " << l;
    if (layer3.size() > 0) {
      std::cout << "\nThere are " << layer3.size() << " of types:" << std::endl;
      for (const auto& l : layer3)
        std::cout << " " << l;
    }
    std::cout << std::endl << std::endl;

    //Now write separately for EE, HEsil and HEmix
    writeSilicon(outfile1, layer1, module1, "EE", global, (debug % 10 > 0));
    // Next HEsil part
    writeSilicon(outfile2, layer2, module2, "HE", global, ((debug / 10) % 10 > 0));
    // Finally HEmix part
    if (!global)
      writeSilicon(outfile3, layer3, module3, "HE", global, ((debug / 100) % 10 > 0));
  }
}

void ConvertSiliconV1::writeSilicon(const char* outfile,
                                    const std::vector<int>& layers,
                                    const std::map<int, wafer>& module,
                                    const std::string& tag,
                                    const bool& mode,
                                    const bool& debug) {
  char apost('"');
  unsigned int k0(0), k1(0), k2(0);
  std::map<int, wafer>::const_iterator itr;
  std::string blank = (mode) ? "  " : "    ";
  std::ofstream fOut(outfile);
  std::vector<int> layerStart;
  int layer(-1);
  if (mode)
    fOut << blank << "<Vector name=" << apost << "LayerTypes" << tag << apost << " type=" << apost << "numeric" << apost
         << " nEntries=" << apost << layers.size() << apost << ">";
  else
    fOut << blank << "<Vector name=" << apost << "LayerTypes" << apost << " type=" << apost << "numeric" << apost
         << " nEntries=" << apost << layers.size() << apost << ">";
  for (const auto& l : layers) {
    std::string last = ((k0 + 1) == layers.size()) ? " " : ",";
    if (k0 % 20 == 0)
      fOut << "\n  " << blank << std::setw(2) << l << last;
    else
      fOut << std::setw(2) << l << last;
    ++k0;
  }
  fOut << "\n" << blank << "</Vector>\n";
  if (mode) {
    fOut << blank << "<Vector name=" << apost << "WaferIndex" << tag << apost << " type=" << apost << "numeric" << apost
         << " nEntries=" << apost << module.size() << apost << ">";
  } else {
    fOut << blank << "<Vector name=" << apost << "WaferIndex" << apost << " type=" << apost << "numeric" << apost
         << " nEntries=" << apost << module.size() << apost << ">";
  }
  for (itr = module.begin(); itr != module.end(); ++itr) {
    std::string last = ((k1 + 1) == module.size()) ? " " : ",";
    if (k1 % 7 == 0)
      fOut << "\n  " << blank << std::setw(8) << itr->first << last;
    else
      fOut << std::setw(8) << itr->first << last;
    if (HGCalWaferIndex::waferLayer(itr->first) != layer) {
      layerStart.emplace_back(k1);
      layer = HGCalWaferIndex::waferLayer(itr->first);
    }
    ++k1;
    if (debug)
      std::cout << "Wafer " << HGCalWaferIndex::waferLayer(itr->first) << ":" << HGCalWaferIndex::waferU(itr->first)
                << ":" << HGCalWaferIndex::waferV(itr->first) << " T " << (itr->second).thick << " P "
                << (itr->second).partial << " O " << (itr->second).orient << std::endl;
  }
  fOut << "\n" << blank << "</Vector>\n";
  if (mode)
    fOut << blank << "<Vector name=" << apost << "WaferProperties" << tag << apost << " type=" << apost << "numeric"
         << apost << " nEntries=" << apost << module.size() << apost << ">";
  else
    fOut << blank << "<Vector name=" << apost << "WaferProperties" << apost << " type=" << apost << "numeric" << apost
         << " nEntries=" << apost << module.size() << apost << ">";
  for (itr = module.begin(); itr != module.end(); ++itr) {
    int property = HGCalProperty::waferProperty(
        (itr->second).thick, (itr->second).partial, (itr->second).orient, (itr->second).cassette);
    std::string last = ((k2 + 1) == module.size()) ? " " : ",";
    if (k2 % 10 == 0)
      fOut << "\n  " << blank << std::setw(5) << property << last;
    else
      fOut << std::setw(5) << property << last;
    ++k2;
  }
  fOut << "\n" << blank << "</Vector>\n";
  if (mode) {
    fOut << blank << "<Vector name=" << apost << "WaferLayerStart" << tag << apost << " type=" << apost << "numeric"
         << apost << " nEntries=" << apost << layerStart.size() << apost << ">";
  } else {
    fOut << blank << "<Vector name=" << apost << "WaferLayerStart" << apost << " type=" << apost << "numeric" << apost
         << " nEntries=" << apost << layerStart.size() << apost << ">";
  }
  for (unsigned k3 = 0; k3 < layerStart.size(); ++k3) {
    std::string last = ((k3 + 1) == layerStart.size()) ? " " : ",";
    if (k3 % 10 == 0)
      fOut << "\n  " << blank << std::setw(5) << layerStart[k3] << last;
    else
      fOut << std::setw(5) << layerStart[k3] << last;
  }
  fOut << "\n" << blank << "</Vector>\n";
  fOut.close();
}

ConvertSiliconV2::ConvertSiliconV2(unsigned int layMax1, unsigned int layMax2, unsigned int layMax3)
    : layMax1_(layMax1), layMax2_(layMax2), layMax3_(layMax3) {}

void ConvertSiliconV2::convert(
    const char* infile, const char* outfile1, const char* outfile2, const char* outfile3, int modeGlobal, int debug) {
  std::ifstream fInput(infile);
  if (!fInput.good()) {
    std::cout << "Cannot open file " << infile << std::endl;
  } else {
    //First read in all records
    char buffer[1024];
    const int thksize = 4;
    std::string thick[thksize] = {"h120", "l200", "l300", "h200"};
    int addType[thksize] = {HGCalTypes::WaferFineThin,
                            HGCalTypes::WaferCoarseThin,
                            HGCalTypes::WaferCoarseThick,
                            HGCalTypes::WaferFineThick};
    const int partTypeH[6] = {HGCalTypes::WaferFull,
                              HGCalTypes::WaferHDTop,
                              HGCalTypes::WaferHDBottom,
                              HGCalTypes::WaferHDLeft,
                              HGCalTypes::WaferHDRight,
                              HGCalTypes::WaferHDFive};
    const int partTypeL[7] = {HGCalTypes::WaferFull,
                              HGCalTypes::WaferLDTop,
                              HGCalTypes::WaferLDBottom,
                              HGCalTypes::WaferLDLeft,
                              HGCalTypes::WaferLDRight,
                              HGCalTypes::WaferLDFive,
                              HGCalTypes::WaferLDThree};
    const unsigned int cassetteEE(12), cassetteHE(24);
    std::map<int, wafer> module1, module2, module3;
    unsigned int all(0), comments(0), others(0), bad(0), good(0);
    unsigned int layers(layMax3_);
    std::vector<layerInfo> layer1, layer2, layer3;
    int cminEE(-1), cmaxEE(-1), cminHE1(-1), cmaxHE1(-1), cminHE2(-1), cmaxHE2(-1);
    bool global = (modeGlobal < 1);
    while (fInput.getline(buffer, 1024)) {
      ++all;
      if (debug % 10 > 1)
        std::cout << "[" << all << "] " << buffer << std::endl;
      if (buffer[0] == '#') {
        ++comments;
      } else {
        ++others;
        std::vector<std::string> items = splitString(std::string(buffer));
        if (others <= layMax3_) {
          unsigned int cassettes = (others <= layMax1_) ? cassetteEE : cassetteHE;
          if (items.size() < (cassettes + 2)) {
            ++bad;
          } else {
            int layer = std::atoi(items[0].c_str());
            int type = std::atoi(items[1].c_str());
            std::vector<double> dR;
            for (unsigned int k = 0; k < cassettes; ++k)
              dR.emplace_back(std::atof(items[k + 2].c_str()));
            layerInfo ltype(layer, type, dR);
            if (others <= layMax1_) {
              layer1.emplace_back(ltype);
            } else if ((others <= layMax2_) || global) {
              layer2.emplace_back(ltype);
            } else {
              layer3.emplace_back(ltype);
            }
          }
        } else if (items.size() != 9) {
          ++bad;
        } else {
          ++good;
          unsigned int layer = std::atoi(items[0].c_str());
          int waferU = std::atoi(items[6].c_str());
          int waferV = std::atoi(items[7].c_str());
          int cassette = std::atoi(items[8].c_str()) + 1;  // Start cassette # = 1
          int thck = static_cast<int>(std::find(thick, thick + thksize, items[2]) - thick);
          int part = std::atoi(items[1].c_str());
          if ((thck <= thksize) && (part >= 0)) {
            if ((addType[thck] == HGCalTypes::WaferFineThin) || (addType[thck] == HGCalTypes::WaferFineThick))
              part = partTypeH[part];
            else
              part = partTypeL[part];
          }
          int orient = std::atoi(items[5].c_str());
          wafer waf(thck, part, orient, cassette);
          if (layer <= layMax1_) {
            int index = HGCalWaferIndex::waferIndex(layer, waferU, waferV, false);
            module1[index] = waf;
            if ((cminEE < 0) || (cassette < cminEE))
              cminEE = cassette;
            if ((cmaxEE < 0) || (cassette > cmaxEE))
              cmaxEE = cassette;
          } else if ((layer <= layMax2_) || global) {
            int index = HGCalWaferIndex::waferIndex(layer - layMax1_, waferU, waferV, false);
            module2[index] = waf;
            if ((cminHE1 < 0) || (cassette < cminHE1))
              cminHE1 = cassette;
            if ((cmaxHE1 < 0) || (cassette > cmaxHE1))
              cmaxHE1 = cassette;
          } else {
            int index = HGCalWaferIndex::waferIndex(layer - layMax1_, waferU, waferV, false);
            module3[index] = waf;
            if ((cminHE2 < 0) || (cassette < cminHE2))
              cminHE2 = cassette;
            if ((cmaxHE2 < 0) || (cassette > cmaxHE2))
              cmaxHE2 = cassette;
          }
        }
      }
    }
    fInput.close();
    std::cout << "Read " << all << " records with " << comments << " comments " << others
              << " non-comment records out of which " << good << ":" << module1.size() << ":" << module2.size() << ":"
              << module3.size() << " are good and " << bad << " are bad and with " << layers << " layers\n";
    std::cout << "\nThere are " << layer1.size() << " of types:" << std::endl;
    for (const auto& l : layer1) {
      std::cout << "Layer " << l.layer << " Type " << l.type << " DR";
      for (unsigned int k = 0; k < l.deltaR.size(); ++k)
        std::cout << ": " << l.deltaR[k];
      std::cout << std::endl;
    }
    std::cout << "\nThere are " << layer2.size() << " of types:" << std::endl;
    for (const auto& l : layer2) {
      std::cout << "Layer " << l.layer << " Type " << l.type << " DR";
      for (unsigned int k = 0; k < l.deltaR.size(); ++k)
        std::cout << ": " << l.deltaR[k];
      std::cout << std::endl;
    }
    if (layer3.size() > 0) {
      std::cout << "\nThere are " << layer3.size() << " of types:" << std::endl;
      for (const auto& l : layer3) {
        std::cout << "Layer " << l.layer << " Type " << l.type << " DR";
        for (unsigned int k = 0; k < l.deltaR.size(); ++k)
          std::cout << ": " << l.deltaR[k];
        std::cout << std::endl;
      }
    }
    std::cout << "\nMinimum and Maximum Cassette #'s:: EE: " << cminEE << ":" << cmaxEE << " HEF: " << cminHE1 << ":"
              << cmaxHE1 << " HEB: " << cminHE2 << ":" << cmaxHE2 << std::endl;
    std::cout << std::endl << std::endl;

    //Now write separately for EE, HEsil and HEmix
    writeSilicon(outfile1, cassetteEE, layer1, module1, "EE", global, (debug % 10 > 0));
    // Next HEsil part
    writeSilicon(outfile2, cassetteHE, layer2, module2, "HE", global, ((debug / 10) % 10 > 0));
    // Finally HEmix part
    if (!global)
      writeSilicon(outfile3, cassetteHE, layer3, module3, "HE", global, ((debug / 100) % 10 > 0));
  }
}

void ConvertSiliconV2::writeSilicon(const char* outfile,
                                    const unsigned int cassettes,
                                    const std::vector<layerInfo>& layers,
                                    const std::map<int, wafer>& module,
                                    const std::string& tag,
                                    const bool& mode,
                                    const bool& debug) {
  char apost('"');
  unsigned int k0(0), k1(0), k2(0), k3(0);
  std::map<int, wafer>::const_iterator itr;
  std::string blank = (mode) ? "  " : "    ";
  std::ofstream fOut(outfile);
  std::vector<int> layerStart;
  int layer(-1);
  if (mode) {
    fOut << blank << "<Vector name=" << apost << "LayerTypes" << tag << apost << " type=" << apost << "numeric" << apost
         << " nEntries=" << apost << layers.size() << apost << ">";
  } else {
    fOut << blank << "<Vector name=" << apost << "LayerTypes" << apost << " type=" << apost << "numeric" << apost
         << " nEntries=" << apost << layers.size() << apost << ">";
  }
  for (const auto& l : layers) {
    std::string last = ((k0 + 1) == layers.size()) ? " " : ",";
    if (k0 % 20 == 0)
      fOut << "\n  " << blank << std::setw(2) << l.type << last;
    else
      fOut << std::setw(2) << l.type << last;
    ++k0;
  }
  fOut << "\n" << blank << "</Vector>\n";
  if (mode) {
    fOut << blank << "<Vector name=" << apost << "WaferIndex" << tag << apost << " type=" << apost << "numeric" << apost
         << " nEntries=" << apost << module.size() << apost << ">";
  } else {
    fOut << blank << "<Vector name=" << apost << "WaferIndex" << apost << " type=" << apost << "numeric" << apost
         << " nEntries=" << apost << module.size() << apost << ">";
  }
  for (itr = module.begin(); itr != module.end(); ++itr) {
    std::string last = ((k1 + 1) == module.size()) ? " " : ",";
    if (k1 % 7 == 0)
      fOut << "\n  " << blank << std::setw(8) << itr->first << last;
    else
      fOut << std::setw(8) << itr->first << last;
    if (HGCalWaferIndex::waferLayer(itr->first) != layer) {
      layerStart.emplace_back(k1);
      layer = HGCalWaferIndex::waferLayer(itr->first);
    }
    ++k1;
    if (debug)
      std::cout << "Wafer " << HGCalWaferIndex::waferLayer(itr->first) << ":" << HGCalWaferIndex::waferU(itr->first)
                << ":" << HGCalWaferIndex::waferV(itr->first) << " T " << (itr->second).thick << " P "
                << (itr->second).partial << " O " << (itr->second).orient << " C " << (itr->second).cassette
                << " Property "
                << HGCalProperty::waferProperty(
                       (itr->second).thick, (itr->second).partial, (itr->second).orient, (itr->second).cassette)
                << std::endl;
  }
  fOut << "\n" << blank << "</Vector>\n";
  if (mode)
    fOut << blank << "<Vector name=" << apost << "WaferProperties" << tag << apost << " type=" << apost << "numeric"
         << apost << " nEntries=" << apost << module.size() << apost << ">";
  else
    fOut << blank << "<Vector name=" << apost << "WaferProperties" << apost << " type=" << apost << "numeric" << apost
         << " nEntries=" << apost << module.size() << apost << ">";
  for (itr = module.begin(); itr != module.end(); ++itr) {
    int property = HGCalProperty::waferProperty(
        (itr->second).thick, (itr->second).partial, (itr->second).orient, (itr->second).cassette);
    std::string last = ((k2 + 1) == module.size()) ? " " : ",";
    if (k2 % 8 == 0)
      fOut << "\n  " << blank << std::setw(7) << property << last;
    else
      fOut << std::setw(7) << property << last;
    ++k2;
  }
  fOut << "\n" << blank << "</Vector>\n";
  if (mode) {
    fOut << blank << "<Vector name=" << apost << "WaferLayerStart" << tag << apost << " type=" << apost << "numeric"
         << apost << " nEntries=" << apost << layerStart.size() << apost << ">";
  } else {
    fOut << blank << "<Vector name=" << apost << "WaferLayerStart" << apost << " type=" << apost << "numeric" << apost
         << " nEntries=" << apost << layerStart.size() << apost << ">";
  }
  for (unsigned k = 0; k < layerStart.size(); ++k) {
    std::string last = ((k + 1) == layerStart.size()) ? " " : ",";
    if (k % 10 == 0)
      fOut << "\n  " << blank << std::setw(5) << layerStart[k] << last;
    else
      fOut << std::setw(5) << layerStart[k] << last;
  }
  fOut << "\n" << blank << "</Vector>\n";
  unsigned int csize = cassettes * layers.size();
  if (mode) {
    fOut << blank << "<Vector name=" << apost << "CassetteShift" << tag << apost << " type=" << apost << "numeric"
         << apost << " nEntries=" << apost << csize << apost << ">";
  } else {
    fOut << blank << "<Vector name=" << apost << "CastteShift" << apost << " type=" << apost << "numeric" << apost
         << " nEntries=" << apost << csize << apost << ">";
  }
  for (const auto& l : layers) {
    ++k3;
    for (unsigned int k = 0; k < cassettes; ++k) {
      std::string last = ((k3 == layers.size()) && ((k + 1) == cassettes)) ? " " : ",";
      if ((k % 6) == 0)
        fOut << "\n  " << blank << std::setw(9) << l.deltaR[k] << last;
      else
        fOut << std::setw(9) << l.deltaR[k] << last;
    }
  }
  fOut << "\n" << blank << "</Vector>\n";
  fOut.close();
}

ConvertScintillator::ConvertScintillator(int layMin, int cassette) : layMin_(layMin), cassette_(cassette) {}

void ConvertScintillator::convert(const char* infile, const char* outfile1, const char* outfile2, int debug) {
  std::ifstream fInput(infile);
  if (!fInput.good()) {
    std::cout << "Cannot open file " << infile << std::endl;
  } else {
    //First read in all records
    char buffer[1024];
    std::string types[2] = {"c", "m"};
    std::map<int, tile> module;
    std::map<int, std::pair<double, double> > ringR;
    std::map<int, std::pair<int, int> > layerRing;
    unsigned int all(0), comments(0), others(0), bad(0), good(0);
    int lmin(99), lmax(0);
    while (fInput.getline(buffer, 1024)) {
      ++all;
      if (buffer[0] == '#') {
        ++comments;
      } else {
        ++others;
        std::vector<std::string> items = splitString(std::string(buffer));
        if (items.size() != 10) {
          ++bad;
        } else {
          ++good;
          int layer, ring, hex1, hex2, hex3, hex4;
          float rstart, rend, sipm;
          sscanf(
              buffer, "%d %d %f %f %f %X %X %X %X", &layer, &ring, &rstart, &rend, &sipm, &hex1, &hex2, &hex3, &hex4);
          int type = static_cast<int>(std::find(types, types + 1, items[9]) - types);
          if (layer > layMin_) {
            tile tl(sipm, hex1, hex2, hex3, hex4, type);
            int index = HGCalTileIndex::tileIndex(layer - layMin_, ring + 1, 0);
            module[index] = tl;
            lmin = std::min((layer - layMin_), lmin);
            lmax = std::max((layer - layMin_), lmax);
            ringR[ring] = std::pair<double, double>(rstart, rend);
            if (layerRing.find(layer) == layerRing.end()) {
              layerRing[layer] = std::pair<int, int>(ring, ring);
            } else {
              int rmin = std::min(ring, layerRing[layer].first);
              int rmax = std::max(ring, layerRing[layer].second);
              layerRing[layer] = std::pair<int, int>(rmin, rmax);
            }
          }
        }
      }
    }
    fInput.close();
    std::cout << "Read " << all << " records with " << comments << " comments " << others
              << " non-comment records out of which " << good << ":" << module.size() << ":" << ringR.size() << ":"
              << layerRing.size() << " are good and " << bad << " are bad\n"
              << std::endl;

    //Now write the content in the first file
    std::ofstream fOut(outfile1);
    char apost('"');
    unsigned int l1(0), l2(0);
    std::map<int, std::pair<double, double> >::const_iterator it1;
    fOut << "  <Vector name=" << apost << "TileRMin" << apost << " type=" << apost << "numeric" << apost
         << " nEntries=" << apost << ringR.size() << apost << ">";
    for (it1 = ringR.begin(); it1 != ringR.end(); ++it1) {
      std::string last = ((l1 + 1) == ringR.size()) ? " " : ",";
      if (l1 % 6 == 0)
        fOut << "\n    " << std::setw(8) << std::setprecision(6) << (it1->second).first << "*mm" << last;
      else
        fOut << std::setw(8) << std::setprecision(6) << (it1->second).first << "*mm" << last;
      ++l1;
    }
    fOut << "\n  </Vector>\n";
    fOut << "  <Vector name=" << apost << "TileRMax" << apost << " type=" << apost << "numeric" << apost
         << " nEntries=" << apost << ringR.size() << apost << ">";
    for (it1 = ringR.begin(); it1 != ringR.end(); ++it1) {
      std::string last = ((l2 + 1) == ringR.size()) ? " " : ",";
      if (l2 % 6 == 0)
        fOut << "\n    " << std::setw(8) << std::setprecision(6) << (it1->second).second << "*mm" << last;
      else
        fOut << std::setw(8) << std::setprecision(6) << (it1->second).second << "*mm" << last;
      ++l2;
    }
    fOut << "\n  </Vector>\n";

    unsigned int l3(0), l4(0);
    std::map<int, std::pair<int, int> >::const_iterator it2;
    fOut << "  <Vector name=" << apost << "TileRingMin" << apost << " type=" << apost << "numeric" << apost
         << " nEntries=" << apost << layerRing.size() << apost << ">";
    for (it2 = layerRing.begin(); it2 != layerRing.end(); ++it2) {
      std::string last = ((l3 + 1) == layerRing.size()) ? " " : ",";
      if (l3 % 14 == 0)
        fOut << "\n    " << std::setw(4) << (it2->second).first << last;
      else
        fOut << std::setw(4) << (it2->second).first << last;
      ++l3;
    }
    fOut << "\n  </Vector>\n";
    fOut << "  <Vector name=" << apost << "TileRingMax" << apost << " type=" << apost << "numeric" << apost
         << " nEntries=" << apost << layerRing.size() << apost << ">";
    for (it2 = layerRing.begin(); it2 != layerRing.end(); ++it2) {
      std::string last = ((l4 + 1) == layerRing.size()) ? " " : ",";
      if (l4 % 14 == 0)
        fOut << "\n    " << std::setw(4) << (it2->second).second << last;
      else
        fOut << std::setw(4) << (it2->second).second << last;
      ++l4;
    }
    fOut << "\n  </Vector>\n";

    unsigned int k1(0), k2(0), k3(0), k4(0), k5(0), k6(0);
    std::map<int, tile>::const_iterator itr;
    fOut << "  <Vector name=" << apost << "TileIndex" << apost << " type=" << apost << "numeric" << apost
         << " nEntries=" << apost << module.size() << apost << ">";
    for (itr = module.begin(); itr != module.end(); ++itr) {
      std::string last = ((k1 + 1) == module.size()) ? " " : ",";
      if (k1 % 7 == 0)
        fOut << "\n    " << std::setw(8) << itr->first << last;
      else
        fOut << std::setw(8) << itr->first << last;
      ++k1;
      if ((debug % 10) > 0)
        std::cout << "Tile " << HGCalTileIndex::tileLayer(itr->first) << ":" << HGCalTileIndex::tileRing(itr->first)
                  << " Type " << (itr->second).type << " Area " << (itr->second).sipm << std::hex << " HEX "
                  << (itr->second).hex[0] << " " << (itr->second).hex[1] << " " << (itr->second).hex[2] << " "
                  << (itr->second).hex[3] << std::dec << "\n";
    }
    fOut << "\n  </Vector>\n";
    fOut << "  <Vector name=" << apost << "TileProperty" << apost << " type=" << apost << "numeric" << apost
         << " nEntries=" << apost << module.size() << apost << ">";
    for (itr = module.begin(); itr != module.end(); ++itr) {
      std::string last = ((k2 + 1) == module.size()) ? " " : ",";
      int property = HGCalTileIndex::tileProperty((itr->second).type, (itr->second).sipm);
      if (k2 % 15 == 0)
        fOut << "\n    " << std::setw(3) << property << last;
      else
        fOut << std::setw(3) << property << last;
      ++k2;
    }
    fOut << "\n  </Vector>\n";
    fOut << "  <Vector name=" << apost << "TileHEX1" << apost << " type=" << apost << "numeric" << apost
         << " nEntries=" << apost << module.size() << apost << ">" << std::hex;
    for (itr = module.begin(); itr != module.end(); ++itr) {
      std::string last = ((k3 + 1) == module.size()) ? " " : ",";
      if (k3 % 6 == 0)
        fOut << "\n     0x" << (itr->second).hex[0] << last;
      else
        fOut << " 0x" << (itr->second).hex[0] << last;
      ++k3;
    }
    fOut << "\n  </Vector>\n" << std::dec;
    fOut << "  <Vector name=" << apost << "TileHEX2" << apost << " type=" << apost << "numeric" << apost
         << " nEntries=" << apost << module.size() << apost << ">" << std::hex;
    for (itr = module.begin(); itr != module.end(); ++itr) {
      std::string last = ((k4 + 1) == module.size()) ? " " : ",";
      if (k4 % 6 == 0)
        fOut << "\n     0x" << (itr->second).hex[1] << last;
      else
        fOut << " 0x" << (itr->second).hex[1] << last;
      ++k4;
    }
    fOut << "\n  </Vector>\n" << std::dec;
    fOut << "  <Vector name=" << apost << "TileHEX3" << apost << " type=" << apost << "numeric" << apost
         << " nEntries=" << apost << module.size() << apost << ">" << std::hex;
    for (itr = module.begin(); itr != module.end(); ++itr) {
      std::string last = ((k5 + 1) == module.size()) ? " " : ",";
      if (k5 % 6 == 0)
        fOut << "\n     0x" << (itr->second).hex[2] << last;
      else
        fOut << " 0x" << (itr->second).hex[2] << last;
      ++k5;
    }
    fOut << "\n  </Vector>\n" << std::dec;
    fOut << "  <Vector name=" << apost << "TileHEX4" << apost << " type=" << apost << "numeric" << apost
         << " nEntries=" << apost << module.size() << apost << ">" << std::hex;
    for (itr = module.begin(); itr != module.end(); ++itr) {
      std::string last = ((k6 + 1) == module.size()) ? " " : ",";
      if (k6 % 6 == 0)
        fOut << "\n     0x" << (itr->second).hex[3] << last;
      else
        fOut << " 0x" << (itr->second).hex[3] << last;
      ++k6;
    }
    fOut << "\n  </Vector>\n" << std::dec;
    fOut.close();

    //Now write for the second file
    makeTitle(outfile2, module, ringR, lmin, lmax, (((debug / 10) % 10) > 0));
  }
}

void ConvertScintillator::makeTitle(const char* outfile,
                                    const std::map<int, tile>& module,
                                    const std::map<int, std::pair<double, double> >& ringR,
                                    int lmin,
                                    int lmax,
                                    bool debug) {
  const int zside = 1;
  const int phiCassette = 24;
  std::vector<tileZone> zones;
  for (int layer = lmin; layer <= lmax; ++layer) {
    tileZone tile0;
    int kk, irmin, irmax;
    for (int phi = 1; phi <= HGCalProperty::kHGCalTilePhis; ++phi) {
      kk = irmin = irmax = 0;
      for (std::map<int, tile>::const_iterator itr = module.begin(); itr != module.end(); ++itr) {
        if ((HGCalTileIndex::tileLayer(itr->first) == layer) &&
            HGCalTileIndex::tileExist((itr->second).hex, zside, phi)) {
          int ir = HGCalTileIndex::tileRing(itr->first);
          if (kk == 0) {
            irmin = irmax = ir;
          } else {
            irmax = ir;
          }
          ++kk;
        }
      }
      if (debug)
        std::cout << "Layer|Phi|Ring " << layer << ":" << phi << ":" << irmin << ":" << irmax << std::endl;
      if (phi == 1) {
        tile0.layer = layer;
        tile0.rmin = irmin;
        tile0.rmax = irmax;
        tile0.phimin = phi;
        tile0.phimax = phi;
        tile0.cassette = (cassette_ == 0) ? 0 : 1;
      } else if ((tile0.rmin != irmin) || (tile0.rmax != irmax)) {
        if (cassette_ != 0) {
          if (tile0.cassette * phiCassette < tile0.phimax) {
            do {
              int phimax = tile0.phimax;
              tile0.phimax = tile0.cassette * phiCassette;
              zones.push_back(tile0);
              tile0.phimin = tile0.phimax + 1;
              tile0.phimax = phimax;
              ++tile0.cassette;
            } while (tile0.cassette * phiCassette < tile0.phimax);
          }
        }
        zones.push_back(tile0);
        int cassette = (cassette_ == 0) ? 0 : (1 + ((phi - 1) / phiCassette));
        tile0.layer = layer;
        tile0.rmin = irmin;
        tile0.rmax = irmax;
        tile0.phimin = phi;
        tile0.phimax = phi;
        tile0.cassette = cassette;
        if (phi == HGCalProperty::kHGCalTilePhis)
          zones.push_back(tile0);
      } else {
        tile0.phimax = phi;
        if (phi == HGCalProperty::kHGCalTilePhis)
          zones.push_back(tile0);
      }
    }
  }

  int nmax = zones.size();
  if (debug) {
    std::cout << "\nA total of " << nmax << " zones " << std::endl;
    for (int k = 0; k < nmax; ++k)
      std::cout << "[" << k << "] Layer " << zones[k].layer << " Ring " << zones[k].rmin << ":" << zones[k].rmax
                << " phi " << zones[k].phimin << ":" << zones[k].phimax << " Cassette " << zones[k].cassette
                << std::endl;
  }
  if (nmax > 0) {
    std::ofstream fout(outfile);
    char apost('"');
    unsigned int l1(0), l2(0);
    std::map<int, std::pair<double, double> >::const_iterator it1;
    fout << "  <Vector name=" << apost << "TileRMin" << apost << " type=" << apost << "numeric" << apost
         << " nEntries=" << apost << ringR.size() << apost << ">";
    for (it1 = ringR.begin(); it1 != ringR.end(); ++it1) {
      std::string last = ((l1 + 1) == ringR.size()) ? " " : ",";
      if (l1 % 6 == 0)
        fout << "\n    " << std::setw(8) << std::setprecision(6) << (it1->second).first << "*mm" << last;
      else
        fout << std::setw(8) << std::setprecision(6) << (it1->second).first << "*mm" << last;
      ++l1;
    }
    fout << "\n  </Vector>\n";
    fout << "  <Vector name=" << apost << "TileRMax" << apost << " type=" << apost << "numeric" << apost
         << " nEntries=" << apost << ringR.size() << apost << ">";
    for (it1 = ringR.begin(); it1 != ringR.end(); ++it1) {
      std::string last = ((l2 + 1) == ringR.size()) ? " " : ",";
      if (l2 % 6 == 0)
        fout << "\n    " << std::setw(8) << std::setprecision(6) << (it1->second).second << "*mm" << last;
      else
        fout << std::setw(8) << std::setprecision(6) << (it1->second).second << "*mm" << last;
      ++l2;
    }
    fout << "\n  </Vector>\n";
    fout << "  <Vector name=" << apost << "TileLayerRings" << apost << " type=" << apost << "numeric" << apost
         << " nEntries=" << apost << nmax << apost << ">";
    if (debug)
      std::cout << "  <Vector name=" << apost << "TileLayerRings" << apost << " type=" << apost << "numeric" << apost
                << " nEntries=" << apost << nmax << apost << ">";
    for (int k = 0; k < nmax; ++k) {
      std::string last = ((k + 1) == nmax) ? " " : ",";
      int lyr1r2 = HGCalTileIndex::tilePack(zones[k].layer, zones[k].rmin, zones[k].rmax);
      if (k % 7 == 0) {
        fout << "\n    " << std::setw(9) << lyr1r2 << last;
        if (debug)
          std::cout << "\n    " << std::setw(9) << lyr1r2 << last;
      } else {
        fout << std::setw(9) << lyr1r2 << last;
        if (debug)
          std::cout << std::setw(9) << lyr1r2 << last;
      }
    }
    fout << "\n  </Vector>\n";
    if (debug)
      std::cout << "\n  </Vector>\n";
    int layer = -1;
    std::vector<int> layerStart;
    fout << "  <Vector name=" << apost << "TilePhiRange" << apost << " type=" << apost << "numeric" << apost
         << " nEntries=" << apost << nmax << apost << ">";
    if (debug)
      std::cout << "  <Vector name=" << apost << "TilePhiRange" << apost << " type=" << apost << "numeric" << apost
                << " nEntries=" << apost << nmax << apost << ">";
    for (int k = 0; k < nmax; ++k) {
      std::string last = ((k + 1) == nmax) ? " " : ",";
      int f1f2 = HGCalTileIndex::tilePack(zones[k].cassette, zones[k].phimin, zones[k].phimax);
      if (k % 7 == 0) {
        fout << "\n    " << std::setw(9) << f1f2 << last;
        if (debug)
          std::cout << "\n    " << std::setw(9) << f1f2 << last;
      } else {
        fout << std::setw(9) << f1f2 << last;
        if (debug)
          std::cout << std::setw(9) << f1f2 << last;
      }
      if (zones[k].layer != layer) {
        layerStart.emplace_back(k);
        layer = zones[k].layer;
      }
    }
    fout << "\n  </Vector>\n";
    if (debug)
      std::cout << "\n  </Vector>\n";
    fout << "  <Vector name=" << apost << "TileLayerStart" << apost << " type=" << apost << "numeric" << apost
         << " nEntries=" << apost << layerStart.size() << apost << ">";
    if (debug)
      std::cout << "  <Vector name=" << apost << "TileLayerStart" << apost << " type=" << apost << "numeric" << apost
                << " nEntries=" << apost << layerStart.size() << apost << ">";
    for (unsigned int k = 0; k < layerStart.size(); ++k) {
      std::string last = ((k + 1) == layerStart.size()) ? " " : ",";
      if (k % 10 == 0) {
        fout << "\n    " << std::setw(5) << layerStart[k] << last;
        if (debug)
          std::cout << "\n    " << std::setw(5) << layerStart[k] << last;
      } else {
        fout << std::setw(5) << layerStart[k] << last;
        if (debug)
          std::cout << std::setw(5) << layerStart[k] << last;
      }
    }
    fout << "\n  </Vector>\n";
    if (debug)
      std::cout << "\n  </Vector>\n";
    fout.close();
  }
}
