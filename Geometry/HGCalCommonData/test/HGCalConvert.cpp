//////////////////////////////////////////////////////////////////////////////
//
//  Contains 2 macros to extract from the ASCII files from Philippe, Chris
//  and Katya to make part of the xml file to be used to define the
//  geometry of the HGCal detector
//
//  HGCalConvert 1 infile outfile1 outfile2 outfile3 modeGlobal debug
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
//  HGCalConvert 2 infile outfile debug
//  infile   (std::string)   Input file from Katya (modified by Chris)
//                           containing layer #. ring #, start and end
//                           of ring radius, SiPM size, 4 hexadecimal
//                           words specifying if the phi location in
//                           120-degree sector is filled, scintillator
//                           type (c: cast, m: mould)
//  outfile  (std::string)   Output file
//  debug    (int)           Flag to switch on (1) or off (0) debug
//
//////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

const int kHGCalWaferUOffset = 0;
const int kHGCalWaferUMask = 0x1F;
const int kHGCalWaferUSignOffset = 5;
const int kHGCalWaferUSignMask = 0x1;
const int kHGCalWaferVOffset = 6;
const int kHGCalWaferVMask = 0x1F;
const int kHGCalWaferVSignOffset = 11;
const int kHGCalWaferVSignMask = 0x1;
const int kHGCalLayerOffset = 18;
const int kHGCalLayerMask = 0x1F;
const int kHGCalPhiOffset = 0;
const int kHGCalPhiMask = 0x1FF;
const int kHGCalRingOffset = 9;
const int kHGCalRingMask = 0x1FF;

struct wafer {
  int thick, partial, orient;
  wafer(int t = 0, int p = 0, int o = 0) : thick(t), partial(p), orient(o){};
};

struct tile {
  double sipm;
  int hex1, hex2, hex3, hex4, type;
  tile(double s = 0, int h1 = 0, int h2 = 0, int h3 = 0, int h4 = 0, int t = 0)
      : sipm(s), hex1(h1), hex2(h2), hex3(h3), hex4(h4), type(t){};
};

std::vector<std::string> splitString(const std::string& fLine);

class ConvertSilicon {
public:
  ConvertSilicon(int layMax1 = 28, int layMax2 = 36);
  void convert(const char* infile,
               const char* outfile1,
               const char* outfile2,
               const char* outfile3,
               int modeGlobal = 0,
               int debug = 0);

private:
  void writeSilicon(const char*, const std::map<int, wafer>&, const std::string&, const bool&, const bool&);
  int waferIndex(const int&, const int&, const int&);
  int waferLayer(const int&);
  int waferU(const int&);
  int waferV(const int&);

  const int layMax1_, layMax2_;
};

class ConvertScintillator {
public:
  ConvertScintillator(int layMin = 28);
  void convert(const char*, const char*, int debug = 0);

private:
  int tileIndex(const int&, const int&, const int&);
  int tileLayer(const int&);
  int tileRing(const int&);
  int tilePhi(const int&);

  const int layMin_;
};

int main(int argc, char* argv[]) {
  if (argc < 4) {
    std::cout << "Please give a minimum of 7/4 arguments \n"
              << "mode (1:silicon; 2:scintillator)\n"
              << "input file name\n"
              << "output file name\n"
              << "for silicon 4 additional parameters:\n"
              << "  second output file name\n"
              << "  third output file name\n"
              << "  output mode (0: gobal; 1: local)\n"
              << "  debug (three digis to set debug for each output)\n"
              << "for scintillator 1 additional parameter after the first 3\n"
              << "  debug flag\n"
              << std::endl;
    return 0;
  }

  int mode = std::atoi(argv[1]);
  if (mode < 2 && argc < 7) {
    std::cout << "Please give a minimum of 7 arguments \n"
              << "mode == 1\n"
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
  if (mode < 2) {
    ConvertSilicon c1;
    const char* outfile1 = argv[3];
    const char* outfile2 = argv[4];
    const char* outfile3 = argv[5];
    int modeGlobal = atoi(argv[6]);
    int debug = atoi(argv[7]);
    c1.convert(infile, outfile1, outfile2, outfile3, modeGlobal, debug);
  } else {
    ConvertScintillator c1;
    const char* outfile = argv[3];
    int debug = atoi(argv[4]);
    c1.convert(infile, outfile, debug);
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

ConvertSilicon::ConvertSilicon(int layMax1, int layMax2) : layMax1_(layMax1), layMax2_(layMax2) {}

void ConvertSilicon::convert(
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
          wafer waf(thck, part, orient);
          if (layer <= layMax1_) {
            int index = waferIndex(layer, waferU, waferV);
            module1[index] = waf;
          } else if ((layer <= layMax2_) || global) {
            int index = waferIndex(layer - layMax1_, waferU, waferV);
            module2[index] = waf;
          } else {
            int index = waferIndex(layer - layMax1_, waferU, waferV);
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
    writeSilicon(outfile1, module1, "EE", global, (debug % 10 == 1));
    // Next HEsil part
    writeSilicon(outfile2, module2, "HE", global, ((debug / 10) % 10 == 1));
    // Finally HEmix part
    if (!global)
      writeSilicon(outfile3, module3, "HE", global, ((debug / 100) % 10 == 1));
  }
}

void ConvertSilicon::writeSilicon(const char* outfile,
                                  const std::map<int, wafer>& module,
                                  const std::string& tag,
                                  const bool& mode,
                                  const bool& debug) {
  char apost('"');
  unsigned int k1(0), k2(0), k3(0), k4(0);
  std::map<int, wafer>::const_iterator itr;
  std::string blank("  ");
  std::ofstream fOut(outfile);
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
    if (k1 % 7 == 0)
      fOut << "\n  " << blank << std::setw(8) << itr->first << ",";
    else
      fOut << std::setw(8) << itr->first << ",";
    ++k1;
    if (debug)
      std::cout << "Wafer " << waferLayer(itr->first) << ":" << waferU(itr->first) << ":" << waferV(itr->first) << " T "
                << (itr->second).thick << " P " << (itr->second).partial << " O " << (itr->second).orient << std::endl;
  }
  fOut << "\n" << blank << "</Vector>\n";
  if (mode)
    fOut << blank << "<Vector name=" << apost << "WaferTypes" << tag << apost << " type=" << apost << "numeric" << apost
         << " nEntries=" << apost << module.size() << apost << ">";
  else
    fOut << blank << "<Vector name=" << apost << "WaferTypes" << apost << " type=" << apost << "numeric" << apost
         << " nEntries=" << apost << module.size() << apost << ">";
  for (itr = module.begin(); itr != module.end(); ++itr) {
    if (k2 % 20 == 0)
      fOut << "\n  " << blank << std::setw(2) << (itr->second).thick << ",";
    else
      fOut << std::setw(2) << (itr->second).thick << ",";
    ++k2;
  }
  fOut << "\n" << blank << "</Vector>\n";
  if (mode) {
    fOut << blank << "<Vector name=" << apost << "WaferPartial" << tag << apost << " type=" << apost << "numeric"
         << apost << " nEntries=" << apost << module.size() << apost << ">";
    for (itr = module.begin(); itr != module.end(); ++itr) {
      if (k3 % 20 == 0)
        fOut << "\n  " << blank << std::setw(2) << (itr->second).partial << ",";
      else
        fOut << std::setw(2) << (itr->second).partial << ",";
      ++k3;
    }
    fOut << "\n" << blank << "</Vector>\n";
    fOut << blank << "<Vector name=" << apost << "WaferOrient" << tag << apost << " type=" << apost << "numeric"
         << apost << " nEntries=" << apost << module.size() << apost << ">";
    for (itr = module.begin(); itr != module.end(); ++itr) {
      if (k4 % 20 == 0)
        fOut << "\n  " << blank << std::setw(2) << (itr->second).orient << ",";
      else
        fOut << std::setw(2) << (itr->second).orient << ",";
      ++k4;
    }
    fOut << "\n" << blank << "</Vector>\n";
  }
  fOut.close();
}

int ConvertSilicon::waferIndex(const int& layer, const int& waferU, const int& waferV) {
  int id(0);
  int waferUabs(std::abs(waferU)), waferVabs(std::abs(waferV));
  int waferUsign = (waferU >= 0) ? 0 : 1;
  int waferVsign = (waferV >= 0) ? 0 : 1;
  id |= (((waferUabs & kHGCalWaferUMask) << kHGCalWaferUOffset) |
         ((waferUsign & kHGCalWaferUSignMask) << kHGCalWaferUSignOffset) |
         ((waferVabs & kHGCalWaferVMask) << kHGCalWaferVOffset) |
         ((waferVsign & kHGCalWaferVSignMask) << kHGCalWaferVSignOffset) |
         ((layer & kHGCalLayerMask) << kHGCalLayerOffset));
  return id;
}

int ConvertSilicon::waferLayer(const int& id) { return (id >> kHGCalLayerOffset) & kHGCalLayerMask; }

int ConvertSilicon::waferU(const int& id) {
  int32_t iu = (id >> kHGCalWaferUOffset) & kHGCalWaferUMask;
  return (((id >> kHGCalWaferUSignOffset) & kHGCalWaferUSignMask) ? -iu : iu);
}

int ConvertSilicon::waferV(const int& id) {
  int32_t iv = (id >> kHGCalWaferVOffset) & kHGCalWaferVMask;
  return (((id >> kHGCalWaferVSignOffset) & kHGCalWaferVSignMask) ? -iv : iv);
}

ConvertScintillator::ConvertScintillator(int layMin) : layMin_(layMin) {}

void ConvertScintillator::convert(const char* infile, const char* outfile, int debug) {
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
            int index = tileIndex(layer - layMin_, ring + 1, 0);
            module[index] = tl;
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

    //Now write the content
    std::ofstream fOut(outfile);
    char apost('"');
    unsigned int l1(0), l2(0);
    std::map<int, std::pair<double, double> >::const_iterator it1;
    fOut << "  <Vector name=" << apost << "TileRMin" << apost << " type=" << apost << "numeric" << apost
         << " nEntries=" << apost << ringR.size() << apost << ">";
    for (it1 = ringR.begin(); it1 != ringR.end(); ++it1) {
      if (l1 % 6 == 0)
        fOut << "\n    " << std::setw(8) << std::setprecision(6) << (it1->second).first << "*mm,";
      else
        fOut << std::setw(8) << std::setprecision(6) << (it1->second).first << "*mm,";
      ++l1;
    }
    fOut << "\n  </Vector>\n";
    fOut << "  <Vector name=" << apost << "TileRMax" << apost << " type=" << apost << "numeric" << apost
         << " nEntries=" << apost << ringR.size() << apost << ">";
    for (it1 = ringR.begin(); it1 != ringR.end(); ++it1) {
      if (l2 % 6 == 0)
        fOut << "\n    " << std::setw(8) << std::setprecision(6) << (it1->second).second << "*mm,";
      else
        fOut << std::setw(8) << std::setprecision(6) << (it1->second).second << "*mm,";
      ++l2;
    }
    fOut << "\n  </Vector>\n";

    unsigned int l3(0), l4(0);
    std::map<int, std::pair<int, int> >::const_iterator it2;
    fOut << "  <Vector name=" << apost << "TileRingMin" << apost << " type=" << apost << "numeric" << apost
         << " nEntries=" << apost << layerRing.size() << apost << ">";
    for (it2 = layerRing.begin(); it2 != layerRing.end(); ++it2) {
      if (l3 % 14 == 0)
        fOut << "\n    " << std::setw(4) << (it2->second).first << ",";
      else
        fOut << std::setw(4) << (it2->second).first << ",";
      ++l3;
    }
    fOut << "\n  </Vector>\n";
    fOut << "  <Vector name=" << apost << "TileRingMax" << apost << " type=" << apost << "numeric" << apost
         << " nEntries=" << apost << layerRing.size() << apost << ">";
    for (it2 = layerRing.begin(); it2 != layerRing.end(); ++it2) {
      if (l4 % 14 == 0)
        fOut << "\n    " << std::setw(4) << (it2->second).second << ",";
      else
        fOut << std::setw(4) << (it2->second).second << ",";
      ++l4;
    }
    fOut << "\n  </Vector>\n";

    unsigned int k1(0), k2(0), k3(0), k4(0), k5(0), k6(0), k7(0);
    std::map<int, tile>::const_iterator itr;
    fOut << "  <Vector name=" << apost << "TileIndex" << apost << " type=" << apost << "numeric" << apost
         << " nEntries=" << apost << module.size() << apost << ">";
    for (itr = module.begin(); itr != module.end(); ++itr) {
      if (k1 % 7 == 0)
        fOut << "\n    " << std::setw(8) << itr->first << ",";
      else
        fOut << std::setw(8) << itr->first << ",";
      ++k1;
      if (debug > 0)
        std::cout << "Tile " << tileLayer(itr->first) << ":" << tileRing(itr->first) << " Type " << (itr->second).type
                  << " Area " << (itr->second).sipm << std::hex << " HEX " << (itr->second).hex1 << " "
                  << (itr->second).hex2 << " " << (itr->second).hex3 << " " << (itr->second).hex4 << std::dec << "\n";
    }
    fOut << "\n  </Vector>\n";
    fOut << "  <Vector name=" << apost << "TileType" << apost << " type=" << apost << "numeric" << apost
         << " nEntries=" << apost << module.size() << apost << ">";
    for (itr = module.begin(); itr != module.end(); ++itr) {
      if (k2 % 20 == 0)
        fOut << "\n    " << std::setw(2) << (itr->second).type << ",";
      else
        fOut << std::setw(2) << (itr->second).type << ",";
      ++k2;
    }
    fOut << "\n  </Vector>\n";
    fOut << "  <Vector name=" << apost << "TileSiPM" << apost << " type=" << apost << "numeric" << apost
         << " nEntries=" << apost << module.size() << apost << ">";
    for (itr = module.begin(); itr != module.end(); ++itr) {
      if (k3 % 20 == 0)
        fOut << "\n    " << std::setw(2) << (itr->second).sipm << ",";
      else
        fOut << std::setw(2) << (itr->second).sipm << ",";
      ++k3;
    }
    fOut << "\n  </Vector>\n";
    fOut << "  <Vector name=" << apost << "TileHEX1" << apost << " type=" << apost << "numeric" << apost
         << " nEntries=" << apost << module.size() << apost << ">" << std::hex;
    for (itr = module.begin(); itr != module.end(); ++itr) {
      if (k4 % 6 == 0)
        fOut << "\n     0x" << (itr->second).hex1 << ",";
      else
        fOut << " 0x" << (itr->second).hex1 << ",";
      ++k4;
    }
    fOut << "\n  </Vector>\n" << std::dec;
    fOut << "  <Vector name=" << apost << "TileHEX2" << apost << " type=" << apost << "numeric" << apost
         << " nEntries=" << apost << module.size() << apost << ">" << std::hex;
    for (itr = module.begin(); itr != module.end(); ++itr) {
      if (k5 % 6 == 0)
        fOut << "\n     0x" << (itr->second).hex2 << ",";
      else
        fOut << " 0x" << (itr->second).hex2 << ",";
      ++k5;
    }
    fOut << "\n  </Vector>\n" << std::dec;
    fOut << "  <Vector name=" << apost << "TileHEX3" << apost << " type=" << apost << "numeric" << apost
         << " nEntries=" << apost << module.size() << apost << ">" << std::hex;
    for (itr = module.begin(); itr != module.end(); ++itr) {
      if (k6 % 6 == 0)
        fOut << "\n     0x" << (itr->second).hex3 << ",";
      else
        fOut << " 0x" << (itr->second).hex3 << ",";
      ++k6;
    }
    fOut << "\n  </Vector>\n" << std::dec;
    fOut << "  <Vector name=" << apost << "TileHEX4" << apost << " type=" << apost << "numeric" << apost
         << " nEntries=" << apost << module.size() << apost << ">" << std::hex;
    for (itr = module.begin(); itr != module.end(); ++itr) {
      if (k7 % 6 == 0)
        fOut << std::setw(6) << "\n     0x" << (itr->second).hex4 << ",";
      else
        fOut << " 0x" << (itr->second).hex4 << ",";
      ++k7;
    }
    fOut << "\n  </Vector>\n" << std::dec;
    fOut.close();
  }
}

int ConvertScintillator::tileIndex(const int& layer, const int& ring, const int& iphi) {
  int id(0);
  id |= (((iphi & kHGCalPhiMask) << kHGCalPhiOffset) | ((ring & kHGCalRingMask) << kHGCalRingOffset) |
         ((layer & kHGCalLayerMask) << kHGCalLayerOffset));
  return id;
}

int ConvertScintillator::tileLayer(const int& id) { return (id >> kHGCalLayerOffset) & kHGCalLayerMask; }

int ConvertScintillator::tileRing(const int& id) { return (id >> kHGCalRingOffset) & kHGCalRingMask; }

int ConvertScintillator::tilePhi(const int& id) { return (id >> kHGCalPhiOffset) & kHGCalPhiMask; }
