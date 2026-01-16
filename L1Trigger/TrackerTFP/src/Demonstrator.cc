#include "L1Trigger/TrackerTFP/interface/Demonstrator.h"

#include <vector>
#include <set>
#include <string>
#include <sstream>
#include <fstream>
#include <numeric>

namespace trackerTFP {

  Demonstrator::Demonstrator(const Config& iConfig, const tt::Setup* setup)
      : dirIPBB_(iConfig.dirIPBB_),
        runTime_(iConfig.runTime_),
        linkMappingIn_(iConfig.linkMappingIn_),
        linkMappingOut_(iConfig.linkMappingOut_),
        dirIn_(dirIPBB_ + "in.txt"),
        dirOut_(dirIPBB_ + "out.txt"),
        dirPre_(dirIPBB_ + "pre.txt"),
        dirDiff_(dirIPBB_ + "diff.txt"),
        numFrames_(setup->numFramesIOHigh()),
        numFramesInfra_(setup->numFramesInfra()),
        numRegions_(setup->numRegions()) {}

  // plays input through modelsim and compares result with output
  bool Demonstrator::analyze(const std::vector<std::vector<tt::Frame>>& input,
                             const std::vector<std::vector<tt::Frame>>& output) const {
    // default link mapping
    auto linkMapping =
        [this](const std::vector<int>& mapC, std::vector<int>& map, const std::vector<std::vector<tt::Frame>>& data) {
          if (mapC.empty()) {
            map.resize(data.size() / numRegions_);
            std::iota(map.begin(), map.end(), 0);
          } else
            map = mapC;
        };
    // converts input into stringstream
    std::stringstream ss;
    std::vector<int> map;
    linkMapping(linkMappingIn_, map, input);
    convert(input, ss, map);
    // play input through modelsim
    sim(ss);
    // converts output into stringstream
    map.clear();
    linkMapping(linkMappingOut_, map, output);
    convert(output, ss, map);
    // compares output with modelsim output
    return compare(ss);
  }

  // converts streams of bv into stringstream
  void Demonstrator::convert(const std::vector<std::vector<tt::Frame>>& bits,
                             std::stringstream& ss,
                             const std::vector<int>& mapping) const {
    // reset ss
    ss.str("");
    ss.clear();
    // number of tranceiver in a quad
    static constexpr int quad = 4;
    std::set<int> quads;
    for (int channel : mapping)
      quads.insert(channel / quad);
    std::vector<int> links;
    links.reserve(quads.size() * quad);
    for (int q : quads) {
      const int offset = q * quad;
      for (int c = 0; c < quad; c++)
        links.push_back(offset + c);
    }
    // start with header
    ss << header(links);
    int nFrame(0);
    // create one packet per region
    bool first = true;
    for (int region = 0; region < numRegions_; region++) {
      const int offset = region * mapping.size();
      // start with emp 6 frame gap
      ss << infraGap(nFrame, links.size());
      for (int frame = 0; frame < numFrames_; frame++) {
        // write one frame for all channel
        ss << this->frame(nFrame);
        for (int link : links) {
          const auto channel = find(mapping.begin(), mapping.end(), link);
          if (channel == mapping.end())
            ss << "  0000 " << std::string(TTBV::S_ / 4, '0');
          else {
            const std::vector<tt::Frame>& bvs = bits[offset + std::distance(mapping.begin(), channel)];
            ss << (frame < static_cast<int>(bvs.size()) ? hex(bvs[frame], first) : hex(tt::Frame(), first));
          }
        }
        ss << std::endl;
        first = false;
      }
    }
  }

  // plays stringstream through modelsim
  void Demonstrator::sim(const std::stringstream& ss) const {
    // write ss to disk
    std::fstream fs;
    fs.open(dirIn_.c_str(), std::fstream::out);
    fs << ss.rdbuf();
    fs.close();
    // run modelsim
    std::stringstream cmd;
    cmd << "cd " << dirIPBB_ << " && ./run_sim -quiet -c work.top -do 'run " << runTime_
        << "us' -do 'quit' &> /dev/null";
    std::system(cmd.str().c_str());
  }

  // compares stringstream with modelsim output
  bool Demonstrator::compare(std::stringstream& ss) const {
    // write ss to disk
    std::fstream fs;
    fs.open(dirPre_.c_str(), std::fstream::out);
    fs << ss.rdbuf();
    fs.close();
    // use linux diff on disk
    const std::string c = "diff " + dirPre_ + " " + dirOut_ + " &> " + dirDiff_;
    std::system(c.c_str());
    ss.str("");
    ss.clear();
    // read diff output
    fs.open(dirDiff_.c_str(), std::fstream::in);
    ss << fs.rdbuf();
    fs.close();
    // count lines, 4 are expected
    int n(0);
    std::string token;
    while (getline(ss, token))
      n++;
    return n == 4;
  }

  // creates emp file header
  std::string Demonstrator::header(const std::vector<int>& links) const {
    std::stringstream ss;
    // file header
    ss << "Id: CMSSW" << std::endl
       << "Metadata: (strobe,) start of orbit, start of packet, end of packet, valid" << std::endl
       << std::endl;
    // link header
    ss << "      Link  ";
    for (int link : links)
      ss << "            " << std::setfill('0') << std::setw(3) << link << "        ";
    ss << std::endl;
    return ss.str();
  }

  // creates 6 frame gap between packets
  std::string Demonstrator::infraGap(int& nFrame, int numLinks) const {
    std::stringstream ss;
    for (int gap = 0; gap < numFramesInfra_; gap++) {
      ss << frame(nFrame);
      for (int link = 0; link < numLinks; link++)
        ss << "  0000 " << std::string(TTBV::S_ / 4, '0');
      ss << std::endl;
    }
    return ss.str();
  }

  // creates frame number
  std::string Demonstrator::frame(int& nFrame) const {
    std::stringstream ss;
    ss << "Frame " << std::setfill('0') << std::setw(4) << nFrame++ << "  ";
    return ss.str();
  }

  // converts bv into hex
  std::string Demonstrator::hex(const tt::Frame& bv, bool first) const {
    std::stringstream ss;
    ss << (first ? "  1001 " : "  0001 ") << std::setfill('0') << std::setw(TTBV::S_ / 4) << std::hex << bv.to_ullong();
    return ss.str();
  }

}  // namespace trackerTFP
