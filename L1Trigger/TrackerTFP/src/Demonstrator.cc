#include "L1Trigger/TrackerTFP/interface/Demonstrator.h"

#include <vector>
#include <string>
#include <sstream>
#include <fstream>

using namespace std;
using namespace edm;
using namespace tt;

namespace trackerTFP {

  Demonstrator::Demonstrator(const ParameterSet& iConfig, const Setup* setup)
      : dirIPBB_(iConfig.getParameter<string>("DirIPBB")),
        runTime_(iConfig.getParameter<double>("RunTime")),
        dirIn_(dirIPBB_ + "in.txt"),
        dirOut_(dirIPBB_ + "out.txt"),
        dirPre_(dirIPBB_ + "pre.txt"),
        dirDiff_(dirIPBB_ + "diff.txt"),
        numFrames_(setup->numFramesIO()),
        numFramesInfra_(setup->numFramesInfra()),
        numRegions_(setup->numRegions()) {}

  // plays input through modelsim and compares result with output
  bool Demonstrator::analyze(const vector<vector<Frame>>& input, const vector<vector<Frame>>& output) const {
    stringstream ss;
    // converts input into stringstream
    convert(input, ss);
    // play input through modelsim
    sim(ss);
    // converts output into stringstream
    convert(output, ss);
    // compares output with modelsim output
    return compare(ss);
  }

  // converts streams of bv into stringstream
  void Demonstrator::convert(const vector<vector<Frame>>& bits, stringstream& ss) const {
    // reset ss
    ss.str("");
    ss.clear();
    // number of tranceiver in a quad
    static constexpr int quad = 4;
    const int numChannel = bits.size() / numRegions_;
    const int voidChannel = numChannel % quad == 0 ? 0 : quad - numChannel % quad;
    // start with header
    ss << header(numChannel + voidChannel);
    int nFrame(0);
    // create one packet per region
    for (int region = 0; region < numRegions_; region++) {
      const int offset = region * numChannel;
      // start with emp 6 frame gap
      ss << infraGap(nFrame, numChannel + voidChannel);
      for (int frame = 0; frame < numFrames_; frame++) {
        // write one frame for all channel
        ss << this->frame(nFrame);
        for (int channel = 0; channel < numChannel; channel++) {
          const vector<Frame>& bvs = bits[offset + channel];
          ss << (frame < (int)bvs.size() ? hex(bvs[frame]) : hex(Frame()));
        }
        for (int channel = 0; channel < voidChannel; channel++)
          ss << " 0v" << string(TTBV::S_ / 4, '0');
        ss << endl;
      }
    }
  }

  // plays stringstream through modelsim
  void Demonstrator::sim(const stringstream& ss) const {
    // write ss to disk
    fstream fs;
    fs.open(dirIn_.c_str(), fstream::out);
    fs << ss.rdbuf();
    fs.close();
    // run modelsim
    stringstream cmd;
    cmd << "cd " << dirIPBB_ << " && ./run_sim -quiet -c work.top -do 'run " << runTime_
        << "us' -do 'quit' &> /dev/null";
    system(cmd.str().c_str());
  }

  // compares stringstream with modelsim output
  bool Demonstrator::compare(stringstream& ss) const {
    // write ss to disk
    fstream fs;
    fs.open(dirPre_.c_str(), fstream::out);
    fs << ss.rdbuf();
    fs.close();
    // use linux diff on disk
    const string c = "diff " + dirPre_ + " " + dirOut_ + " &> " + dirDiff_;
    system(c.c_str());
    ss.str("");
    ss.clear();
    // read diff output
    fs.open(dirDiff_.c_str(), fstream::in);
    ss << fs.rdbuf();
    fs.close();
    // count lines, 4 are expected
    int n(0);
    string token;
    while (getline(ss, token))
      n++;
    return n == 4;
  }

  // creates emp file header
  string Demonstrator::header(int numLinks) const {
    stringstream ss;
    // file header
    ss << "Board CMSSW" << endl << " Quad/Chan :";
    // quad header
    for (int link = 0; link < numLinks; link++)
      ss << "        q" << setfill('0') << setw(2) << link / 4 << "c" << link % 4 << "      ";
    ss << endl;
    // link header
    ss << "      Link :";
    for (int link = 0; link < numLinks; link++)
      ss << "         " << setfill('0') << setw(3) << link << "       ";
    ss << endl;
    return ss.str();
  }

  // creates 6 frame gap between packets
  string Demonstrator::infraGap(int& nFrame, int numLinks) const {
    stringstream ss;
    for (int gap = 0; gap < numFramesInfra_; gap++) {
      ss << frame(nFrame);
      for (int link = 0; link < numLinks; link++)
        ss << " 0v" << string(TTBV::S_ / 4, '0');
      ss << endl;
    }
    return ss.str();
  }

  // creates frame number
  string Demonstrator::frame(int& nFrame) const {
    stringstream ss;
    ss << "Frame " << setfill('0') << setw(4) << nFrame++ << " :";
    return ss.str();
  }

  // converts bv into hex
  string Demonstrator::hex(const Frame& bv) const {
    stringstream ss;
    ss << " 1v" << setfill('0') << setw(TTBV::S_ / 4) << std::hex << bv.to_ullong();
    return ss.str();
  }

}  // namespace trackerTFP