// STL includes
#include <bitset>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// ROOT includes
#include "TROOT.h"
#include "TSystem.h"
#include "TFile.h"
#include "TTree.h"
#include "TLorentzVector.h"

// CMSSW includes
#include "FWCore/FWLite/interface/FWLiteEnabler.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/FWLite/interface/Handle.h"
#include "DataFormats/FWLite/interface/Run.h"
#include "DataFormats/FWLite/interface/LuminosityBlock.h"
#include "DataFormats/FWLite/interface/Event.h"
#include "DataFormats/L1TParticleFlow/interface/PFTrack.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/DiscretePFInputsIO.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/Region.h"

#define NTEST 64
#define REPORT_EVERY_N 50
#define NTRACKS_PER_SECTOR 110
#define NBITS_PER_TRACK 96
static std::vector<l1tpf_impl::Region> regions_;

typedef l1tpf_impl::InputRegion Region;
typedef std::pair<int, int> SectorTrackIndex;
typedef std::map<SectorTrackIndex, TLorentzVector> TrackMap;

struct Event {
  uint32_t run, lumi;
  uint64_t event;
  float z0, genZ0;
  std::vector<float> puGlobals;  // [float] alphaCMed, alphaCRms, alphaFMed, alphaFRms
  std::vector<Region> regions;

  Event() : run(0), lumi(0), event(0), z0(0.), regions() {}
  bool readFromFile(FILE *fRegionDump) {
    if (!fread(&run, sizeof(uint32_t), 1, fRegionDump))
      return false;
    fread(&lumi, sizeof(uint32_t), 1, fRegionDump);
    fread(&event, sizeof(uint64_t), 1, fRegionDump);
    l1tpf_impl::readManyFromFile(regions, fRegionDump);
    fread(&z0, sizeof(float), 1, fRegionDump);
    fread(&genZ0, sizeof(float), 1, fRegionDump);
    l1tpf_impl::readManyFromFile(puGlobals, fRegionDump);
    return true;
  }
};

TLorentzVector makeTLorentzVectorPtEtaPhiE(float pt, float eta, float phi, float e) {
  TLorentzVector v;
  v.SetPtEtaPhiE(pt, eta, phi, e);
  return v;
}

/*
 * Convert a bitset to a signed int64_t.
 * std::bitset has built-ins for ulong and ullong.
 */
template <size_t N, class = std::enable_if_t<(N > 0 && N < 64)>>
int64_t to_int64_from_bitset(const std::bitset<N> &b) {
  int const shift = 64 - N;
  return (((int64_t)b.to_ullong() << shift) >> shift);
}

/*
 * Generic implementation to search if a given value exists in a map or not.
 * Adds all the keys with given value in the vector
 */
template <typename K, typename V, typename T>
bool findAllInRegion(std::vector<K> &vec, std::map<K, V> mapOfElemen, T value) {
  bool bResult = false;
  auto it = mapOfElemen.begin();
  // Iterate through the map
  while (it != mapOfElemen.end()) {
    // Check if value of this entry matches with given value
    if (it->first.first == value) {
      // Yes found
      bResult = true;
      // Push the key in given map
      vec.push_back(it->first);
    }
    // Go to next entry in map
    it++;
  }
  return bResult;
}

TrackMap get_tracks_from_root_file(fwlite::Event &ev, int entry = 0, bool print = false) {
  TrackMap tracks_root;

  // clear the tracks currently stored in the regions
  for (l1tpf_impl::Region &r : regions_) {
    r.track.clear();
  }

  // go to the event under test
  if (!ev.to(entry)) {
    std::cerr << "ERROR::testDumpFile::get_tracks_from_root_file Unable to load the event at entry " << entry
              << std::endl;
    assert(ev.to(entry));
  }
  if (print)
    printf("ROOT::Run %u, lumi %u, event %llu \n",
           ev.getRun().id().run(),
           ev.getLuminosityBlock().id().luminosityBlock(),
           ev.eventAuxiliary().id().event());

  edm::InputTag trackLabel("pfTracksFromL1TracksBarrel");
  edm::Handle<std::vector<l1t::PFTrack>> h_track;
  ev.getByLabel(trackLabel, h_track);
  assert(h_track.isValid());

  int ntrackstotal(0);
  const auto &tracks = *h_track;
  for (unsigned int itk = 0, ntk = tracks.size(); itk < ntk; ++itk) {
    const auto &tk = tracks[itk];
    if (tk.pt() <= 2.0 || tk.nStubs() < 4 || tk.normalizedChi2() >= 15.0)
      continue;
    for (l1tpf_impl::Region &r : regions_) {
      bool inside = r.contains(tk.eta(), tk.phi());
      ;
      if (inside) {
        l1tpf_impl::PropagatedTrack prop;
        prop.fillInput(
            tk.pt(), r.localEta(tk.eta()), r.localPhi(tk.phi()), tk.charge(), tk.vertex().Z(), tk.quality(), &tk);
        prop.fillPropagated(tk.pt(),
                            tk.trkPtError(),
                            tk.caloPtError(),
                            r.localEta(tk.caloEta()),
                            r.localPhi(tk.caloPhi()),
                            tk.quality(),
                            tk.isMuon());
        prop.hwStubs = tk.nStubs();
        prop.hwChi2 = round(tk.chi2() * 10);
        r.track.push_back(prop);
      }
    }
    //if (print) printf("\t\t Track %u (pT,eta,phi): (%.4f,%.4f,%.4f)\n", ntrackstotal, tk.pt(), tk.eta(), tk.phi());
    ntrackstotal++;
  }
  for (unsigned int iregion = 0; iregion < regions_.size(); ++iregion) {
    std::vector<l1tpf_impl::PropagatedTrack> tracks_in_region = regions_[iregion].track;
    if (print)
      printf("\tFound region %u (eta=[%0.4f,%0.4f] phi=[%0.4f,%0.4f]) with %lu tracks\n",
             iregion,
             regions_[iregion].etaMin,
             regions_[iregion].etaMax,
             regions_[iregion].phiCenter - regions_[iregion].phiHalfWidth,
             regions_[iregion].phiCenter + regions_[iregion].phiHalfWidth,
             tracks_in_region.size());
    for (unsigned int it = 0; it < tracks_in_region.size(); it++) {
      if (print)
        printf("\t\t Track %u (pT,eta,phi): (%.4f,%.4f,%.4f)\n",
               it,
               tracks_in_region[it].src->p4().pt(),
               tracks_in_region[it].src->p4().eta(),
               tracks_in_region[it].src->p4().phi());
      tracks_root[std::make_pair(iregion, it)] = makeTLorentzVectorPtEtaPhiE(tracks_in_region[it].src->pt(),
                                                                             tracks_in_region[it].src->eta(),
                                                                             tracks_in_region[it].src->phi(),
                                                                             tracks_in_region[it].src->pt());
    }
  }
  if (print) {
    printf("\t================================= \n");
    printf("\tTotal tracks %u \n\n", ntrackstotal);
  }

  return tracks_root;
}

std::map<std::pair<int, int>, TLorentzVector> get_tracks_from_dump_file(FILE *dfile_ = nullptr, bool print = false) {
  std::map<std::pair<int, int>, TLorentzVector> tracks_dump;
  Event event_;

  if (feof(dfile_)) {
    std::cerr << "ERROR::testDumpFile::get_tracks_from_dump_file We have already reached the end of the dump file"
              << std::endl;
    assert(!feof(dfile_));
  }
  if (!event_.readFromFile(dfile_)) {
    std::cerr << "ERROR::testDumpFile::get_tracks_from_dump_file Something went wrong reading from the dump file"
              << std::endl;
    assert(event_.readFromFile(dfile_));
  }
  if (event_.regions.size() != regions_.size()) {
    printf("ERROR::testDumpFile::get_tracks_from_dump_file Mismatching number of input regions: %lu\n",
           event_.regions.size());
    assert(event_.regions.size() == regions_.size());
  }
  if (print)
    printf("Dump::Run %u, lumi %u, event %lu, regions %lu \n",
           event_.run,
           event_.lumi,
           event_.event,
           event_.regions.size());

  unsigned int ntrackstotal(0);
  float maxabseta(0), maxz(0), minz(0);

  int pv_gen = round(event_.genZ0 * l1tpf_impl::InputTrack::Z0_SCALE);
  int pv_cmssw = round(event_.z0 * l1tpf_impl::InputTrack::Z0_SCALE);

  for (unsigned int is = 0; is < regions_.size(); ++is) {
    const Region &r = event_.regions[is];
    if (print)
      printf("\tRead region %u [%0.2f,%0.2f] with %lu tracks\n",
             is,
             r.phiCenter - r.phiHalfWidth,
             r.phiCenter + r.phiHalfWidth,
             r.track.size());
    ntrackstotal += r.track.size();
    for (unsigned int it = 0; it < r.track.size(); it++) {
      tracks_dump[std::make_pair(is, it)] = makeTLorentzVectorPtEtaPhiE(
          r.track[it].floatVtxPt(), r.track[it].floatVtxEta(), r.track[it].floatVtxPhi(), r.track[it].floatVtxPt());
      if (abs(r.track[it].hwVtxEta) > maxabseta)
        maxabseta = abs(r.track[it].hwVtxEta);
      if (r.track[it].hwZ0 > maxz)
        maxz = r.track[it].hwZ0;
      if (r.track[it].hwZ0 < minz)
        minz = r.track[it].hwZ0;
      if (print)
        printf("\t\t Track %u (pT,eta,phi): (%.4f,%.4f,%.4f)\n",
               it,
               r.track[it].floatVtxPt(),
               r.track[it].floatVtxEta(),
               r.track[it].floatVtxPhi());
    }
  }

  if (print) {
    printf("\t================================= \n");
    printf("\tTotal tracks %u \n", ntrackstotal);
    printf("\tMax abs(eta) %.2f [hw units] \n", maxabseta);
    printf("\tMax abs(eta) %.4f \n", maxabseta / l1tpf_impl::InputTrack::VTX_ETA_SCALE);
    printf("\t[Min,max] track z0 [%.2f,%.2f] [hw units] \n", minz, maxz);
    printf("\t[Min,max] track z0 [%.2f,%.2f] [cm] \n",
           minz / l1tpf_impl::InputTrack::Z0_SCALE,
           maxz / l1tpf_impl::InputTrack::Z0_SCALE);
    printf("\tPV (GEN) %u \n", pv_gen);
    printf("\tPV (CMSSW) %u \n\n", pv_cmssw);
  }

  return tracks_dump;
}

std::map<std::pair<int, int>, TLorentzVector> get_tracks_from_coe_file(std::ifstream &cfile_,
                                                                       bool print = false,
                                                                       bool debug = false) {
  std::map<std::pair<int, int>, TLorentzVector> tracks_coe;
  std::string bset_string_;
  int ntrackstotal(0);
  bool skip(false);

  // check that we haven't reached the end of the file (i.e. there a more events to be read out)
  if (cfile_.eof()) {
    std::cerr << "ERROR::testDumpFile::get_tracks_from_coe_file We have already reached the end of the coe file"
              << std::endl;
    assert(!cfile_.eof());
  }
  if (print)
    printf("COE::Run \"unknown\", lumi \"unknown\", event \"unknown\", regions %lu? \n", regions_.size());

  // read the lines one by one
  for (unsigned int iline = 0; iline < NTRACKS_PER_SECTOR; iline++) {
    bset_string_.resize(NBITS_PER_TRACK);
    for (unsigned int isector = 0; isector < regions_.size(); isector++) {
      cfile_.read(&bset_string_[0], 96);
      std::bitset<NBITS_PER_TRACK> bset_(bset_string_);
      if (bset_.none()) {
        skip = true;
        continue;
      } else {
        skip = false;
      }

      std::bitset<14> hwPt;
      std::bitset<16> hwVtxEta;
      std::bitset<12> hwVtxPhi;
      for (int i = 14 - 1; i >= 0; i--) {
        hwPt.set(i, bset_[i]);
      }
      for (int i = 12 - 1; i >= 0; i--) {
        hwVtxPhi.set(i, bset_[i + 15]);
      }
      for (int i = 16 - 1; i >= 0; i--) {
        hwVtxEta.set(i, bset_[i + 27]);
      }
      float hwVtxPt_f = (float(hwPt.to_ulong()) / l1tpf_impl::CaloCluster::PT_SCALE);
      float hwVtxEta_f = float(to_int64_from_bitset(hwVtxEta)) / l1tpf_impl::InputTrack::VTX_ETA_SCALE;
      float hwVtxPhi_f = float(to_int64_from_bitset(hwVtxPhi)) / l1tpf_impl::InputTrack::VTX_PHI_SCALE;

      if (debug) {
        std::cout << "bset_string_ = " << bset_string_ << std::endl;
        std::cout << "\thwPt (0b) = " << std::flush;
        for (int i = 14 - 1; i >= 0; i--) {
          std::cout << bset_[i] << std::flush;
        }
        std::cout << std::endl;
        std::cout << "\thwVtxPhi (0b) = " << std::flush;
        for (int i = 12 - 1; i >= 0; i--) {
          std::cout << bset_[i + 15] << std::flush;
        }
        std::cout << std::endl;
        std::cout << "\thwVtxEta (0b) = " << std::flush;
        for (int i = 16 - 1; i >= 0; i--) {
          std::cout << bset_[i + 27] << std::flush;
        }
        std::cout << std::endl;
        std::cout << "\thwPt (int) = " << hwPt.to_ulong() << std::endl;
        std::cout << "\thwVtxPhi (int) = " << to_int64_from_bitset(hwVtxPhi) << std::endl;
        std::cout << "\thwVtxEta (int) = " << to_int64_from_bitset(hwVtxEta) << std::endl;
        std::cout << "\thwVtxPt_f (float) = " << hwVtxPt_f << std::endl;
        std::cout << "\thwVtxPhi_f (float) = " << hwVtxPhi_f << std::endl;
        std::cout << "\thwVtxEta_f (float) = " << hwVtxEta_f << std::endl;
      }

      if (bset_.any()) {
        ntrackstotal++;
        tracks_coe[std::make_pair(isector, iline)] =
            makeTLorentzVectorPtEtaPhiE(hwVtxPt_f, hwVtxEta_f, hwVtxPhi_f, hwVtxPt_f);
        //if (print) printf("\t\t Track %u (pT,eta,phi): (%.4f,%.4f,%.4f)\n", it, hwPt_f, hwVtxEta_f, hwVtxPhi_f);
      }
    }

    // remove the trailing character
    bset_string_.resize(2);
    cfile_.read(&bset_string_[0], 2);
    if (debug && !skip)
      std::cout << "bset_string_ = " << bset_string_ << std::endl;
    if (bset_string_ != ",\n" && bset_string_ != ";\n") {
      std::cerr << "ERROR::testDumpFile::get_tracks_from_coe_file Something went wrong reading line " << 11 + iline
                << " of the COE file" << std::endl
                << "\tThe line should have ended with \',<newline>\' or \';<newline>\', but instead ended with \'"
                << bset_string_ << "\'" << std::endl;
      assert(bset_string_ != "," || bset_string_ != ";");
    }
  }
  for (unsigned int is = 0; is < regions_.size(); ++is) {
    std::vector<SectorTrackIndex> tracks_in_sector;
    findAllInRegion<SectorTrackIndex, TLorentzVector, int>(tracks_in_sector, tracks_coe, is);
    if (print)
      printf("\tRead region %u (eta=[%0.4f,%0.4f] phi=[%0.4f,%0.4f]) with %lu tracks\n",
             is,
             regions_[is].etaMin,
             regions_[is].etaMax,
             regions_[is].phiCenter - regions_[is].phiHalfWidth,
             regions_[is].phiCenter + regions_[is].phiHalfWidth,
             tracks_in_sector.size());
    for (unsigned int it = 0; it < tracks_in_sector.size(); it++) {
      if (print)
        printf("\t\t Track %u (pT,eta,phi): (%.4f,%.4f,%.4f)\n",
               it,
               tracks_coe[tracks_in_sector[it]].Pt(),
               tracks_coe[tracks_in_sector[it]].Eta(),
               tracks_coe[tracks_in_sector[it]].Phi());
    }
  }

  if (print) {
    printf("\t================================= \n");
    printf("\tTotal tracks %u \n\n", ntrackstotal);
  }

  return tracks_coe;
}

std::ifstream &GotoLine(std::ifstream &file, unsigned int num) {
  file.seekg(std::ios::beg);
  for (unsigned int i = 0; i < num - 1; ++i) {
    file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }
  return file;
}

bool compare_lv_with_tolerance(TLorentzVector a, TLorentzVector b, const std::vector<float> &tolerance = {0, 0, 0, 0}) {
  /*
	Example (Tolerance = 0.0005):
		Track from ROOT file: pt=16.3452797
		InputTrack::INVPT_SCALE = 2E4
		std::numeric_limits<uint16_t>::max() = 65535
		hwInvpt = std::min<double>(round(1/pt  * InputTrack::INVPT_SCALE), std::numeric_limits<uint16_t>::max()) = 1224.0000
		floatVtxPt() = 1/(float(hwInvpt) / InputTrack::INVPT_SCALE) = 16.339869
		So loss of precision comes from rounding
		Difference is DeltaPt=0.00541114807
	*/
  if (abs(a.Pt() - b.Pt()) > tolerance[0] || abs(a.Eta() - b.Eta()) > tolerance[1] ||
      abs(a.Phi() - b.Phi()) > tolerance[2] || abs(a.E() - b.E()) > tolerance[3]) {
    std::cerr << std::setprecision(9);
    std::cerr << std::endl << "\tMismatching " << std::flush;
    if (abs(a.Pt() - b.Pt()) > tolerance[0])
      std::cerr << "pT! " << a.Pt() << " vs " << b.Pt() << " where DeltaPt=" << abs(a.Pt() - b.Pt())
                << " and epsilon=" << tolerance[0] << std::endl;
    else if (abs(a.Eta() - b.Eta()) > tolerance[1])
      std::cerr << "eta! " << a.Eta() << " vs " << b.Eta() << " where DeltaEta=" << abs(a.Eta() - b.Eta())
                << " and epsilon=" << tolerance[1] << std::endl;
    else if (abs(a.Phi() - b.Phi()) > tolerance[2])
      std::cerr << "phi! " << a.Phi() << " vs " << b.Phi() << " where DeltaPhi=" << abs(a.Phi() - b.Phi())
                << " and epsilon=" << tolerance[2] << std::endl;
    else if (abs(a.E() - b.E()) > tolerance[3])
      std::cerr << "E! " << a.E() << " vs " << b.E() << " where DeltaE=" << abs(a.E() - b.E())
                << " and epsilon=" << tolerance[3] << std::endl;
    return false;
  }
  return true;
}

bool compare_maps(TrackMap ref, TrackMap test) {
  TLorentzVector tlv;
  for (auto it = ref.begin(); it != ref.end(); it++) {
    if (test.find(it->first) == test.end()) {
      std::cerr << std::endl
                << "\tERROR::compare_maps Can't find the test track with (sector,index)=(" << it->first.first << ","
                << it->first.second << ")" << std::endl;
      return false;
    }
    tlv = (test.find(it->first)->second);
    // The pT tolerance should be 1.0/l1tpf_impl::CaloCluster::PT_SCALE, but because of the rounding this is not true and the actual resolution isn't always as good
    // Instead, we will use max(1% of the pT of the reference TLorentzVector,0.25)
    // We use the max statement because at low pT, the 1% definition doesn't hold anymore. This wouldn't be a problem if 1/pT were encoded rather than pT.
    if (!compare_lv_with_tolerance(
            (it->second),
            tlv,
            {float(std::max(it->second.Pt() * 1E-2, 1.0 / l1tpf_impl::CaloCluster::PT_SCALE)),
             1.0 / l1tpf_impl::InputTrack::VTX_ETA_SCALE,
             1.0 / l1tpf_impl::InputTrack::VTX_PHI_SCALE,
             float(std::max(it->second.Pt() * 1E-2, 1.0 / l1tpf_impl::CaloCluster::PT_SCALE))})) {
      std::cerr << std::endl
                << "\tERROR::compare_maps Can't find the test track with TLorentzVector (" << it->second.Pt() << ","
                << it->second.Eta() << "," << it->second.Phi() << "," << it->second.E() << ")" << std::endl
                << "\t\tInstead found (" << tlv.Pt() << "," << tlv.Eta() << "," << tlv.Phi() << "," << tlv.E()
                << ") at the position (sector,index)=(" << it->first.first << "," << it->first.second << ")"
                << std::endl;
      return false;
    }
  }
  return true;
}

int main(int argc, char *argv[]) {
  // store some programatic information
  std::stringstream usage;
  usage << "usage: " << argv[0]
        << " <filename>.root <filename>.dump <filename>.coe <etaExtra> <phiExtra> <nRegionsPhi> <etaBoundaries>";

  // load framework libraries
  gSystem->Load("libFWCoreFWLite");
  FWLiteEnabler::enable();

  // argc should be 5 for correct execution
  // We print argv[0] assuming it is the program name
  if (argc < 9) {
    std::cerr << "ERROR::testDumpFile " << argc << " arguments provided" << std::endl;
    for (int i = 0; i < argc; i++) {
      std::cerr << "\tArgument " << i << ": " << argv[i] << std::endl;
    }
    std::cerr << usage.str() << std::endl;
    return -1;
  }

  // assign the command-line parameters to variables and setup the regions
  std::string filename_root = argv[1];
  std::string filename_dump = argv[2];
  std::string filename_coe = argv[3];
  float etaExtra, phiExtra;
  unsigned int nRegionsPhi;
  std::vector<float> etaBoundaries;
  try {
    etaExtra = atof(argv[4]);
    phiExtra = atof(argv[5]);
    nRegionsPhi = atoi(argv[6]);
    std::vector<std::string> etaBoundariesStrings(argv + 7, argv + argc);
    std::size_t pos;
    for (unsigned int i = 0; i < etaBoundariesStrings.size(); i++) {
      etaBoundaries.push_back(std::stoi(etaBoundariesStrings[i], &pos));
      if (pos < etaBoundariesStrings[i].size()) {
        std::cerr << "Trailing characters after number: " << etaBoundariesStrings[i] << '\n';
      }
    }
    float phiWidth = 2 * M_PI / nRegionsPhi;
    for (unsigned int ieta = 0, neta = etaBoundaries.size() - 1; ieta < neta; ++ieta) {
      for (unsigned int iphi = 0; iphi < nRegionsPhi; ++iphi) {
        float phiCenter = (iphi + 0.5) * phiWidth - M_PI;
        regions_.push_back(l1tpf_impl::Region(etaBoundaries[ieta],
                                              etaBoundaries[ieta + 1],
                                              phiCenter,
                                              phiWidth,
                                              phiExtra,
                                              etaExtra,
                                              false,
                                              0,
                                              0,
                                              0,
                                              0,
                                              0,
                                              0));
      }
    }
  } catch (std::invalid_argument const &ex) {
    std::cerr << "Invalid number in one of the eta-phi arguments" << std::endl;
    return -2;
  } catch (std::out_of_range const &ex) {
    std::cerr << "Number out of range in one of the eta-phi arguments" << std::endl;
    return -3;
  }

  // check the filenames
  if (filename_root.find(".root") == std::string::npos) {
    std::cerr << "ERROR::testDumpFile Filename 1 must be a ROOT (.root) file" << std::endl << usage.str() << std::endl;
    return -4;
  } else if (filename_dump.find(".dump") == std::string::npos) {
    std::cerr << "ERROR::testDumpFile Filename 2 must be a binary (.dump) file" << std::endl
              << usage.str() << std::endl;
    return -5;
  } else if (filename_coe.find(".coe") == std::string::npos) {
    std::cerr << "ERROR::testDumpFile Filename 3 must be a COE (.coe) file" << std::endl << usage.str() << std::endl;
    return -6;
  }

  // report the program configuraion
  std::cout << "Configuration:" << std::endl
            << "==============" << std::endl
            << "Number of tests (events): " << NTEST << std::endl
            << "Report every N tests: " << REPORT_EVERY_N << std::endl
            << "Number of regions (in eta-phi): " << regions_.size() << std::endl;
  for (unsigned int iregion = 0; iregion < regions_.size(); iregion++) {
    printf("\t%i : eta=[%0.4f,%0.4f] phi=[%0.4f,%0.4f]\n",
           iregion,
           regions_[iregion].etaMin,
           regions_[iregion].etaMax,
           regions_[iregion].phiCenter - regions_[iregion].phiHalfWidth,
           regions_[iregion].phiCenter + regions_[iregion].phiHalfWidth);
  }
  std::cout << "Number of tracks per sector: " << NTRACKS_PER_SECTOR << std::endl
            << "Number of bits per track: " << NBITS_PER_TRACK << std::endl
            << "==============" << std::endl
            << std::endl;

  // open the files for testing
  TFile *rfile_ = TFile::Open(filename_root.c_str(), "READ");
  if (!rfile_) {
    std::cerr << "ERROR::testDumpFile Cannot open '" << filename_root << "'" << std::endl;
    return -7;
  }
  fwlite::Event rfileentry_(rfile_);
  FILE *dfile_(fopen(filename_dump.c_str(), "rb"));
  if (!dfile_) {
    std::cerr << "ERROR::testDumpFile Cannot read '" << filename_dump << "'" << std::endl;
    return -8;
  }
  std::ifstream cfile_(filename_coe);
  if (!cfile_) {
    std::cerr << "ERROR::testDumpFile Cannot read '" << filename_coe << "'" << std::endl;
    return -9;
  }
  GotoLine(cfile_, 11);  //Skip the header of the COE file

  TrackMap tracks_root, tracks_dump, tracks_coe;

  // run the tests for multiple events
  for (int test = 1; test <= NTEST; ++test) {
    if (test % REPORT_EVERY_N == 1)
      std::cout << "Doing test " << test << " ... " << std::endl;

    tracks_root = get_tracks_from_root_file(rfileentry_, test - 1, test == 1);
    tracks_dump = get_tracks_from_dump_file(dfile_, test == 1);
    tracks_coe = get_tracks_from_coe_file(cfile_, test == 1);

    if (test % REPORT_EVERY_N == 1)
      std::cout << "Comparing the ROOT tracks to the dump tracks in event " << test << " ... " << std::flush;
    if (!compare_maps(tracks_root, tracks_dump))
      return -10;
    if (test % REPORT_EVERY_N == 1)
      std::cout << "DONE" << std::endl;

    if (test % REPORT_EVERY_N == 1)
      std::cout << "Comparing the ROOT tracks to the coe tracks in event " << test << " ... " << std::flush;
    if (!compare_maps(tracks_root, tracks_coe))
      return -11;
    if (test % REPORT_EVERY_N == 1)
      std::cout << "DONE" << std::endl << std::endl;
  }

  std::cout << std::endl << "The dump and coe outputs match the ROOT outputs for all events!" << std::endl;
  return 0;
}

/*
USE:
g++ -I/uscms_data/d2/aperloff/YOURWORKINGAREA/TSABoard/slc7/CMSSW_10_6_0_pre4/src/L1Trigger/Phase2L1ParticleFlow/interface/ -O0 -g3 -Wall -std=c++0x -c -fmessage-length=0 testDumpFile.cpp
g++ -o testDumpFile testDumpFile.o
./testDumpFile trackerRegion_alltracks_sectors_1x18_TTbar_PU200.dump 18

scram b runtests
*/
