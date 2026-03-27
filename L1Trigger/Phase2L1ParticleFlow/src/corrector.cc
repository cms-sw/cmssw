#include "L1Trigger/Phase2L1ParticleFlow/interface/corrector.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/dbgPrintf.h"

#include <iostream>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <unordered_map>
#ifdef L1PF_USE_ROOT
#include <TFile.h>
#include <TKey.h>
#include <TH1.h>
#include <TH2.h>
#include <TAxis.h>
#endif

#ifdef CMSSW_GIT_HASH
#include "FWCore/Utilities/interface/CPUTimer.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/L1TParticleFlow/interface/PFCluster.h"
#else
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <fstream>
#endif

#include <nlohmann/json.hpp>

/* ---
 * Note: #ifdef CMSSW_GIT_HASH is used to determine whether compilation is in CMSSW context or not
 * since this same implementation is used in the standalone tests of corrections in comparison to
 * L1T firmware
 * --- */

l1tpf::corrector::corrector(
    const std::string &iFile, float emfMax, bool debug, bool emulate, l1tpf::corrector::EmulationMode emulationMode)
    : emfMax_(emfMax), emulate_(emulate), debug_(debug), emulationMode_(emulationMode) {
  if (!iFile.empty()) {
#ifdef L1PF_USE_ROOT
    // If a ROOT file path was passed by mistake, fall back to ROOT init
    init_(iFile, "", debug, emulate);
#else
    initJson_(iFile, debug, emulate);
#endif
  }
}

#ifdef L1PF_USE_ROOT
l1tpf::corrector::corrector(const std::string &filename,
                            const std::string &directory,
                            float emfMax,
                            bool debug,
                            bool emulate,
                            l1tpf::corrector::EmulationMode emulationMode)
    : emfMax_(emfMax), emulate_(emulate), debug_(debug), emulationMode_(emulationMode) {
  if (!filename.empty())
    init_(filename, directory, debug, emulate);
}

l1tpf::corrector::corrector(
    TDirectory *src, float emfMax, bool debug, bool emulate, l1tpf::corrector::EmulationMode emulationMode)
    : emfMax_(emfMax), emulate_(emulate), debug_(debug), emulationMode_(emulationMode) {
  init_(src, debug);
}
#endif

#ifdef L1PF_USE_ROOT
void l1tpf::corrector::init_(const std::string &filename, const std::string &directory, bool debug, bool emulate) {
  std::string resolvedFileName = filename;
  if (filename[0] != '/') {
#ifdef CMSSW_GIT_HASH
    resolvedFileName = edm::FileInPath(filename).fullPath();
#else
    resolvedFileName = std::filesystem::absolute(filename);
#endif
  }
  TFile *lFile = TFile::Open(resolvedFileName.c_str());
  if (!lFile || lFile->IsZombie()) {
#ifdef CMSSW_GIT_HASH
    throw cms::Exception("Configuration", "cannot read file " + filename);
#else
    throw std::runtime_error("Cannot read file " + filename);
#endif
  }

  TDirectory *dir = directory.empty() ? lFile : lFile->GetDirectory(directory.c_str());
  if (!dir) {
#ifdef CMSSW_GIT_HASH
    throw cms::Exception("Configuration", "cannot find directory '" + directory + "' in file " + filename);
#else
    throw std::runtime_error("Cannot find directory '" + directory + "' in file " + filename);
#endif
  }
  init_(dir, debug);
  lFile->Close();
}

void l1tpf::corrector::init_(TDirectory *lFile, bool debug) {
  std::string index_name = "INDEX";
  TH1 *index = nullptr;
  if (emulate_) {
    // Loof for emulated version or fallback to regular one
    index_name = "EMUL_INDEX";
    index = (TH1 *)lFile->Get(index_name.c_str());
    if (!index) {
      index_name = "INDEX";
      index = (TH1 *)lFile->Get(index_name.c_str());
    }
  } else {
    index = (TH1 *)lFile->Get(index_name.c_str());
  }
  if (!index) {
#ifdef CMSSW_GIT_HASH
    throw cms::Exception("Configuration")
        << "invalid input file " << lFile->GetPath() << ": INDEX histogram not found.\n";
#else
    std::stringstream ss;
    ss << "invalid input file " << lFile->GetPath() << ": INDEX histogram not found.\n";
    throw std::runtime_error(ss.str());
#endif
  }
  index_.reset((TH1 *)index->Clone());
  index_->SetDirectory(nullptr);

  is2d_ = index_->InheritsFrom("TH2");

  if (emulate_)
    initEmulation_(lFile, debug);
  else
    initGraphs_(lFile, debug);
}
#endif

#ifdef L1PF_USE_ROOT
void l1tpf::corrector::initGraphs_(TDirectory *lFile, bool debug) {
  std::unordered_map<std::string, TGraph *> graphs;
  TKey *key;
  TIter nextkey(lFile->GetListOfKeys());
  while ((key = (TKey *)nextkey())) {
    if (strncmp(key->GetName(), "eta_", 4) == 0) {
      TGraph *gr = (TGraph *)key->ReadObj();
      if (!gr->TestBit(TGraph::kIsSortedX))
        gr->Sort();
      graphs[key->GetName()] = gr;
    }
  }

  neta_ = index_->GetNbinsX();
  nemf_ = (is2d_ ? index_->GetNbinsY() : 1);
  corrections_.resize(neta_ * nemf_);
  std::fill(corrections_.begin(), corrections_.end(), nullptr);
  char buff[32];
  for (unsigned int ieta = 0; ieta < neta_; ++ieta) {
    for (unsigned int iemf = 0; iemf < nemf_; ++iemf) {
      if (is2d_) {
        snprintf(buff, 31, "eta_bin%d_emf_bin%d", ieta + 1, iemf + 1);
      } else {
        snprintf(buff, 31, "eta_bin%d", ieta + 1);
      }
      TGraph *graph = graphs[buff];
      if (debug) {
#ifdef CMSSW_GIT_HASH
        edm::LogPrint("corrector") << "   eta bin " << ieta << " emf bin " << iemf << " graph " << buff
                                   << (graph ? " <valid>" : " <nil>") << "\n";
#else
        dbgCout() << "   eta bin " << ieta << " emf bin " << iemf << " graph " << buff
                  << (graph ? " <valid>" : " <nil>") << "\n";
#endif
      }
      if (graph) {
        corrections_[ieta * nemf_ + iemf] = (TGraph *)graph->Clone();
      }
      if (std::abs(index_->GetXaxis()->GetBinCenter(ieta + 1)) > 3.0) {
        break;  // no EMF bins beyond eta = 3
      }
    }
  }
}
#endif

#ifdef L1PF_USE_ROOT
void l1tpf::corrector::initEmulation_(TDirectory *lFile, bool debug) {
  std::string histo_base_name = "";
  if (emulationMode_ == l1tpf::corrector::EmulationMode::Correction)
    histo_base_name = "emul_corr_eta";
  else if (emulationMode_ == l1tpf::corrector::EmulationMode::CorrectedPt)
    histo_base_name = "emul_eta";

  std::unordered_map<std::string, TH1 *> hists;
  TKey *key;
  TIter nextkey(lFile->GetListOfKeys());
  while ((key = (TKey *)nextkey())) {
    if (strncmp(key->GetName(), histo_base_name.c_str(), histo_base_name.size()) == 0) {
      TH1 *hist = (TH1 *)key->ReadObj();
      hists[key->GetName()] = hist;
    }
  }

  neta_ = index_->GetNbinsX();
  nemf_ = (is2d_ ? index_->GetNbinsY() : 1);
  correctionsEmulated_.resize(neta_ * nemf_);
  std::fill(correctionsEmulated_.begin(), correctionsEmulated_.end(), nullptr);
  char buff[32];
  for (unsigned int ieta = 0; ieta < neta_; ++ieta) {
    for (unsigned int iemf = 0; iemf < nemf_; ++iemf) {
      if (is2d_) {
        snprintf(buff, 31, "%s_bin%d_emf_bin%d", histo_base_name.c_str(), ieta + 1, iemf + 1);
      } else {
        snprintf(buff, 31, "%s_bin%d", histo_base_name.c_str(), ieta + 1);
      }
      TH1 *hist = hists[buff];
      if (debug) {
#ifdef CMSSW_GIT_HASH
        edm::LogPrint("corrector") << "   eta bin " << ieta << " emf bin " << iemf << " hist " << buff
                                   << (hist ? " <valid>" : " <nil>") << "\n";
#else
        dbgCout() << "   eta bin " << ieta << " emf bin " << iemf << " hist " << buff << (hist ? " <valid>" : " <nil>")
                  << "\n";
#endif
      }
      if (hist) {
        correctionsEmulated_[ieta * nemf_ + iemf] = (TH1 *)hist->Clone();
        correctionsEmulated_[ieta * nemf_ + iemf]->SetDirectory(nullptr);
      }
      if (std::abs(index_->GetXaxis()->GetBinCenter(ieta + 1)) > 3.0) {
        break;  // no EMF bins beyond eta = 3
      }
    }
  }
}
#endif

#ifdef L1PF_USE_ROOT
l1tpf::corrector::corrector(const TH1 *index, float emfMax)
    : index_((TH1 *)index->Clone("INDEX")),
      is2d_(index->InheritsFrom("TH2")),
      neta_(index->GetNbinsX()),
      nemf_(is2d_ ? index->GetNbinsY() : 1),
      emfMax_(emfMax) {
  index_->SetDirectory(nullptr);
  corrections_.resize(neta_ * nemf_);
  std::fill(corrections_.begin(), corrections_.end(), nullptr);
}
#endif

l1tpf::corrector::~corrector() {
#ifdef L1PF_USE_ROOT
  for (TGraph *&p : corrections_) {
    delete p;
    p = nullptr;
  }
  corrections_.clear();
  for (TH1 *&p : correctionsEmulated_) {
    delete p;
    p = nullptr;
  }
  correctionsEmulated_.clear();
#endif
}

l1tpf::corrector::corrector(corrector &&corr)
#ifdef L1PF_USE_ROOT
    : index_(std::move(corr.index_)),
      corrections_(std::move(corr.corrections_)),
      correctionsEmulated_(std::move(corr.correctionsEmulated_)),
#else
    : index_(std::move(corr.index_)),
      correctionsJson_(std::move(corr.correctionsJson_)),
#endif
      is2d_(corr.is2d_),
      neta_(corr.neta_),
      nemf_(corr.nemf_),
      emfMax_(corr.emfMax_),
      emulate_(corr.emulate_) {
}

l1tpf::corrector &l1tpf::corrector::operator=(corrector &&corr) {
  std::swap(is2d_, corr.is2d_);
  std::swap(neta_, corr.neta_);
  std::swap(nemf_, corr.nemf_);
  std::swap(emfMax_, corr.emfMax_);
  std::swap(emulate_, corr.emulate_);

#ifdef L1PF_USE_ROOT
  index_.swap(corr.index_);
  corrections_.swap(corr.corrections_);
  correctionsEmulated_.swap(corr.correctionsEmulated_);
#else
  index_.swap(corr.index_);
  correctionsJson_.swap(corr.correctionsJson_);
#endif
  return *this;
}

#ifndef L1PF_USE_ROOT
// Helper: find bin index given edges (0-based bin), clamps to [0, nbins-1]
static unsigned int findBinFromEdges_(const std::vector<float> &edges, float v) {
  if (edges.empty())
    return 0;
  const unsigned int nb = edges.size() - 1;
  if (v <= edges.front())
    return 0;
  if (v >= edges.back())
    return nb - 1;
  // linear search (edges are small); can replace with binary search if needed
  for (unsigned int i = 0; i < nb; ++i) {
    if (v >= edges[i] && v < edges[i + 1])
      return i;
  }
  return nb - 1;
}

// JSON init (non-ROOT path)
void l1tpf::corrector::initJson_(const std::string &jsonPath, bool debug, bool emulate) {
  emulate_ = emulate;
  debug_ = debug;
  // Read file
  std::ifstream fin(jsonPath);
  if (!fin.good()) {
    throw std::runtime_error("Cannot read JSON file " + jsonPath);
  }
  nlohmann::json j;
  fin >> j;
  fin.close();
  const auto &idx = j["index"];
  std::vector<float> eta_edges = idx["eta_edges"].get<std::vector<float>>();
  std::vector<float> emf_edges =
      idx.contains("emf_edges") ? idx["emf_edges"].get<std::vector<float>>() : std::vector<float>{0.f, 1.f};

  neta_ = (unsigned int)(eta_edges.size() - 1);
  nemf_ = (unsigned int)(emf_edges.size() - 1);
  is2d_ = nemf_ > 1;

  index_ = std::make_unique<Binning2D>(Binning2D{Binning1D{eta_edges}, Binning1D{emf_edges}});
  correctionsJson_.clear();
  correctionsJson_.resize(neta_ * nemf_);
  for (const auto &entry : j["corr"]) {
    unsigned int ieta = entry["ieta"].get<unsigned int>() - 1;
    unsigned int iemf = entry["iemf"].get<unsigned int>() - 1;
    const auto pt_edges = entry["pt_bins"].get<std::vector<float>>();
    const auto values = entry["values"].get<std::vector<float>>();
    correctionsJson_[ieta * nemf_ + iemf] = EmulHistogram{pt_edges, values};
  }
}
#endif

float l1tpf::corrector::correctedPt(float pt, float emPt, float eta) const {
  unsigned int ipt, ieta;
  float total = std::max(pt, emPt), abseta = std::abs(eta);
  float emf = emPt / total;
  if (emfMax_ > 0 && emf > emfMax_)
    return total;  // no correction
#ifdef L1PF_USE_ROOT
  ieta = std::min(std::max<unsigned>(1, index_->GetXaxis()->FindBin(abseta)), neta_) - 1;
  // FIXME: why eta 3.1 is hardcoded here?
  static const float maxeta = 3.1;
  unsigned int iemf =
      is2d_ && abseta < maxeta ? std::min(std::max<unsigned>(1, index_->GetYaxis()->FindBin(emf)), nemf_) - 1 : 0;
  float ptcorr = 0;
  if (!emulate_) {  // not emulation - read from the TGraph as normal
    TGraph *graph = corrections_[ieta * nemf_ + iemf];
    if (!graph) {
#ifdef CMSSW_GIT_HASH
      throw cms::Exception("RuntimeError") << "Error trying to read calibration for eta " << eta << " emf " << emf
                                           << " which are not available." << std::endl;
#else
      std::stringstream ss;
      ss << "Error trying to read calibration for eta " << eta << " emf " << emf << " which are not available."
         << std::endl;
      throw std::runtime_error(ss.str());
#endif
    }

    ptcorr = std::min<float>(graph->Eval(total), 4 * total);
  } else {  // emulation - read from the pt binned histogram
    TH1 *hist = correctionsEmulated_[ieta * nemf_ + iemf];
    if (!hist) {
#ifdef CMSSW_GIT_HASH
      throw cms::Exception("RuntimeError")
          << "Error trying to read emulated calibration for eta " << eta << " which are not available." << std::endl;
#else
      std::stringstream ss;
      ss << "Error trying to read emulated calibration for eta " << eta << " which are not available." << std::endl;
      throw std::runtime_error(ss.str());
#endif
    }
    ipt = hist->GetXaxis()->FindBin(pt);
    ptcorr = hist->GetBinContent(ipt);
    if (emulationMode_ == l1tpf::corrector::EmulationMode::Correction) {
      ptcorr = ptcorr * pt;
    }
    if (debug_)
      dbgCout() << "[EMU] ieta: " << ieta << " iemf: " << iemf << " ipt: " << ipt - 1
                << "corr: " << hist->GetBinContent(ipt) << " ptcorr: " << ptcorr << std::endl;
  }
  return ptcorr;
#else
  // Non-ROOT path using JSON-backed binning and histograms
  unsigned int iemf;
  ieta = findBinFromEdges_(index_->eta.edges, abseta);
  static const float maxeta = 3.1;
  iemf = is2d_ && abseta < maxeta ? findBinFromEdges_(index_->emf.edges, emf) : 0;
  float ptcorr = 0.f;
  if (!emulate_) {
    throw std::runtime_error("Non-emulation (graph) mode not available without ROOT.");
  } else {
    const EmulHistogram &h = correctionsJson_[ieta * nemf_ + iemf];
    if (h.binEdges.empty()) {
      throw std::runtime_error("JSON emulated histogram missing for given eta/emf bin");
    }
    ipt = findBinFromEdges_(h.binEdges, pt);
    // values includes overflow element; clamp index
    unsigned int vi = std::min<unsigned int>(ipt, h.values.size() > 0 ? (unsigned int)(h.values.size() - 1) : 0);
    ptcorr = h.values[vi];
    if (emulationMode_ == l1tpf::corrector::EmulationMode::Correction) {
      ptcorr = ptcorr * pt;
    }
    if (debug_) {
      dbgCout() << "[EMU-JSON] ieta: " << ieta << " iemf: " << iemf << " ipt: " << ipt << " corr: " << h.values[vi]
                << " ptcorr: " << ptcorr << std::endl;
    }
  }
  return ptcorr;
#endif
}

#ifdef CMSSW_GIT_HASH
void l1tpf::corrector::correctPt(l1t::PFCluster &cluster, float preserveEmEt) const {
  float newpt = correctedPt(cluster.pt(), cluster.emEt(), cluster.eta());
  cluster.calibratePt(newpt, preserveEmEt);
}
#endif

#ifdef L1PF_USE_ROOT
void l1tpf::corrector::setGraph(const TGraph &graph, int ieta, int iemf) {
  char buff[32];
  if (is2d_) {
    snprintf(buff, 31, "eta_bin%d_emf_bin%d", (unsigned int)(ieta + 1), (unsigned int)(iemf + 1));
  } else {
    snprintf(buff, 31, "eta_bin%d", (unsigned int)(ieta + 1));
  }
  TGraph *gclone = (TGraph *)graph.Clone(buff);
  delete corrections_[ieta * nemf_ + iemf];
  corrections_[ieta * nemf_ + iemf] = gclone;
}
#endif

void l1tpf::corrector::writeToFile(const std::string &filename, const std::string &directory) const {
#ifdef L1PF_USE_ROOT
  TFile *lFile = TFile::Open(filename.c_str(), "RECREATE");
  TDirectory *dir = directory.empty() ? lFile : lFile->mkdir(directory.c_str());
  writeToFile(dir);
  lFile->Close();
#else
  // JSON path: write a portable file
  nlohmann::json j;
  j["meta"] = {{"format", "pt_corrections_v1"}, {"neta", neta_}, {"nemf", nemf_}};
  j["index"]["eta_edges"] = index_->eta.edges;
  j["index"]["emf_edges"] = index_->emf.edges;
  j["corr"] = nlohmann::json::array();
  for (unsigned int ieta = 0; ieta < neta_; ++ieta) {
    for (unsigned int iemf = 0; iemf < nemf_; ++iemf) {
      const auto &h = correctionsJson_[ieta * nemf_ + iemf];
      j["corr"].push_back({{"ieta", ieta}, {"iemf", iemf}, {"pt_bins", h.binEdges}, {"values", h.values}});
    }
  }
  std::ofstream fout(filename);
  fout << j.dump(2);
  fout.close();
#endif
}

#ifdef L1PF_USE_ROOT
void l1tpf::corrector::writeToFile(TDirectory *dest) const {
  TH1 *index = (TH1 *)index_->Clone();
  index->SetDirectory(dest);
  dest->WriteTObject(index);

  for (const TGraph *p : corrections_) {
    if (p != nullptr) {
      dest->WriteTObject((TGraph *)p->Clone(), p->GetName());
    }
  }

  for (const TH1 *p : correctionsEmulated_) {
    if (p != nullptr) {
      dest->WriteTObject((TH1 *)p->Clone(), p->GetName());
    }
  }
}
#endif
