#include "L1Trigger/Phase2L1ParticleFlow/src/corrector.h"

#include <iostream>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <unordered_map>
#include <TFile.h>
#include <TKey.h>
#include <TH1.h>
#include <TH2.h>
#include <TAxis.h>
#include "FWCore/Utilities/interface/CPUTimer.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/L1TParticleFlow/interface/PFCluster.h"

l1tpf::corrector::corrector(const std::string &filename, float emfMax, bool debug) : emfMax_(emfMax) {
  if (!filename.empty())
    init_(filename, "", debug);
}
l1tpf::corrector::corrector(const std::string &filename, const std::string &directory, float emfMax, bool debug)
    : emfMax_(emfMax) {
  if (!filename.empty())
    init_(filename, directory, debug);
}

l1tpf::corrector::corrector(TDirectory *src, float emfMax, bool debug) : emfMax_(emfMax) { init_(src, debug); }

void l1tpf::corrector::init_(const std::string &filename, const std::string &directory, bool debug) {
  std::string resolvedFileName = filename;
  if (filename[0] != '/')
    resolvedFileName = edm::FileInPath(filename).fullPath();
  TFile *lFile = TFile::Open(resolvedFileName.c_str());
  if (!lFile || lFile->IsZombie())
    throw cms::Exception("Configuration", "cannot read file " + filename);

  TDirectory *dir = directory.empty() ? lFile : lFile->GetDirectory(directory.c_str());
  if (!dir)
    throw cms::Exception("Configuration", "cannot find directory '" + directory + "' in file " + filename);
  init_(dir, debug);

  lFile->Close();
}

void l1tpf::corrector::init_(TDirectory *lFile, bool debug) {
  TH1 *index = (TH1 *)lFile->Get("INDEX");
  if (!index)
    throw cms::Exception("Configuration")
        << "invalid input file " << lFile->GetPath() << ": INDEX histogram not found.\n";
  index_.reset((TH1 *)index->Clone());
  index_->SetDirectory(nullptr);

  is2d_ = index_->InheritsFrom("TH2");

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
  int ngraphs = 0;
  for (unsigned int ieta = 0; ieta < neta_; ++ieta) {
    for (unsigned int iemf = 0; iemf < nemf_; ++iemf) {
      if (is2d_) {
        snprintf(buff, 31, "eta_bin%d_emf_bin%d", ieta + 1, iemf + 1);
      } else {
        snprintf(buff, 31, "eta_bin%d", ieta + 1);
      }
      TGraph *graph = graphs[buff];
      if (debug)
        edm::LogPrint("corrector") << "   eta bin " << ieta << " emf bin " << iemf << " graph " << buff
                                   << (graph ? " <valid>" : " <nil>") << "\n";
      if (graph) {
        ngraphs++;
        corrections_[ieta * nemf_ + iemf] = (TGraph *)graph->Clone();
      }
      if (std::abs(index_->GetXaxis()->GetBinCenter(ieta + 1)) > 3.0) {
        break;  // no EMF bins beyond eta = 3
      }
    }
  }
}

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

l1tpf::corrector::~corrector() {
  for (TGraph *&p : corrections_) {
    delete p;
    p = nullptr;
  }
  corrections_.clear();
}

l1tpf::corrector::corrector(corrector &&corr)
    : index_(std::move(corr.index_)),
      corrections_(std::move(corr.corrections_)),
      is2d_(corr.is2d_),
      neta_(corr.neta_),
      nemf_(corr.nemf_),
      emfMax_(corr.emfMax_) {}

l1tpf::corrector &l1tpf::corrector::operator=(corrector &&corr) {
  std::swap(is2d_, corr.is2d_);
  std::swap(neta_, corr.neta_);
  std::swap(nemf_, corr.nemf_);
  std::swap(emfMax_, corr.emfMax_);

  index_.swap(corr.index_);
  corrections_.swap(corr.corrections_);
  return *this;
}

float l1tpf::corrector::correctedPt(float pt, float emPt, float eta) const {
  float total = std::max(pt, emPt), abseta = std::abs(eta);
  float emf = emPt / total;
  if (emfMax_ > 0 && emf > emfMax_)
    return total;  // no correction
  unsigned int ieta = std::min(std::max<unsigned>(1, index_->GetXaxis()->FindBin(abseta)), neta_) - 1;
  unsigned int iemf =
      is2d_ && abseta < 3.0 ? std::min(std::max<unsigned>(1, index_->GetYaxis()->FindBin(emf)), nemf_) - 1 : 0;
  TGraph *graph = corrections_[ieta * nemf_ + iemf];
  if (!graph) {
    throw cms::Exception("RuntimeError") << "Error trying to read calibration for eta " << eta << " emf " << emf
                                         << " which are not available." << std::endl;
  }
  float ptcorr = std::min<float>(graph->Eval(total), 4 * total);
  return ptcorr;
}

void l1tpf::corrector::correctPt(l1t::PFCluster &cluster, float preserveEmEt) const {
  float newpt = correctedPt(cluster.pt(), cluster.emEt(), cluster.eta());
  cluster.calibratePt(newpt, preserveEmEt);
}

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

void l1tpf::corrector::writeToFile(const std::string &filename, const std::string &directory) const {
  TFile *lFile = TFile::Open(filename.c_str(), "RECREATE");
  TDirectory *dir = directory.empty() ? lFile : lFile->mkdir(directory.c_str());
  writeToFile(dir);
  lFile->Close();
}

void l1tpf::corrector::writeToFile(TDirectory *dest) const {
  TH1 *index = (TH1 *)index_->Clone();
  index->SetDirectory(dest);
  dest->WriteTObject(index);

  for (const TGraph *p : corrections_) {
    if (p != nullptr) {
      dest->WriteTObject((TGraph *)p->Clone(), p->GetName());
    }
  }
}
