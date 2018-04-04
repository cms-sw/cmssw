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

#include "DataFormats/Phase2L1ParticleFlow/interface/PFCluster.h"


l1tpf::corrector::corrector(const std::string &filename, float emfMax, bool debug) :
    emfMax_(emfMax)
{
    if (!filename.empty()) init_(filename, debug);
}

void l1tpf::corrector::init_(const std::string &filename, bool debug)
{
    std::string resolvedFileName = filename; 
    if (filename[0] != '/') resolvedFileName = edm::FileInPath(filename).fullPath();
    TFile *lFile = TFile::Open(resolvedFileName.c_str());
    if (!lFile || lFile->IsZombie()) throw cms::Exception("Configuration", "cannot read file "+filename);

    TH1 *index = (TH1*) lFile->Get("INDEX");
    if (!index) throw cms::Exception("Configuration", "invalid input file "+filename+": index histogram not found.\n");
    index_.reset((TH1*)index->Clone()); 
    index_->SetDirectory(nullptr);

    is2d_ = index_->InheritsFrom("TH2");

    edm::CPUTimer timer;
    timer.start();
    std::unordered_map<std::string,TGraph *> graphs;
    TKey *key;
    TIter nextkey(lFile->GetListOfKeys());
    while ((key = (TKey *) nextkey())) {
        if (strncmp(key->GetName(), "eta_", 4) == 0) {
            TGraph *gr = (TGraph*) key->ReadObj();
            if (!gr->TestBit(TGraph::kIsSortedX)) gr->Sort();
            graphs[key->GetName()] = gr;
        }
    }

    neta_ = index_->GetNbinsX();
    nemf_ = (is2d_ ? index_->GetNbinsY() : 1);
    corrections_.resize(neta_*nemf_);
    std::fill(corrections_.begin(), corrections_.end(), nullptr);
    char buff[32];
    int ngraphs = 0;
    for (unsigned int ieta = 0; ieta  < neta_; ++ieta) {
        for (unsigned int iemf = 0; iemf  < nemf_; ++iemf) {
            if (is2d_) {
                snprintf(buff, 31, "eta_bin%d_emf_bin%d", ieta+1, iemf+1);
            } else {
                snprintf(buff, 31, "eta_bin%d", ieta+1);
            }
            TGraph *graph = graphs[buff];
            if (debug) std::cout << "   eta bin " << ieta << " emf bin " << iemf << " graph " << buff << ( graph ? " <valid>" : " <nil>")  << std::endl; 
            if (graph) {
                ngraphs++;
                corrections_[ieta * nemf_ + iemf] = (TGraph*) graph->Clone(); 
            }
            if (std::abs(index_->GetXaxis()->GetBinCenter(ieta+1)) > 3.0) {
                break; // no EMF bins beyond eta = 3
            }
        }
    }
    timer.stop();
    if (debug) std::cout << "Read " << ngraphs << " graphs from " << filename << " in " << timer.realTime() << " s"  << std::endl; 

    lFile->Close();
}

l1tpf::corrector::~corrector() {
    for (TGraph * & p : corrections_) {
        delete p; 
        p = nullptr;
    }
    corrections_.clear();
}

l1tpf::corrector::corrector(corrector && corr) :
        index_(std::move(corr.index_)),
        corrections_(std::move(corr.corrections_)),
        is2d_(corr.is2d_),
        neta_(corr.neta_),
        nemf_(corr.nemf_),
        emfMax_(corr.emfMax_) {
}

l1tpf::corrector & l1tpf::corrector::operator=(corrector && corr) {
    std::swap(is2d_, corr.is2d_);
    std::swap(neta_, corr.neta_);
    std::swap(nemf_, corr.nemf_);
    std::swap(emfMax_, corr.emfMax_);
   
    index_.swap(corr.index_); 
    corrections_.swap(corr.corrections_); 
    return *this;
}

float l1tpf::corrector::correctedPt(float pt, float emPt, float eta) { 
    float total = std::max(pt, emPt), abseta = std::abs(eta);
    float emf   = emPt/total;
    if (emfMax_ > 0 && emf > emfMax_) return total; // no correction
    unsigned int ieta = std::min(std::max<unsigned>(1,index_->GetXaxis()->FindBin(abseta)), neta_) - 1;
    unsigned int iemf = is2d_  && abseta < 3.0 ? std::min(std::max<unsigned>(1,index_->GetYaxis()->FindBin(emf)), nemf_) - 1 : 0;
    TGraph *graph = corrections_[ieta*nemf_ + iemf];
    if (!graph) {
        throw cms::Exception("RuntimeError") << "Error trying to read calibration for eta " << eta << " emf " <<  emf << " which are not available." << std::endl;
    }
    float ptcorr = std::min<float>(graph->Eval(total), 4*total);
    return ptcorr;
}

void l1tpf::corrector::correctPt(l1t::PFCluster & cluster, float preserveEmEt) {
    float newpt = correctedPt(cluster.pt(), cluster.emEt(), cluster.eta());
    cluster.calibratePt(newpt, preserveEmEt);
}


