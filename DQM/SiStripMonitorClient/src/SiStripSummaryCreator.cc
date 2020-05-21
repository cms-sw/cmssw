#include "DQM/SiStripMonitorClient/interface/SiStripSummaryCreator.h"
#include "DQM/SiStripMonitorClient/interface/SiStripConfigParser.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include <iostream>

SiStripSummaryCreator::SiStripSummaryCreator() {
  edm::LogInfo("SiStripSummaryCreator") << " Creating SiStripSummaryCreator "
                                        << "\n";
}

SiStripSummaryCreator::~SiStripSummaryCreator() {
  edm::LogInfo("SiStripSummaryCreator") << " Deleting SiStripSummaryCreator "
                                        << "\n";
}

bool SiStripSummaryCreator::readConfiguration(std::string const& file_path) {
  summaryMEs_.clear();
  SiStripConfigParser config_parser;
  config_parser.getDocument(edm::FileInPath(file_path).fullPath());
  if (!config_parser.getFrequencyForSummary(summaryFrequency_)) {
    std::cout << "SiStripSummaryCreator::readConfiguration: Failed to read Summary configuration parameters!! ";
    summaryFrequency_ = -1;
    return false;
  }
  if (!config_parser.getMENamesForSummary(summaryMEs_)) {
    std::cout << "SiStripSummaryCreator::readConfiguration: Failed to read Summary configuration parameters!! ";
    return false;
  }
  return true;
}

void SiStripSummaryCreator::setSummaryMENames(std::map<std::string, std::string>& me_names) {
  summaryMEs_.clear();
  for (std::map<std::string, std::string>::const_iterator isum = me_names.begin(); isum != me_names.end(); isum++) {
    summaryMEs_.insert(std::pair<std::string, std::string>(isum->first, isum->second));
  }
}

//
// -- Browse through the Folder Structure
//
void SiStripSummaryCreator::createSummary(DQMStore& dqm_store) {
  if (summaryMEs_.empty())
    return;
  std::string currDir = dqm_store.pwd();
  std::vector<std::string> subdirs = dqm_store.getSubdirs();
  int nmod = 0;
  for (std::vector<std::string>::const_iterator it = subdirs.begin(); it != subdirs.end(); it++) {
    if ((*it).find("module_") == std::string::npos)
      continue;
    nmod++;
  }
  if (nmod > 0) {
    fillSummaryHistos(dqm_store);
  } else {
    for (std::vector<std::string>::const_iterator it = subdirs.begin(); it != subdirs.end(); it++) {
      dqm_store.cd(*it);
      createSummary(dqm_store);
      dqm_store.goUp();
    }
    fillGrandSummaryHistos(dqm_store);
  }
}
//
// -- Create and Fill Summary Histograms at the lowest level of the structure
//
void SiStripSummaryCreator::fillSummaryHistos(DQMStore& dqm_store) {
  std::string currDir = dqm_store.pwd();
  std::map<std::string, MonitorElement*> MEMap;
  std::vector<std::string> subdirs = dqm_store.getSubdirs();
  if (subdirs.empty())
    return;

  for (std::map<std::string, std::string>::const_iterator isum = summaryMEs_.begin(); isum != summaryMEs_.end();
       isum++) {
    std::string name = isum->first;
    int iBinStep = 0;
    int ndet = 0;
    std::string htype = isum->second;
    for (std::vector<std::string>::const_iterator it = subdirs.begin(); it != subdirs.end(); it++) {
      if ((*it).find("module_") == std::string::npos)
        continue;
      dqm_store.cd(*it);
      ndet++;
      std::vector<MonitorElement*> contents = dqm_store.getContents(dqm_store.pwd());
      dqm_store.goUp();
      for (std::vector<MonitorElement*>::const_iterator im = contents.begin(); im != contents.end(); im++) {
        MonitorElement* me_i = (*im);
        if (!me_i)
          continue;
        std::string name_i = me_i->getName();
        if (name_i.find(name) == std::string::npos)
          continue;
        std::map<std::string, MonitorElement*>::iterator iPos = MEMap.find(name);
        MonitorElement* me;
        // Get the Summary ME
        if (iPos == MEMap.end()) {
          me = getSummaryME(dqm_store, name, htype);
          MEMap.insert(std::pair<std::string, MonitorElement*>(name, me));
        } else
          me = iPos->second;
        // Fill it now
        fillHistos(ndet, iBinStep, htype, me_i, me);
        iBinStep += me_i->getNbinsX();
        break;
      }
    }
  }
}

//
//  -- Fill Summary Histogram at higher level
//
void SiStripSummaryCreator::fillGrandSummaryHistos(DQMStore& dqm_store) {
  std::map<std::string, MonitorElement*> MEMap;
  std::string currDir = dqm_store.pwd();
  std::string dir_name = currDir.substr(currDir.find_last_of('/') + 1);
  if ((dir_name.find("SiStrip") == 0) || (dir_name.find("Collector") == 0) || (dir_name.find("MechanicalView") == 0) ||
      (dir_name.find("FU") == 0))
    return;
  std::vector<std::string> subdirs = dqm_store.getSubdirs();
  if (subdirs.empty())
    return;
  ;
  for (std::map<std::string, std::string>::const_iterator isum = summaryMEs_.begin(); isum != summaryMEs_.end();
       isum++) {
    std::string name, summary_name;
    name = isum->first;
    if (isum->second == "sum" || isum->second == "sum")
      summary_name = "Summary_" + isum->first;
    else
      summary_name = "Summary_Mean" + isum->first;
    std::string htype = isum->second;
    int ibinStep = 0;
    for (std::vector<std::string>::const_iterator it = subdirs.begin(); it != subdirs.end(); it++) {
      dqm_store.cd(*it);
      std::vector<MonitorElement*> contents = dqm_store.getContents(dqm_store.pwd());
      dqm_store.goUp();
      for (std::vector<MonitorElement*>::const_iterator im = contents.begin(); im != contents.end(); im++) {
        MonitorElement* me_i = (*im);
        if (!me_i)
          continue;
        std::string name_i = me_i->getName();
        if (name_i.find((summary_name)) != std::string::npos) {
          std::map<std::string, MonitorElement*>::iterator iPos = MEMap.find(name);
          MonitorElement* me;
          if (iPos == MEMap.end()) {
            if (htype == "sum" || htype == "Sum") {
              me = getSummaryME(dqm_store, name, htype);
            } else {
              me = getSummaryME(dqm_store, name, "bin-by-bin");
            }
            MEMap.insert(std::pair<std::string, MonitorElement*>(name, me));
          } else
            me = iPos->second;
          if (htype == "sum" || htype == "Sum") {
            fillHistos(0, ibinStep, htype, me_i, me);
          } else {
            fillHistos(0, ibinStep, "bin-by-bin", me_i, me);
          }
          ibinStep += me_i->getNbinsX();
          break;
        }
      }
    }
  }
}

SiStripSummaryCreator::MonitorElement* SiStripSummaryCreator::getSummaryME(DQMStore& dqm_store,
                                                                           std::string& name,
                                                                           std::string htype) {
  MonitorElement* me = nullptr;
  std::string currDir = dqm_store.pwd();
  std::string sum_name, tag_name;

  std::string dname = currDir.substr(currDir.find_last_of('/') + 1);
  if (dname.find('_') != std::string::npos)
    dname.insert(dname.find('_'), "_");
  if (htype == "sum" && htype == "Sum") {
    sum_name = "Summary" + name + "__" + dname;
    tag_name = "Summary" + name;
  } else {
    sum_name = "Summary_Mean" + name + "__" + dname;
    tag_name = "Summary_Mean" + name;
  }
  // If already booked
  std::vector<MonitorElement*> contents = dqm_store.getContents(currDir);
  for (std::vector<MonitorElement*>::const_iterator im = contents.begin(); im != contents.end(); im++) {
    MonitorElement* me = (*im);
    if (!me)
      continue;
    std::string me_name = me->getName();
    if (me_name.find(sum_name) == 0) {
      if (me->kind() == MonitorElement::Kind::TH1F || me->kind() == MonitorElement::Kind::TH2F ||
          me->kind() == MonitorElement::Kind::TPROFILE) {
        TH1* hist1 = me->getTH1();
        if (hist1) {
          hist1->Reset();
          return me;
        }
      }
    }
  }
  std::map<int, std::string> tags;
  if (!me) {
    int nBins = 0;
    std::vector<std::string> subdirs = dqm_store.getSubdirs();
    // set # of bins of the histogram
    if (htype == "mean" || htype == "Mean") {
      nBins = subdirs.size();
      me = dqm_store.book1D(sum_name, sum_name, nBins, 0.5, nBins + 0.5);
      int ibin = 0;
      for (auto const& subdir : subdirs) {
        std::string subdir_name = subdir.substr(subdir.find_last_of('/') + 1);
        ++ibin;
        tags.emplace(ibin, subdir_name);
      }
    } else if (htype == "bin-by-bin" || htype == "Bin-by-Bin") {
      for (auto const& subdir : subdirs) {
        dqm_store.cd(subdir);
        std::string subdir_name = subdir.substr(subdir.find_last_of('/') + 1);
        auto const& s_contents = dqm_store.getContents(dqm_store.pwd());
        for (auto const* s_me : s_contents) {
          if (!s_me)
            continue;
          std::string s_me_name = s_me->getName();
          if (s_me_name.find(name) == 0 || s_me_name.find(tag_name) == 0) {
            int ibin = s_me->getNbinsX();
            nBins += ibin;
            tags.emplace(nBins - ibin / 2, subdir_name);
            break;
          }
        }
        dqm_store.goUp();
      }
      me = dqm_store.book1D(sum_name, sum_name, nBins, 0.5, nBins + 0.5);
    } else if (htype == "sum" || htype == "Sum") {
      for (auto const& subdir : subdirs) {
        dqm_store.cd(subdir);
        auto const& s_contents = dqm_store.getContents(dqm_store.pwd());
        dqm_store.goUp();
        for (auto* s_me : s_contents) {
          if (!s_me)
            continue;
          std::string s_me_name = s_me->getName();
          if (s_me_name.find(name) == std::string::npos)
            continue;
          if (s_me->kind() == MonitorElement::Kind::TH1F) {
            TH1F* hist1 = s_me->getTH1F();
            if (hist1) {
              nBins = s_me->getNbinsX();
              me = dqm_store.book1D(
                  sum_name, sum_name, nBins, hist1->GetXaxis()->GetXmin(), hist1->GetXaxis()->GetXmax());
              break;
            }
          }
        }
      }
    }
  }
  // Set the axis title
  if (me && me->kind() == MonitorElement::Kind::TH1F && (htype != "sum" || htype != "Sum")) {
    TH1F* hist = me->getTH1F();
    if (hist) {
      if (name.find("NoisyStrips") != std::string::npos)
        hist->GetYaxis()->SetTitle("Noisy Strips (%)");
      else
        hist->GetYaxis()->SetTitle(name.c_str());

      for (auto const& [bin, label] : tags) {
        hist->GetXaxis()->SetBinLabel(bin, label.c_str());
      }
      hist->LabelsOption("uv");
    }
  }
  return me;
}

void SiStripSummaryCreator::fillHistos(
    int const ival, int const istep, std::string const htype, MonitorElement* me_src, MonitorElement* me) {
  if (me->getTH1()) {
    TH1F* hist1 = nullptr;
    TH2F* hist2 = nullptr;
    if (me->kind() == MonitorElement::Kind::TH1F)
      hist1 = me->getTH1F();
    if (me->kind() == MonitorElement::Kind::TH2F)
      hist2 = me->getTH2F();

    int nbins = me_src->getNbinsX();
    const std::string& name = me_src->getName();
    if (htype == "mean" || htype == "Mean") {
      if (hist2 && name.find("NoisyStrips") != std::string::npos) {
        float bad = 0.0;
        float entries = me_src->getEntries();
        if (entries > 0.0) {
          float binEntry = entries / nbins;
          for (int k = 1; k < nbins + 1; k++) {
            float noisy = me_src->getBinContent(k, 3) + me_src->getBinContent(k, 5);
            float dead = me_src->getBinContent(k, 2) + me_src->getBinContent(k, 4);
            if (noisy >= binEntry * 0.5 || dead >= binEntry * 0.5)
              bad++;
          }
          bad = bad * 100.0 / nbins;
          me->Fill(ival, bad);
        }
      } else
        me->Fill(ival, me_src->getMean());
    } else if (htype == "bin-by-bin" || htype == "Bin-by-Bin") {
      for (int k = 1; k < nbins + 1; k++) {
        me->setBinContent(istep + k, me_src->getBinContent(k));
      }
    } else if (htype == "sum" || htype == "Sum") {
      if (hist1) {
        for (int k = 1; k < nbins + 1; k++) {
          float val = me_src->getBinContent(k) + me->getBinContent(k);
          me->setBinContent(k, val);
        }
      }
    }
  }
}
