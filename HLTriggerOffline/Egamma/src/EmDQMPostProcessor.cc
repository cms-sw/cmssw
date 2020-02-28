#include "HLTriggerOffline/Egamma/interface/EmDQMPostProcessor.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>

#include <TEfficiency.h>

//----------------------------------------------------------------------

EmDQMPostProcessor::EmDQMPostProcessor(const edm::ParameterSet &pset) {
  subDir_ = pset.getUntrackedParameter<std::string>("subDir");

  dataSet_ = pset.getUntrackedParameter<std::string>("dataSet", "unknown");

  noPhiPlots = pset.getUntrackedParameter<bool>("noPhiPlots", true);

  normalizeToReco = pset.getUntrackedParameter<bool>("normalizeToReco", false);

  ignoreEmpty = pset.getUntrackedParameter<bool>("ignoreEmpty", true);
}

//----------------------------------------------------------------------

void EmDQMPostProcessor::dqmEndJob(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter) {
  // go to the directory to be processed
  if (igetter.dirExists(subDir_))
    ibooker.cd(subDir_);
  else {
    edm::LogWarning("EmDQMPostProcessor") << "cannot find directory: " << subDir_ << " , skipping";
    return;
  }

  //--------------------
  // with respect to what are (some) efficiencies calculated ?
  std::string shortReferenceName;
  if (normalizeToReco)
    shortReferenceName = "RECO";
  else
    shortReferenceName = "gen";
  //--------------------

  //////////////////////////////////
  // loop over all triggers/samples//
  //////////////////////////////////

  // store dataset name (if defined) in output file
  // DQMStore:bookString seems to put a key in the file which is
  // of the form <dataSet>s=....</dataSet> which is not very convenient
  // (it points to a null pointer, one would have to loop over
  // all keys of the corresponding directory in the ROOT file
  // and check whether it is of the desired form and then parse
  // it from this string...).
  //
  // So we store the name of the dataset as the title of a histogram,
  // which is much easier to access...
  // TH1D *dataSetNameHisto =
  ibooker.book1D("DataSetNameHistogram", dataSet_, 1, 0, 1);

  std::vector<std::string> subdirectories = igetter.getSubdirs();
  ////////////////////////////////////////////////////////
  // Do everything twice: once for mc-matched histos,   //
  // once for unmatched histos                          //
  ////////////////////////////////////////////////////////

  std::vector<std::string> postfixes;
  postfixes.push_back("");               // unmatched histograms
  postfixes.push_back("_RECO_matched");  // for data
  // we put this on the list even when we're running on
  // data (where there is no generator information).
  // The first test in the loop will then fail and
  // the iteration is skipped.
  postfixes.push_back("_MC_matched");

  std::vector<TProfile *> allElePaths;
  int nEle = 0;
  int nPhoton = 0;

  // find the number of electron and photon paths
  for (std::vector<std::string>::iterator dir = subdirectories.begin(); dir != subdirectories.end(); ++dir) {
    if (dir->find("Ele") != std::string::npos || dir->find("_SC") != std::string::npos)
      ++nEle;
    else if (dir->find("Photon") != std::string::npos)
      ++nPhoton;
  }

  std::vector<TProfile *> allPhotonPaths;
  for (std::vector<std::string>::iterator postfix = postfixes.begin(); postfix != postfixes.end(); postfix++) {
    bool pop = false;
    int elePos = 1;
    int photonPos = 1;

    /////////////////////////////////////
    // computer per-event efficiencies //
    /////////////////////////////////////

    std::string histoName = "efficiency_by_step" + *postfix;
    std::string baseName = "total_eff" + *postfix;

    std::string allEleHistoName = "EfficiencyByPath_Ele" + *postfix;
    std::string allEleHistoLabel = "Efficiency_for_each_validated_electron_path" + *postfix;
    allElePaths.push_back(
        new TProfile(allEleHistoName.c_str(), allEleHistoLabel.c_str(), nEle, 0., (double)nEle, 0., 1.2));
    std::string allPhotonHistoName = "EfficiencyByPath_Photon" + *postfix;
    std::string allPhotonHistoLabel = "Efficiency_for_each_validated_photon_path" + *postfix;
    allPhotonPaths.push_back(
        new TProfile(allPhotonHistoName.c_str(), allPhotonHistoLabel.c_str(), nPhoton, 0., (double)nPhoton, 0., 1.2));

    for (std::vector<std::string>::iterator dir = subdirectories.begin(); dir != subdirectories.end(); dir++) {
      ibooker.cd(*dir);

      // get the current trigger name
      std::string trigName = dir->substr(dir->rfind("/") + 1);
      trigName = trigName.replace(trigName.rfind("_DQM"), 4, "");

      // Get the gen-level (or reco, for data) plots
      std::string genName;

      // only generate efficiency plots if there are generated/recoed events
      // selected but label the bin in the overview plot, even if the bin is
      // empty
      if (ignoreEmpty) {
        if (normalizeToReco)
          genName = ibooker.pwd() + "/reco_et";
        else
          genName = ibooker.pwd() + "/gen_et";
        TH1F *genHist = getHistogram(ibooker, igetter, genName);
        if (genHist->GetEntries() == 0) {
          edm::LogInfo("EmDQMPostProcessor")
              << "Zero events were selected for path '" << trigName << "'. No efficiency plots will be generated.";
          if (trigName.find("Ele") != std::string::npos || trigName.find("_SC") != std::string::npos) {
            allElePaths.back()->GetXaxis()->SetBinLabel(elePos, trigName.c_str());
            ++elePos;
          } else if (trigName.find("Photon") != std::string::npos) {
            allPhotonPaths.back()->GetXaxis()->SetBinLabel(photonPos, trigName.c_str());
            ++photonPos;
          }
          ibooker.goUp();
          continue;
        }
      }

      TH1F *basehist = getHistogram(ibooker, igetter, ibooker.pwd() + "/" + baseName);
      if (basehist == nullptr) {
        // edm::LogWarning("EmDQMPostProcessor") << "histogram " <<
        // (ibooker.pwd() + "/" + baseName) << " does not exist, skipping
        // postfix
        // '" << *postfix << "'";
        pop = true;
        ibooker.goUp();
        continue;
      }
      // at least one histogram with postfix was found
      pop = false;

      ibooker.goUp();
      MonitorElement *meTotal = ibooker.bookProfile(trigName + "__" + histoName,
                                                    trigName + "__" + histoName,
                                                    basehist->GetXaxis()->GetNbins(),
                                                    basehist->GetXaxis()->GetXmin(),
                                                    basehist->GetXaxis()->GetXmax(),
                                                    0.,
                                                    1.2);
      meTotal->setEfficiencyFlag();
      TProfile *total = meTotal->getTProfile();
      ibooker.cd(*dir);
      total->GetXaxis()->SetBinLabel(1, basehist->GetXaxis()->GetBinLabel(1));

      //       std::vector<std::string> mes = igetter.getMEs();
      //       for(std::vector<std::string>::iterator me = mes.begin() ;me!=
      //       mes.end(); me++ )
      // 	std::cout <<*me <<std::endl;
      //       std::cout <<std::endl;

      double value = 0;
      double errorh = 0, errorl = 0, error = 0;
      // compute stepwise total efficiencies
      for (int bin = total->GetNbinsX() - 2; bin > 1; bin--) {
        value = 0;
        errorl = 0;
        errorh = 0;
        error = 0;
        if (basehist->GetBinContent(bin - 1) != 0) {
          Efficiency(
              (int)basehist->GetBinContent(bin), (int)basehist->GetBinContent(bin - 1), 0.683, value, errorl, errorh);
          error = value - errorl > errorh - value ? value - errorl : errorh - value;
        }
        total->SetBinContent(bin, value);
        total->SetBinEntries(bin, 1);
        total->SetBinError(bin, sqrt(value * value + error * error));
        total->GetXaxis()->SetBinLabel(bin, basehist->GetXaxis()->GetBinLabel(bin));
      }

      // set first bin to L1 efficiency
      if (basehist->GetBinContent(basehist->GetNbinsX()) != 0) {
        Efficiency((int)basehist->GetBinContent(1),
                   (int)basehist->GetBinContent(basehist->GetNbinsX()),
                   0.683,
                   value,
                   errorl,
                   errorh);
        error = value - errorl > errorh - value ? value - errorl : errorh - value;
      } else {
        value = 0;
        error = 0;
      }
      total->SetBinContent(1, value);
      total->SetBinEntries(1, 1);
      total->SetBinError(1, sqrt(value * value + error * error));

      // total efficiency relative to gen or reco
      if (basehist->GetBinContent(basehist->GetNbinsX()) != 0) {
        Efficiency((int)basehist->GetBinContent(basehist->GetNbinsX() - 2),
                   (int)basehist->GetBinContent(basehist->GetNbinsX()),
                   0.683,
                   value,
                   errorl,
                   errorh);
        error = value - errorl > errorh - value ? value - errorl : errorh - value;
      } else {
        value = 0;
        error = 0;
      }
      total->SetBinContent(total->GetNbinsX(), value);
      total->SetBinEntries(total->GetNbinsX(), 1);
      total->SetBinError(total->GetNbinsX(), sqrt(value * value + error * error));
      total->GetXaxis()->SetBinLabel(total->GetNbinsX(), ("total efficiency rel. " + shortReferenceName).c_str());

      // total efficiency relative to L1
      if (basehist->GetBinContent(1) != 0) {
        Efficiency((int)basehist->GetBinContent(basehist->GetNbinsX() - 2),
                   (int)basehist->GetBinContent(1),
                   0.683,
                   value,
                   errorl,
                   errorh);
        error = value - errorl > errorh - value ? value - errorl : errorh - value;
      } else {
        value = 0;
        error = 0;
      }
      total->SetBinContent(total->GetNbinsX() - 1, value);
      total->SetBinError(total->GetNbinsX() - 1, sqrt(value * value + error * error));
      total->SetBinEntries(total->GetNbinsX() - 1, 1);
      total->GetXaxis()->SetBinLabel(total->GetNbinsX() - 1, "total efficiency rel. L1");

      //----------------------------------------

      ///////////////////////////////////////////
      // compute per-object efficiencies       //
      ///////////////////////////////////////////
      // MonitorElement *eff, *num, *denom, *genPlot, *effVsGen, *effL1VsGen;
      std::vector<std::string> varNames;
      varNames.push_back("et");
      varNames.push_back("eta");

      if (!noPhiPlots) {
        varNames.push_back("phi");
      }
      varNames.push_back("etaphi");

      std::string filterName;
      std::string filterName2;
      std::string denomName;
      std::string numName;

      // Get the L1 over gen filter first
      filterName2 = total->GetXaxis()->GetBinLabel(1);
      // loop over variables (eta/phi/et)
      for (std::vector<std::string>::iterator var = varNames.begin(); var != varNames.end(); var++) {
        numName = ibooker.pwd() + "/" + filterName2 + *var + *postfix;

        if (normalizeToReco)
          genName = ibooker.pwd() + "/reco_" + *var;
        else
          genName = ibooker.pwd() + "/gen_" + *var;

        if ((*var).find("etaphi") != std::string::npos) {
          if (!dividehistos2D(ibooker,
                              igetter,
                              numName,
                              genName,
                              "efficiency_" + filterName2 + "_vs_" + *var + *postfix,
                              *var,
                              "eff. of" + filterName2 + " vs " + *var + *postfix))
            break;
        } else if (!dividehistos(ibooker,
                                 igetter,
                                 numName,
                                 genName,
                                 "efficiency_" + filterName2 + "_vs_" + *var + *postfix,
                                 *var,
                                 "eff. of" + filterName2 + " vs " + *var + *postfix))
          break;
      }  // loop over variables

      // get the filter names from the bin-labels of the master-histogram
      for (int filter = 1; filter < total->GetNbinsX() - 2; filter++) {
        filterName = total->GetXaxis()->GetBinLabel(filter);
        filterName2 = total->GetXaxis()->GetBinLabel(filter + 1);

        // loop over variables (eta/et/phi)
        for (std::vector<std::string>::iterator var = varNames.begin(); var != varNames.end(); var++) {
          numName = ibooker.pwd() + "/" + filterName2 + *var + *postfix;
          denomName = ibooker.pwd() + "/" + filterName + *var + *postfix;

          // Is this the last filter? Book efficiency vs gen (or reco, for data)
          // level
          std::string temp = *postfix;
          if (filter == total->GetNbinsX() - 3 && temp.find("matched") != std::string::npos) {
            if (normalizeToReco)
              genName = ibooker.pwd() + "/reco_" + *var;
            else
              genName = ibooker.pwd() + "/gen_" + *var;

            if ((*var).find("etaphi") != std::string::npos) {
              if (!dividehistos2D(ibooker,
                                  igetter,
                                  numName,
                                  genName,
                                  "final_eff_vs_" + *var,
                                  *var,
                                  "Efficiency Compared to " + shortReferenceName + " vs " + *var))
                break;
            } else if (!dividehistos(ibooker,
                                     igetter,
                                     numName,
                                     genName,
                                     "final_eff_vs_" + *var,
                                     *var,
                                     "Efficiency Compared to " + shortReferenceName + " vs " + *var))
              break;
          }

          if ((*var).find("etaphi") != std::string::npos) {
            if (!dividehistos2D(ibooker,
                                igetter,
                                numName,
                                denomName,
                                "efficiency_" + filterName2 + "_vs_" + *var + *postfix,
                                *var,
                                "efficiency_" + filterName2 + "_vs_" + *var + *postfix))
              break;
          } else if (!dividehistos(ibooker,
                                   igetter,
                                   numName,
                                   denomName,
                                   "efficiency_" + filterName2 + "_vs_" + *var + *postfix,
                                   *var,
                                   "efficiency_" + filterName2 + "_vs_" + *var + *postfix))
            break;

        }  // loop over variables
      }    // loop over monitoring modules within path

      ibooker.goUp();

      // fill overall efficiency histograms
      double totCont = total->GetBinContent(total->GetNbinsX());
      double totErr = total->GetBinError(total->GetNbinsX());
      if (trigName.find("Ele") != std::string::npos || trigName.find("_SC") != std::string::npos) {
        allElePaths.back()->SetBinContent(elePos, totCont);
        allElePaths.back()->SetBinEntries(elePos, 1);
        allElePaths.back()->SetBinError(elePos, sqrt(totCont * totCont + totErr * totErr));
        allElePaths.back()->GetXaxis()->SetBinLabel(elePos, trigName.c_str());
        ++elePos;
      } else if (trigName.find("Photon") != std::string::npos) {
        allPhotonPaths.back()->SetBinContent(photonPos, totCont);
        allPhotonPaths.back()->SetBinEntries(photonPos, 1);
        allPhotonPaths.back()->SetBinError(photonPos, sqrt(totCont * totCont + totErr * totErr));
        allPhotonPaths.back()->GetXaxis()->SetBinLabel(photonPos, trigName.c_str());
        ++photonPos;
      }

    }  // loop over dirs
    if (pop) {
      allElePaths.pop_back();
      allPhotonPaths.pop_back();
    } else {
      allElePaths.back()->GetXaxis()->SetLabelSize(0.03);
      allPhotonPaths.back()->GetXaxis()->SetLabelSize(0.03);
      ibooker.bookProfile(allEleHistoName, allElePaths.back())->setEfficiencyFlag();
      ibooker.bookProfile(allPhotonHistoName, allPhotonPaths.back())->setEfficiencyFlag();
    }
  }  // loop over postfixes
}

//----------------------------------------------------------------------

TProfile *EmDQMPostProcessor::dividehistos(DQMStore::IBooker &ibooker,
                                           DQMStore::IGetter &igetter,
                                           const std::string &numName,
                                           const std::string &denomName,
                                           const std::string &outName,
                                           const std::string &label,
                                           const std::string &titel) {
  // std::cout << numName <<std::endl;

  TH1F *num = getHistogram(ibooker, igetter, numName);

  // std::cout << denomName << std::endl;
  TH1F *denom = getHistogram(ibooker, igetter, denomName);

  if (num == nullptr)
    edm::LogWarning("EmDQMPostProcessor") << "numerator histogram " << numName << " does not exist";

  if (denom == nullptr)
    edm::LogWarning("EmDQMPostProcessor") << "denominator histogram " << denomName << " does not exist";

  // Check if histograms actually exist

  if (num == nullptr || denom == nullptr)
    return nullptr;

  MonitorElement *meOut = ibooker.bookProfile(
      outName, titel, num->GetXaxis()->GetNbins(), num->GetXaxis()->GetXmin(), num->GetXaxis()->GetXmax(), 0., 1.2);
  meOut->setEfficiencyFlag();
  TProfile *out = meOut->getTProfile();
  out->GetXaxis()->SetTitle(label.c_str());
  out->SetYTitle("Efficiency");
  out->SetOption("PE");
  out->SetLineColor(2);
  out->SetLineWidth(2);
  out->SetMarkerStyle(20);
  out->SetMarkerSize(0.8);
  out->SetStats(kFALSE);
  for (int i = 1; i <= num->GetNbinsX(); i++) {
    double e, low, high;
    Efficiency((int)num->GetBinContent(i), (int)denom->GetBinContent(i), 0.683, e, low, high);
    double err = e - low > high - e ? e - low : high - e;
    // here is the trick to store info in TProfile:
    out->SetBinContent(i, e);
    out->SetBinEntries(i, 1);
    out->SetBinError(i, sqrt(e * e + err * err));
  }

  return out;
}
//----------------------------------------------------------------------
TH2F *EmDQMPostProcessor::dividehistos2D(DQMStore::IBooker &ibooker,
                                         DQMStore::IGetter &igetter,
                                         const std::string &numName,
                                         const std::string &denomName,
                                         const std::string &outName,
                                         const std::string &label,
                                         const std::string &titel) {
  // std::cout << numName <<std::endl;
  TH2F *num = get2DHistogram(ibooker, igetter, numName);
  // std::cout << denomName << std::endl;
  TH2F *denom = get2DHistogram(ibooker, igetter, denomName);
  if (num == nullptr)
    edm::LogWarning("EmDQMPostProcessor") << "2D numerator histogram " << numName << " does not exist";

  if (denom == nullptr)
    edm::LogWarning("EmDQMPostProcessor") << "2D denominator histogram " << denomName << " does not exist";

  // Check if histograms actually exist
  if (num == nullptr || denom == nullptr)
    return nullptr;

  MonitorElement *meOut = ibooker.book2D(outName,
                                         titel,
                                         num->GetXaxis()->GetNbins(),
                                         num->GetXaxis()->GetXmin(),
                                         num->GetXaxis()->GetXmax(),
                                         num->GetYaxis()->GetNbins(),
                                         num->GetYaxis()->GetXmin(),
                                         num->GetYaxis()->GetXmax());
  TH2F *out = meOut->getTH2F();
  out->Add(num);
  out->Divide(denom);
  out->GetXaxis()->SetTitle(label.c_str());
  out->SetYTitle("#phi");
  out->SetXTitle("#eta");
  out->SetOption("COLZ");
  out->SetStats(kFALSE);

  return out;
}
//--------------------
TH1F *EmDQMPostProcessor::getHistogram(DQMStore::IBooker &ibooker,
                                       DQMStore::IGetter &igetter,
                                       const std::string &histoPath) {
  MonitorElement *monElement = igetter.get(histoPath);
  if (monElement != nullptr)
    return monElement->getTH1F();
  else
    return nullptr;
}
//----------------------------------------------------------------------
TH2F *EmDQMPostProcessor::get2DHistogram(DQMStore::IBooker &ibooker,
                                         DQMStore::IGetter &igetter,
                                         const std::string &histoPath) {
  MonitorElement *monElement = igetter.get(histoPath);
  if (monElement != nullptr)
    return monElement->getTH2F();
  else
    return nullptr;
}

//----------------------------------------------------------------------

void EmDQMPostProcessor::Efficiency(
    int passing, int total, double level, double &mode, double &lowerBound, double &upperBound) {
  // protection (see also TGraphAsymmErrors::Efficiency(..), mimick the old
  // behaviour )
  if (total == 0) {
    mode = 0.5;
    lowerBound = 0;
    upperBound = 1;
    return;
  }

  mode = passing / ((double)total);

  // note that the order of the two first arguments ('total' and 'passed') is
  // the opposited with respect to the earlier TGraphAsymmErrors::Efficiency(..)
  // method
  //
  // see http://root.cern.ch/root/html/TEfficiency.html#compare for
  // an illustration of the coverage of the different methods provided by
  // TEfficiency

  lowerBound = TEfficiency::Wilson(total, passing, level, false);
  upperBound = TEfficiency::Wilson(total, passing, level, true);
}

//----------------------------------------------------------------------

DEFINE_FWK_MODULE(EmDQMPostProcessor);

//----------------------------------------------------------------------
