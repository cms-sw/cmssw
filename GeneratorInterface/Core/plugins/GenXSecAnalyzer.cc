#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TMath.h"
#include <iostream>
#include <iomanip>

// analyzer of a summary information product on filter efficiency for a user specified path
// meant for the generator filter efficiency calculation

// system include files
#include <memory>
#include <vector>
#include <map>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/GeneratorProducts/interface/GenLumiInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenFilterInfo.h"
#include "SimDataFormats/GeneratorProducts/interface/LHERunInfoProduct.h"

#include "FWCore/Framework/interface/MakerMacros.h"

//
// class declaration
//
namespace gxsec {
  struct LumiCache {};
  struct RunCache {
    RunCache()
        : product_(-9999),
          filterOnlyEffRun_(0, 0, 0, 0, 0., 0., 0., 0.),
          hepMCFilterEffRun_(0, 0, 0, 0, 0., 0., 0., 0.) {}
    // for weight before GenFilter and HepMCFilter and before matching
    CMS_THREAD_GUARD(GenXSecAnalyzer::mutex_) mutable double thisRunWeightPre_ = 0;

    // for weight after GenFilter and HepMCFilter and after matching
    CMS_THREAD_GUARD(GenXSecAnalyzer::mutex_) mutable double thisRunWeight_ = 0;

    // GenLumiInfo before HepMCFilter and GenFilter, this is used
    // for computation
    CMS_THREAD_GUARD(GenXSecAnalyzer::mutex_) mutable GenLumiInfoProduct product_;

    // statistics from additional generator filter, for computation
    // reset for each run
    CMS_THREAD_GUARD(GenXSecAnalyzer::mutex_) mutable GenFilterInfo filterOnlyEffRun_;

    // statistics from HepMC filter, for computation
    CMS_THREAD_GUARD(GenXSecAnalyzer::mutex_) mutable GenFilterInfo hepMCFilterEffRun_;

    // the following vectors all have the same size
    // LHE or Pythia/Herwig cross section of previous luminosity block
    // vector size = number of processes, used for computation
    CMS_THREAD_GUARD(GenXSecAnalyzer::mutex_) mutable std::map<int, GenLumiInfoProduct::XSec> previousLumiBlockLHEXSec_;

    // LHE or Pythia/Herwig combined cross section of current luminosity block
    // updated for each luminosity block, initialized in every run
    // used for computation
    CMS_THREAD_GUARD(GenXSecAnalyzer::mutex_) mutable std::map<int, GenLumiInfoProduct::XSec> currentLumiBlockLHEXSec_;
  };
}  // namespace gxsec

class GenXSecAnalyzer
    : public edm::global::EDAnalyzer<edm::RunCache<gxsec::RunCache>, edm::LuminosityBlockCache<gxsec::LumiCache>> {
public:
  explicit GenXSecAnalyzer(const edm::ParameterSet &);
  ~GenXSecAnalyzer() override;

private:
  void beginJob() final;
  std::shared_ptr<gxsec::RunCache> globalBeginRun(edm::Run const &, edm::EventSetup const &) const final;
  std::shared_ptr<gxsec::LumiCache> globalBeginLuminosityBlock(edm::LuminosityBlock const &,
                                                               edm::EventSetup const &) const final;
  void analyze(edm::StreamID, const edm::Event &, const edm::EventSetup &) const final;
  void globalEndLuminosityBlock(edm::LuminosityBlock const &, edm::EventSetup const &) const final;
  void globalEndRun(edm::Run const &, edm::EventSetup const &) const final;
  void endJob() final;
  // computation of cross section after matching and before HepcFilter and GenFilter
  GenLumiInfoProduct::XSec compute(const GenLumiInfoProduct &) const;
  // combination of cross section from different MCs after matching (could be either before or after HepcFilter and GenFilter)
  void combine(GenLumiInfoProduct::XSec &, double &, const GenLumiInfoProduct::XSec &, const double &) const;
  void combine(double &, double &, double &, const double &, const double &, const double &) const;

  edm::EDGetTokenT<GenFilterInfo> genFilterInfoToken_;
  edm::EDGetTokenT<GenFilterInfo> hepMCFilterInfoToken_;
  edm::EDGetTokenT<GenLumiInfoProduct> genLumiInfoToken_;
  edm::EDGetTokenT<LHERunInfoProduct> lheRunInfoToken_;

  // ----------member data --------------------------

  mutable std::atomic<int> nMCs_;

  mutable std::atomic<int> hepidwtup_;

  mutable std::mutex mutex_;

  // for weight before GenFilter and HepMCFilter and before matching
  CMS_THREAD_GUARD(mutex_) mutable double totalWeightPre_;

  // for weight after GenFilter and HepMCFilter and after matching
  CMS_THREAD_GUARD(mutex_) mutable double totalWeight_;

  // combined cross sections before HepMCFilter and GenFilter
  CMS_THREAD_GUARD(mutex_) mutable GenLumiInfoProduct::XSec xsecPreFilter_;

  // final combined cross sections
  CMS_THREAD_GUARD(mutex_) mutable GenLumiInfoProduct::XSec xsec_;

  // statistics from additional generator filter, for print-out only
  CMS_THREAD_GUARD(mutex_) mutable GenFilterInfo filterOnlyEffStat_;

  // statistics from HepMC filter, for print-out only
  CMS_THREAD_GUARD(mutex_) mutable GenFilterInfo hepMCFilterEffStat_;

  // the vector/map size is the number of LHE processes + 1
  // needed only for printouts, not used for computation
  // only printed out when combining the same physics process
  // uncertainty-averaged cross sections before matching
  CMS_THREAD_GUARD(mutex_) mutable std::vector<GenLumiInfoProduct::XSec> xsecBeforeMatching_;
  // uncertainty-averaged cross sections after matching
  CMS_THREAD_GUARD(mutex_) mutable std::vector<GenLumiInfoProduct::XSec> xsecAfterMatching_;
  // statistics from jet matching
  CMS_THREAD_GUARD(mutex_) mutable std::map<int, GenFilterInfo> jetMatchEffStat_;
};

GenXSecAnalyzer::GenXSecAnalyzer(const edm::ParameterSet &iConfig)
    : nMCs_(0),
      hepidwtup_(-9999),
      totalWeightPre_(0),
      totalWeight_(0),
      xsecPreFilter_(-1, -1),
      xsec_(-1, -1),
      filterOnlyEffStat_(0, 0, 0, 0, 0., 0., 0., 0.),
      hepMCFilterEffStat_(0, 0, 0, 0, 0., 0., 0., 0.) {
  genFilterInfoToken_ = consumes<GenFilterInfo, edm::InLumi>(edm::InputTag("genFilterEfficiencyProducer", ""));
  hepMCFilterInfoToken_ = consumes<GenFilterInfo, edm::InLumi>(edm::InputTag("generator", ""));
  genLumiInfoToken_ = consumes<GenLumiInfoProduct, edm::InLumi>(edm::InputTag("generator", ""));
  lheRunInfoToken_ = consumes<LHERunInfoProduct, edm::InRun>(edm::InputTag("externalLHEProducer", ""));
}

GenXSecAnalyzer::~GenXSecAnalyzer() {}

void GenXSecAnalyzer::beginJob() {}

std::shared_ptr<gxsec::RunCache> GenXSecAnalyzer::globalBeginRun(edm::Run const &iRun, edm::EventSetup const &) const {
  // initialization for every different physics MC

  nMCs_++;

  {
    std::lock_guard l{mutex_};
    xsecBeforeMatching_.clear();
    xsecAfterMatching_.clear();
    jetMatchEffStat_.clear();
  }
  return std::make_shared<gxsec::RunCache>();
}

std::shared_ptr<gxsec::LumiCache> GenXSecAnalyzer::globalBeginLuminosityBlock(edm::LuminosityBlock const &iLumi,
                                                                              edm::EventSetup const &) const {
  return std::shared_ptr<gxsec::LumiCache>();
}

void GenXSecAnalyzer::analyze(edm::StreamID, const edm::Event &, const edm::EventSetup &) const {}

void GenXSecAnalyzer::globalEndLuminosityBlock(edm::LuminosityBlock const &iLumi, edm::EventSetup const &) const {
  edm::Handle<GenLumiInfoProduct> genLumiInfo;
  iLumi.getByToken(genLumiInfoToken_, genLumiInfo);
  if (!genLumiInfo.isValid())
    return;
  hepidwtup_ = genLumiInfo->getHEPIDWTUP();

  std::vector<GenLumiInfoProduct::ProcessInfo> const &theProcesses = genLumiInfo->getProcessInfos();

  unsigned int theProcesses_size = theProcesses.size();

  // if it's a pure parton-shower generator, check there should be only one element in thisProcessInfos
  // the error of lheXSec is -1
  if (hepidwtup_ == -1) {
    if (theProcesses_size != 1) {
      edm::LogError("GenXSecAnalyzer::endLuminosityBlock") << "Pure parton shower has thisProcessInfos size!=1";
      return;
    }
  }

  for (unsigned int ip = 0; ip < theProcesses_size; ip++) {
    if (theProcesses[ip].lheXSec().value() < 0) {
      edm::LogError("GenXSecAnalyzer::endLuminosityBlock")
          << "cross section of process " << ip << " value = " << theProcesses[ip].lheXSec().value();
      return;
    }
  }

  auto runC = runCache(iLumi.getRun().index());
  {
    std::lock_guard g{mutex_};
    runC->product_.mergeProduct(*genLumiInfo);
  }
  edm::Handle<GenFilterInfo> genFilter;
  iLumi.getByToken(genFilterInfoToken_, genFilter);

  if (genFilter.isValid()) {
    std::lock_guard g{mutex_};
    filterOnlyEffStat_.mergeProduct(*genFilter);
    runC->filterOnlyEffRun_.mergeProduct(*genFilter);
    runC->thisRunWeight_ += genFilter->sumPassWeights();
  }

  edm::Handle<GenFilterInfo> hepMCFilter;
  iLumi.getByToken(hepMCFilterInfoToken_, hepMCFilter);

  if (hepMCFilter.isValid()) {
    std::lock_guard g{mutex_};
    hepMCFilterEffStat_.mergeProduct(*hepMCFilter);
    runC->hepMCFilterEffRun_.mergeProduct(*hepMCFilter);
  }

  std::lock_guard g{mutex_};
  // doing generic summing for jet matching statistics
  // and computation of combined LHE information
  for (unsigned int ip = 0; ip < theProcesses_size; ip++) {
    int id = theProcesses[ip].process();
    GenFilterInfo &x = jetMatchEffStat_[id];
    GenLumiInfoProduct::XSec &y = runC->currentLumiBlockLHEXSec_[id];
    GenLumiInfoProduct::FinalStat temp_killed = theProcesses[ip].killed();
    GenLumiInfoProduct::FinalStat temp_selected = theProcesses[ip].selected();
    double passw = temp_killed.sum();
    double passw2 = temp_killed.sum2();
    double totalw = temp_selected.sum();
    double totalw2 = temp_selected.sum2();
    GenFilterInfo tempInfo(theProcesses[ip].nPassPos(),
                           theProcesses[ip].nPassNeg(),
                           theProcesses[ip].nTotalPos(),
                           theProcesses[ip].nTotalNeg(),
                           passw,
                           passw2,
                           totalw,
                           totalw2);

    // matching statistics for all processes
    jetMatchEffStat_[10000].mergeProduct(tempInfo);
    double currentValue = theProcesses[ip].lheXSec().value();
    double currentError = theProcesses[ip].lheXSec().error();

    // this process ID has occurred before
    auto &thisRunWeightPre = runC->thisRunWeightPre_;
    if (y.value() > 0) {
      x.mergeProduct(tempInfo);
      double previousValue = runC->previousLumiBlockLHEXSec_[id].value();

      if (currentValue != previousValue)  // transition of cross section
      {
        double xsec = y.value();
        double err = y.error();
        combine(xsec, err, thisRunWeightPre, currentValue, currentError, totalw);
        y = GenLumiInfoProduct::XSec(xsec, err);
      } else  // LHE cross section is the same as previous lumiblock
        thisRunWeightPre += totalw;

    }
    // this process ID has never occurred before
    else {
      x = tempInfo;
      y = theProcesses[ip].lheXSec();
      thisRunWeightPre += totalw;
    }

    runC->previousLumiBlockLHEXSec_[id] = theProcesses[ip].lheXSec();
  }  // end

  return;
}

void GenXSecAnalyzer::globalEndRun(edm::Run const &iRun, edm::EventSetup const &) const {
  //xsection before matching
  edm::Handle<LHERunInfoProduct> run;

  if (iRun.getByToken(lheRunInfoToken_, run)) {
    const lhef::HEPRUP thisHeprup = run->heprup();

    for (unsigned int iSize = 0; iSize < thisHeprup.XSECUP.size(); iSize++) {
      std::cout << std::setw(14) << std::fixed << thisHeprup.XSECUP[iSize] << std::setw(14) << std::fixed
                << thisHeprup.XERRUP[iSize] << std::setw(14) << std::fixed << thisHeprup.XMAXUP[iSize] << std::setw(14)
                << std::fixed << thisHeprup.LPRUP[iSize] << std::endl;
    }
    std::cout << " " << std::endl;
  }

  auto runC = runCache(iRun.index());
  std::lock_guard l{mutex_};

  // compute cross section for this run first
  // set the correct combined LHE+filter cross sections
  unsigned int i = 0;
  std::vector<GenLumiInfoProduct::ProcessInfo> newInfos;
  for (std::map<int, GenLumiInfoProduct::XSec>::const_iterator iter = runC->currentLumiBlockLHEXSec_.begin();
       iter != runC->currentLumiBlockLHEXSec_.end();
       ++iter, i++) {
    GenLumiInfoProduct::ProcessInfo temp = runC->product_.getProcessInfos()[i];
    temp.setLheXSec(iter->second.value(), iter->second.error());
    newInfos.push_back(temp);
  }
  runC->product_.setProcessInfo(newInfos);

  const GenLumiInfoProduct::XSec thisRunXSecPre = compute(runC->product_);
  // xsection after matching before filters
  combine(xsecPreFilter_, totalWeightPre_, thisRunXSecPre, runC->thisRunWeightPre_);

  double thisHepFilterEff = 1;
  double thisHepFilterErr = 0;

  if (runC->hepMCFilterEffRun_.sumWeights2() > 0) {
    thisHepFilterEff = runC->hepMCFilterEffRun_.filterEfficiency(hepidwtup_);
    thisHepFilterErr = runC->hepMCFilterEffRun_.filterEfficiencyError(hepidwtup_);
    if (thisHepFilterEff < 0) {
      thisHepFilterEff = 1;
      thisHepFilterErr = 0;
    }
  }

  double thisGenFilterEff = 1;
  double thisGenFilterErr = 0;

  if (runC->filterOnlyEffRun_.sumWeights2() > 0) {
    thisGenFilterEff = runC->filterOnlyEffRun_.filterEfficiency(hepidwtup_);
    thisGenFilterErr = runC->filterOnlyEffRun_.filterEfficiencyError(hepidwtup_);
    if (thisGenFilterEff < 0) {
      thisGenFilterEff = 1;
      thisGenFilterErr = 0;
    }
  }
  double thisXsec = thisRunXSecPre.value() > 0 ? thisHepFilterEff * thisGenFilterEff * thisRunXSecPre.value() : 0;
  double thisErr =
      thisRunXSecPre.value() > 0
          ? thisXsec * sqrt(pow(TMath::Max(thisRunXSecPre.error(), (double)0) / thisRunXSecPre.value(), 2) +
                            pow(thisHepFilterErr / thisHepFilterEff, 2) + pow(thisGenFilterErr / thisGenFilterEff, 2))
          : 0;
  const GenLumiInfoProduct::XSec thisRunXSec = GenLumiInfoProduct::XSec(thisXsec, thisErr);
  combine(xsec_, totalWeight_, thisRunXSec, runC->thisRunWeight_);
}

void GenXSecAnalyzer::combine(double &finalValue,
                              double &finalError,
                              double &finalWeight,
                              const double &currentValue,
                              const double &currentError,
                              const double &currentWeight) const {
  if (finalValue <= 0) {
    finalValue = currentValue;
    finalError = currentError;
    finalWeight += currentWeight;
  } else {
    double wgt1 = (finalError <= 0 || currentError <= 0) ? finalWeight : 1 / (finalError * finalError);
    double wgt2 = (finalError <= 0 || currentError <= 0) ? currentWeight : 1 / (currentError * currentError);
    double xsec = (wgt1 * finalValue + wgt2 * currentValue) / (wgt1 + wgt2);
    double err = (finalError <= 0 || currentError <= 0) ? 0 : 1.0 / std::sqrt(wgt1 + wgt2);
    finalValue = xsec;
    finalError = err;
    finalWeight += currentWeight;
  }
  return;
}

void GenXSecAnalyzer::combine(GenLumiInfoProduct::XSec &finalXSec,
                              double &totalw,
                              const GenLumiInfoProduct::XSec &thisRunXSec,
                              const double &thisw) const {
  double value = finalXSec.value();
  double error = finalXSec.error();
  double thisValue = thisRunXSec.value();
  double thisError = thisRunXSec.error();
  combine(value, error, totalw, thisValue, thisError, thisw);
  finalXSec = GenLumiInfoProduct::XSec(value, error);
  return;
}

GenLumiInfoProduct::XSec GenXSecAnalyzer::compute(const GenLumiInfoProduct &iLumiInfo) const {
  // sum of cross sections and errors over different processes
  double sigSelSum = 0.0;
  double err2SelSum = 0.0;

  std::vector<GenLumiInfoProduct::XSec> tempVector_before;
  std::vector<GenLumiInfoProduct::XSec> tempVector_after;

  // loop over different processes for each sample
  unsigned int vectorSize = iLumiInfo.getProcessInfos().size();
  for (unsigned int ip = 0; ip < vectorSize; ip++) {
    GenLumiInfoProduct::ProcessInfo proc = iLumiInfo.getProcessInfos()[ip];
    double hepxsec_value = proc.lheXSec().value();
    double hepxsec_error = proc.lheXSec().error() <= 0 ? 0 : proc.lheXSec().error();
    tempVector_before.push_back(GenLumiInfoProduct::XSec(hepxsec_value, hepxsec_error));

    sigSelSum += hepxsec_value;
    err2SelSum += hepxsec_error * hepxsec_error;

    // skips computation if jet matching efficiency=0
    if (proc.killed().n() < 1) {
      tempVector_after.push_back(GenLumiInfoProduct::XSec(0.0, 0.0));
      continue;
    }

    // computing jet matching efficiency for this process
    double fracAcc = 0;
    double ntotal = proc.nTotalPos() - proc.nTotalNeg();
    double npass = proc.nPassPos() - proc.nPassNeg();
    switch (hepidwtup_) {
      case 3:
      case -3:
        fracAcc = ntotal > 0 ? npass / ntotal : -1;
        break;
      default:
        fracAcc = proc.selected().sum() > 0 ? proc.killed().sum() / proc.selected().sum() : -1;
        break;
    }

    if (fracAcc <= 0) {
      tempVector_after.push_back(GenLumiInfoProduct::XSec(0.0, 0.0));
      continue;
    }

    // cross section after matching for this particular process
    double sigmaFin = hepxsec_value * fracAcc;

    // computing error on jet matching efficiency
    double relErr = 1.0;
    double efferr2 = 0;
    switch (hepidwtup_) {
      case 3:
      case -3: {
        double ntotal_pos = proc.nTotalPos();
        double effp = ntotal_pos > 0 ? (double)proc.nPassPos() / ntotal_pos : 0;
        double effp_err2 = ntotal_pos > 0 ? (1 - effp) * effp / ntotal_pos : 0;

        double ntotal_neg = proc.nTotalNeg();
        double effn = ntotal_neg > 0 ? (double)proc.nPassNeg() / ntotal_neg : 0;
        double effn_err2 = ntotal_neg > 0 ? (1 - effn) * effn / ntotal_neg : 0;

        efferr2 = ntotal > 0
                      ? (ntotal_pos * ntotal_pos * effp_err2 + ntotal_neg * ntotal_neg * effn_err2) / ntotal / ntotal
                      : 0;
        break;
      }
      default: {
        double denominator = pow(proc.selected().sum(), 4);
        double passw = proc.killed().sum();
        double passw2 = proc.killed().sum2();
        double failw = proc.selected().sum() - passw;
        double failw2 = proc.selected().sum2() - passw2;
        double numerator = (passw2 * failw * failw + failw2 * passw * passw);

        efferr2 = denominator > 0 ? numerator / denominator : 0;
        break;
      }
    }
    double delta2Veto = efferr2 / fracAcc / fracAcc;

    // computing total error on cross section after matching efficiency

    double sigma2Sum, sigma2Err;
    sigma2Sum = hepxsec_value * hepxsec_value;
    sigma2Err = hepxsec_error * hepxsec_error;

    double delta2Sum = delta2Veto + sigma2Err / sigma2Sum;
    relErr = (delta2Sum > 0.0 ? std::sqrt(delta2Sum) : 0.0);
    double deltaFin = sigmaFin * relErr;

    tempVector_after.push_back(GenLumiInfoProduct::XSec(sigmaFin, deltaFin));

  }  // end of loop over different processes
  tempVector_before.push_back(GenLumiInfoProduct::XSec(sigSelSum, sqrt(err2SelSum)));

  double total_matcheff = jetMatchEffStat_[10000].filterEfficiency(hepidwtup_);
  double total_matcherr = jetMatchEffStat_[10000].filterEfficiencyError(hepidwtup_);

  double xsec_after = sigSelSum * total_matcheff;
  double xsecerr_after = (total_matcheff > 0 && sigSelSum > 0)
                             ? xsec_after * sqrt(err2SelSum / sigSelSum / sigSelSum +
                                                 total_matcherr * total_matcherr / total_matcheff / total_matcheff)
                             : 0;

  GenLumiInfoProduct::XSec result(xsec_after, xsecerr_after);
  tempVector_after.push_back(result);

  xsecBeforeMatching_ = tempVector_before;
  xsecAfterMatching_ = tempVector_after;

  return result;
}

void GenXSecAnalyzer::endJob() {
  edm::LogPrint("GenXSecAnalyzer") << "\n"
                                   << "------------------------------------"
                                   << "\n"
                                   << "GenXsecAnalyzer:"
                                   << "\n"
                                   << "------------------------------------";

  if (jetMatchEffStat_.empty()) {
    edm::LogPrint("GenXSecAnalyzer") << "------------------------------------"
                                     << "\n"
                                     << "Cross-section summary not available"
                                     << "\n"
                                     << "------------------------------------";
    return;
  }

  // fraction of negative weights
  double final_fract_neg_w = 0;
  double final_fract_neg_w_unc = 0;

  // below print out is only for combination of same physics MC samples and ME+Pythia MCs

  if (nMCs_ == 1 && hepidwtup_ != -1) {
    edm::LogPrint("GenXSecAnalyzer")
        << "-----------------------------------------------------------------------------------------------------------"
           "--------------------------------------------------------------- \n"
        << "Overall cross-section summary \n"
        << "-----------------------------------------------------------------------------------------------------------"
           "---------------------------------------------------------------";
    edm::LogPrint("GenXSecAnalyzer") << "Process\t\txsec_before [pb]\t\tpassed\tnposw\tnnegw\ttried\tnposw\tnnegw "
                                        "\txsec_match [pb]\t\t\taccepted [%]\t event_eff [%]";

    const unsigned sizeOfInfos = jetMatchEffStat_.size();
    const unsigned last = sizeOfInfos - 1;
    std::string *title = new std::string[sizeOfInfos];
    unsigned int i = 0;
    double jetmatch_eff = 0;
    double jetmatch_err = 0;
    double matching_eff = 1;
    double matching_efferr = 1;

    for (std::map<int, GenFilterInfo>::const_iterator iter = jetMatchEffStat_.begin(); iter != jetMatchEffStat_.end();
         ++iter, i++) {
      GenFilterInfo thisJetMatchStat = iter->second;
      GenFilterInfo thisEventEffStat =
          GenFilterInfo(thisJetMatchStat.numPassPositiveEvents() + thisJetMatchStat.numPassNegativeEvents(),
                        0,
                        thisJetMatchStat.numTotalPositiveEvents() + thisJetMatchStat.numTotalNegativeEvents(),
                        0,
                        thisJetMatchStat.numPassPositiveEvents() + thisJetMatchStat.numPassNegativeEvents(),
                        thisJetMatchStat.numPassPositiveEvents() + thisJetMatchStat.numPassNegativeEvents(),
                        thisJetMatchStat.numTotalPositiveEvents() + thisJetMatchStat.numTotalNegativeEvents(),
                        thisJetMatchStat.numTotalPositiveEvents() + thisJetMatchStat.numTotalNegativeEvents());

      jetmatch_eff = thisJetMatchStat.filterEfficiency(hepidwtup_);
      jetmatch_err = thisJetMatchStat.filterEfficiencyError(hepidwtup_);

      if (i == last) {
        title[i] = "Total";

        edm::LogPrint("GenXSecAnalyzer")
            << "-------------------------------------------------------------------------------------------------------"
               "------------------------------------------------------------------- ";

        // fill negative fraction of negative weights and uncertainty after matching
        final_fract_neg_w = thisEventEffStat.numEventsPassed() > 0
                                ? thisJetMatchStat.numPassNegativeEvents() / thisEventEffStat.numEventsPassed()
                                : 0;
        final_fract_neg_w_unc =
            thisJetMatchStat.numPassNegativeEvents() > 0
                ? final_fract_neg_w * final_fract_neg_w / thisEventEffStat.numEventsPassed() *
                      sqrt(thisJetMatchStat.numPassPositiveEvents() * thisJetMatchStat.numPassPositiveEvents() /
                               thisJetMatchStat.numPassNegativeEvents() +
                           thisJetMatchStat.numPassPositiveEvents())
                : 0;
      } else {
        title[i] = Form("%d", i);
      }

      edm::LogPrint("GenXSecAnalyzer") << title[i] << "\t\t" << std::scientific << std::setprecision(3)
                                       << xsecBeforeMatching_[i].value() << " +/- " << xsecBeforeMatching_[i].error()
                                       << "\t\t" << thisEventEffStat.numEventsPassed() << "\t"
                                       << thisJetMatchStat.numPassPositiveEvents() << "\t"
                                       << thisJetMatchStat.numPassNegativeEvents() << "\t"
                                       << thisEventEffStat.numEventsTotal() << "\t"
                                       << thisJetMatchStat.numTotalPositiveEvents() << "\t"
                                       << thisJetMatchStat.numTotalNegativeEvents() << "\t" << std::scientific
                                       << std::setprecision(3) << xsecAfterMatching_[i].value() << " +/- "
                                       << xsecAfterMatching_[i].error() << "\t\t" << std::fixed << std::setprecision(1)
                                       << (jetmatch_eff * 100) << " +/- " << (jetmatch_err * 100) << "\t" << std::fixed
                                       << std::setprecision(1) << (thisEventEffStat.filterEfficiency(+3) * 100)
                                       << " +/- " << (thisEventEffStat.filterEfficiencyError(+3) * 100);

      matching_eff = thisEventEffStat.filterEfficiency(+3);
      matching_efferr = thisEventEffStat.filterEfficiencyError(+3);
    }
    delete[] title;

    edm::LogPrint("GenXSecAnalyzer")
        << "-----------------------------------------------------------------------------------------------------------"
           "---------------------------------------------------------------";

    edm::LogPrint("GenXSecAnalyzer") << "Before matching: total cross section = " << std::scientific
                                     << std::setprecision(3) << xsecBeforeMatching_[last].value() << " +- "
                                     << xsecBeforeMatching_[last].error() << " pb";

    edm::LogPrint("GenXSecAnalyzer") << "After matching: total cross section = " << std::scientific
                                     << std::setprecision(3) << xsecAfterMatching_[last].value() << " +- "
                                     << xsecAfterMatching_[last].error() << " pb";

    edm::LogPrint("GenXSecAnalyzer") << "Matching efficiency = " << std::fixed << std::setprecision(1) << matching_eff
                                     << " +/- " << matching_efferr << "   [TO BE USED IN MCM]";

  } else if (hepidwtup_ == -1)
    edm::LogPrint("GenXSecAnalyzer") << "Before Filter: total cross section = " << std::scientific
                                     << std::setprecision(3) << xsecPreFilter_.value() << " +- "
                                     << xsecPreFilter_.error() << " pb";

  // hepMC filter efficiency
  double hepMCFilter_eff = 1.0;
  double hepMCFilter_err = 0.0;
  if (hepMCFilterEffStat_.sumWeights2() > 0) {
    hepMCFilter_eff = hepMCFilterEffStat_.filterEfficiency(-1);
    hepMCFilter_err = hepMCFilterEffStat_.filterEfficiencyError(-1);
    edm::LogPrint("GenXSecAnalyzer") << "HepMC filter efficiency (taking into account weights)= "
                                     << "(" << hepMCFilterEffStat_.sumPassWeights() << ")"
                                     << " / "
                                     << "(" << hepMCFilterEffStat_.sumWeights() << ")"
                                     << " = " << std::scientific << std::setprecision(3) << hepMCFilter_eff << " +- "
                                     << hepMCFilter_err;

    double hepMCFilter_event_total =
        hepMCFilterEffStat_.numTotalPositiveEvents() + hepMCFilterEffStat_.numTotalNegativeEvents();
    double hepMCFilter_event_pass =
        hepMCFilterEffStat_.numPassPositiveEvents() + hepMCFilterEffStat_.numPassNegativeEvents();
    double hepMCFilter_event_eff = hepMCFilter_event_total > 0 ? hepMCFilter_event_pass / hepMCFilter_event_total : 0;
    double hepMCFilter_event_err =
        hepMCFilter_event_total > 0
            ? sqrt((1 - hepMCFilter_event_eff) * hepMCFilter_event_eff / hepMCFilter_event_total)
            : -1;
    edm::LogPrint("GenXSecAnalyzer") << "HepMC filter efficiency (event-level)= "
                                     << "(" << hepMCFilter_event_pass << ")"
                                     << " / "
                                     << "(" << hepMCFilter_event_total << ")"
                                     << " = " << std::scientific << std::setprecision(3) << hepMCFilter_event_eff
                                     << " +- " << hepMCFilter_event_err;
  }

  // gen-particle filter efficiency
  if (filterOnlyEffStat_.sumWeights2() > 0) {
    double filterOnly_eff = filterOnlyEffStat_.filterEfficiency(-1);
    double filterOnly_err = filterOnlyEffStat_.filterEfficiencyError(-1);

    edm::LogPrint("GenXSecAnalyzer") << "Filter efficiency (taking into account weights)= "
                                     << "(" << filterOnlyEffStat_.sumPassWeights() << ")"
                                     << " / "
                                     << "(" << filterOnlyEffStat_.sumWeights() << ")"
                                     << " = " << std::scientific << std::setprecision(3) << filterOnly_eff << " +- "
                                     << filterOnly_err;

    double filterOnly_event_total =
        filterOnlyEffStat_.numTotalPositiveEvents() + filterOnlyEffStat_.numTotalNegativeEvents();
    double filterOnly_event_pass =
        filterOnlyEffStat_.numPassPositiveEvents() + filterOnlyEffStat_.numPassNegativeEvents();
    double filterOnly_event_eff = filterOnly_event_total > 0 ? filterOnly_event_pass / filterOnly_event_total : 0;
    double filterOnly_event_err = filterOnly_event_total > 0
                                      ? sqrt((1 - filterOnly_event_eff) * filterOnly_event_eff / filterOnly_event_total)
                                      : -1;
    edm::LogPrint("GenXSecAnalyzer") << "Filter efficiency (event-level)= "
                                     << "(" << filterOnly_event_pass << ")"
                                     << " / "
                                     << "(" << filterOnly_event_total << ")"
                                     << " = " << std::scientific << std::setprecision(3) << filterOnly_event_eff
                                     << " +- " << filterOnly_event_err << "    [TO BE USED IN MCM]";

    // fill negative fraction of negative weights and uncertainty after filter
    final_fract_neg_w =
        filterOnly_event_pass > 0 ? filterOnlyEffStat_.numPassNegativeEvents() / (filterOnly_event_pass) : 0;
    final_fract_neg_w_unc =
        filterOnlyEffStat_.numPassNegativeEvents() > 0
            ? final_fract_neg_w * final_fract_neg_w / filterOnly_event_pass *
                  sqrt(filterOnlyEffStat_.numPassPositiveEvents() * filterOnlyEffStat_.numPassPositiveEvents() /
                           filterOnlyEffStat_.numPassNegativeEvents() +
                       filterOnlyEffStat_.numPassPositiveEvents())
            : 0;
  }

  edm::LogPrint("GenXSecAnalyzer") << "\nAfter filter: final cross section = " << std::scientific
                                   << std::setprecision(3) << xsec_.value() << " +- " << xsec_.error() << " pb";

  edm::LogPrint("GenXSecAnalyzer") << "After filter: final fraction of events with negative weights = "
                                   << std::scientific << std::setprecision(3) << final_fract_neg_w << " +- "
                                   << final_fract_neg_w_unc;

  // L=[N*(1-2f)^2]/s
  double lumi_1M_evts =
      xsec_.value() > 0 ? 1e6 * (1 - 2 * final_fract_neg_w) * (1 - 2 * final_fract_neg_w) / xsec_.value() / 1e3 : 0;
  double lumi_1M_evts_unc =
      xsec_.value() > 0 ? (1 - 2 * final_fract_neg_w) * lumi_1M_evts *
                              sqrt(1e-6 + 16 * pow(final_fract_neg_w_unc, 2) / pow(1 - 2 * final_fract_neg_w, 2) +
                                   pow(xsec_.error() / xsec_.value(), 2))
                        : 0;
  edm::LogPrint("GenXSecAnalyzer") << "After filter: final equivalent lumi for 1M events (1/fb) = " << std::scientific
                                   << std::setprecision(3) << lumi_1M_evts << " +- " << lumi_1M_evts_unc;
}

DEFINE_FWK_MODULE(GenXSecAnalyzer);
