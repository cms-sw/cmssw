#include "CalibTracker/SiStripLorentzAngle/interface/LA_Filler_Fitter.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <cmath>
#include <regex>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string/erase.hpp>
#include <TF1.h>
#include <TGraphErrors.h>
#include <TProfile.h>

LA_Filler_Fitter::Result LA_Filler_Fitter::result(Method m, const std::string name, const Book& book) {
  Result p;
  const std::string base = boost::erase_all_copy(name, method(m));

  const TH1* const h = book[name];
  const TH1* const reco = book[base + "_reconstruction"];
  const TH1* const field = book[base + "_field"];

  if (reco)
    p.reco = std::make_pair<float, float>(reco->GetMean(), reco->GetMeanError());
  if (field)
    p.field = field->GetMean();
  if (h) {
    switch (m) {
      case WIDTH: {
        const TF1* const f = h->GetFunction("LA_profile_fit");
        if (!f)
          break;
        p.measured = std::make_pair<float, float>(f->GetParameter(0), f->GetParError(0));
        p.chi2 = f->GetChisquare();
        p.ndof = f->GetNDF();
        p.entries = (unsigned)(h->GetEntries());
        break;
      }
      case PROB1:
      case AVGV2:
      case AVGV3:
      case RMSV2:
      case RMSV3: /*case MULTI:*/ {
        const TF1* const f = h->GetFunction("SymmetryFit");
        if (!f)
          break;
        p.measured = std::make_pair<float, float>(p.reco.first + f->GetParameter(0), f->GetParameter(1));
        p.chi2 = f->GetParameter(2);
        p.ndof = (unsigned)(f->GetParameter(3));
        p.entries = (m & PROB1)             ? (unsigned)book[base + "_w1"]->GetEntries()
                    : (m & (AVGV2 | RMSV2)) ? (unsigned)book[base + method(AVGV2, false)]->GetEntries()
                    : (m & (AVGV3 | RMSV3)) ? (unsigned)book[base + method(AVGV3, false)]->GetEntries()
                                            : 0;
        break;
      }
      default:
        break;
    }
  }
  return p;
}

std::map<uint32_t, LA_Filler_Fitter::Result> LA_Filler_Fitter::module_results(const Book& book, const Method m) {
  std::map<uint32_t, Result> results;
  for (Book::const_iterator it = book.begin(".*_module\\d*" + method(m)); it != book.end(); ++it) {
    const uint32_t detid = boost::lexical_cast<uint32_t>(
        std::regex_replace(it->first, std::regex(".*_module(\\d*)_.*"), std::string("\\1")));
    results[detid] = result(m, it->first, book);
  }
  return results;
}

std::map<std::string, LA_Filler_Fitter::Result> LA_Filler_Fitter::layer_results(const Book& book, const Method m) {
  std::map<std::string, Result> results;
  for (Book::const_iterator it = book.begin(".*layer\\d.*" + method(m)); it != book.end(); ++it) {
    const std::string name = boost::erase_all_copy(it->first, method(m));
    results[name] = result(m, it->first, book);
  }
  return results;
}

std::map<std::string, std::vector<LA_Filler_Fitter::Result> > LA_Filler_Fitter::ensemble_results(const Book& book,
                                                                                                 const Method m) {
  std::map<std::string, std::vector<Result> > results;
  for (Book::const_iterator it = book.begin(".*_sample.*" + method(m)); it != book.end(); ++it) {
    const std::string name = std::regex_replace(it->first, std::regex("sample\\d*_"), "");
    results[name].push_back(result(m, it->first, book));
  }
  return results;
}

void LA_Filler_Fitter::summarize_ensembles(Book& book) const {
  typedef std::map<std::string, std::vector<Result> > results_t;
  results_t results;
  for (int m = FIRST_METHOD; m <= LAST_METHOD; m <<= 1)
    if (methods_ & m) {
      results_t g = ensemble_results(book, (Method)(m));
      results.insert(g.begin(), g.end());
    }

  for (auto const& group : results) {
    const std::string name = group.first;
    for (auto const& p : group.second) {
      const float pad = (ensembleUp_ - ensembleLow_) / 10;
      book.fill(p.reco.first, name + "_ensembleReco", 12 * ensembleBins_, ensembleLow_ - pad, ensembleUp_ + pad);
      book.fill(p.measured.first, name + "_measure", 12 * ensembleBins_, ensembleLow_ - pad, ensembleUp_ + pad);
      book.fill(p.measured.second, name + "_merr", 500, 0, 0.01);
      book.fill((p.measured.first - p.reco.first) / p.measured.second, name + "_pull", 500, -10, 10);
    }
    //Need our own copy for thread safety
    TF1 gaus("mygaus", "gaus");
    book[name + "_measure"]->Fit(&gaus, "LLQ");
    book[name + "_merr"]->Fit(&gaus, "LLQ");
    book[name + "_pull"]->Fit(&gaus, "LLQ");
  }
}

std::map<std::string, std::vector<LA_Filler_Fitter::EnsembleSummary> > LA_Filler_Fitter::ensemble_summary(
    const Book& book) {
  std::map<std::string, std::vector<EnsembleSummary> > summary;
  for (Book::const_iterator it = book.begin(".*_ensembleReco"); it != book.end(); ++it) {
    const std::string base = boost::erase_all_copy(it->first, "_ensembleReco");

    const TH1* const reco = it->second;
    const TH1* const measure = book[base + "_measure"];
    const TH1* const merr = book[base + "_merr"];
    const TH1* const pull = book[base + "_pull"];

    EnsembleSummary s;
    s.samples = reco->GetEntries();
    s.truth = reco->GetMean();
    s.meanMeasured = std::make_pair<float, float>(measure->GetFunction("gaus")->GetParameter(1),
                                                  measure->GetFunction("gaus")->GetParError(1));
    s.sigmaMeasured = std::make_pair<float, float>(measure->GetFunction("gaus")->GetParameter(2),
                                                   measure->GetFunction("gaus")->GetParError(2));
    s.meanUncertainty = std::make_pair<float, float>(merr->GetFunction("gaus")->GetParameter(1),
                                                     merr->GetFunction("gaus")->GetParError(1));
    s.pull = std::make_pair<float, float>(pull->GetFunction("gaus")->GetParameter(2),
                                          pull->GetFunction("gaus")->GetParError(2));

    const std::string name = std::regex_replace(base, std::regex("ensembleBin\\d*_"), "");
    summary[name].push_back(s);
  }
  return summary;
}

std::pair<std::pair<float, float>, std::pair<float, float> > LA_Filler_Fitter::offset_slope(
    const std::vector<LA_Filler_Fitter::EnsembleSummary>& ensembles) {
  try {
    std::vector<float> x, y, xerr, yerr;
    for (auto const& ensemble : ensembles) {
      x.push_back(ensemble.truth);
      xerr.push_back(0);
      y.push_back(ensemble.meanMeasured.first);
      yerr.push_back(ensemble.meanMeasured.second);
    }
    TGraphErrors graph(x.size(), &(x[0]), &(y[0]), &(xerr[0]), &(yerr[0]));
    //Need our own copy for thread safety
    TF1 pol1("mypol1", "pol1");
    graph.Fit(&pol1);
    const TF1* const fit = graph.GetFunction("pol1");

    return std::make_pair(std::make_pair(fit->GetParameter(0), fit->GetParError(0)),
                          std::make_pair(fit->GetParameter(1), fit->GetParError(1)));
  } catch (edm::Exception const& e) {
    std::cerr << "Fitting Line Failed " << std::endl << e << std::endl;
    return std::make_pair(std::make_pair(0, 0), std::make_pair(0, 0));
  }
}

float LA_Filler_Fitter::pull(const std::vector<LA_Filler_Fitter::EnsembleSummary>& ensembles) {
  float p(0), w(0);
  for (auto const& ensemble : ensembles) {
    const float unc2 = pow(ensemble.pull.second, 2);
    p += ensemble.pull.first / unc2;
    w += 1 / unc2;
  }
  return p / w;
}

std::ostream& operator<<(std::ostream& strm, const LA_Filler_Fitter::Result& r) {
  return strm << r.reco.first << "\t" << r.reco.second << "\t" << r.measured.first << "\t" << r.measured.second << "\t"
              << r.calMeasured.first << "\t" << r.calMeasured.second << "\t" << r.field << "\t" << r.chi2 << "\t"
              << r.ndof << "\t" << r.entries;
}

std::ostream& operator<<(std::ostream& strm, const LA_Filler_Fitter::EnsembleSummary& e) {
  return strm << e.truth << "\t" << e.meanMeasured.first << "\t" << e.meanMeasured.second << "\t"
              << e.sigmaMeasured.first << "\t" << e.sigmaMeasured.second << "\t" << e.meanUncertainty.first << "\t"
              << e.meanUncertainty.second << "\t" << e.pull.first << "\t" << e.pull.second << "\t" << e.samples;
}
