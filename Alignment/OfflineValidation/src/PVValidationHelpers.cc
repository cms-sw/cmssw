#include "Alignment/OfflineValidation/interface/PVValidationHelpers.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <cassert>
#include <numeric>
#include <algorithm>
#include <iostream>
#include "TMath.h"
#include "TH1.h"
#include "TF1.h"

//*************************************************************
void PVValHelper::add(std::map<std::string, TH1*>& h, TH1* hist)
//*************************************************************
{
  h[hist->GetName()] = hist;
  hist->StatOverflows(kTRUE);
}

//*************************************************************
void PVValHelper::fill(std::map<std::string, TH1*>& h, const std::string& s, double x)
//*************************************************************
{
  if (h.count(s) == 0) {
    edm::LogWarning("PVValidationHelpers") << "Trying to fill non-existing Histogram named " << s << std::endl;
    return;
  }
  h[s]->Fill(x);
}

//*************************************************************
void PVValHelper::fill(std::map<std::string, TH1*>& h, const std::string& s, double x, double y)
//*************************************************************
{
  if (h.count(s) == 0) {
    edm::LogWarning("PVValidationHelpers") << "Trying to fill non-existing Histogram named " << s << std::endl;
    return;
  }
  h[s]->Fill(x, y);
}

//*************************************************************
void PVValHelper::fillByIndex(std::vector<TH1F*>& h, unsigned int index, double x, std::string tag)
//*************************************************************
{
  assert(!h.empty());
  if (index < h.size()) {
    h[index]->Fill(x);
  } else {
    edm::LogWarning("PVValidationHelpers") << "Trying to fill non-existing Histogram with index " << index
                                           << " for array with size: " << h.size() << " tag: " << tag << std::endl;
    return;
  }
}

//*************************************************************
void PVValHelper::shrinkHistVectorToFit(std::vector<TH1F*>& h, unsigned int desired_size)
//*************************************************************
{
  h.erase(h.begin() + desired_size, h.end());
}

//*************************************************************
PVValHelper::plotLabels PVValHelper::getTypeString(PVValHelper::residualType type)
//*************************************************************
{
  PVValHelper::plotLabels returnType;
  switch (type) {
      // absoulte

    case PVValHelper::dxy:
      returnType = std::make_tuple("dxy", "d_{xy}", "[#mum]");
      break;
    case PVValHelper::dx:
      returnType = std::make_tuple("dx", "d_{x}", "[#mum]");
      break;
    case PVValHelper::dy:
      returnType = std::make_tuple("dy", "d_{y}", "[#mum]");
      break;
    case PVValHelper::dz:
      returnType = std::make_tuple("dz", "d_{z}", "[#mum]");
      break;
    case PVValHelper::IP2D:
      returnType = std::make_tuple("IP2D", "IP_{2D}", "[#mum]");
      break;
    case PVValHelper::resz:
      returnType = std::make_tuple("resz", "z_{trk}-z_{vtx}", "[#mum]");
      break;
    case PVValHelper::IP3D:
      returnType = std::make_tuple("IP3D", "IP_{3D}", "[#mum]");
      break;
    case PVValHelper::d3D:
      returnType = std::make_tuple("d3D", "d_{3D}", "[#mum]");
      break;

      // normalized

    case PVValHelper::norm_dxy:
      returnType = std::make_tuple("norm_dxy", "d_{xy}/#sigma_{d_{xy}}", "");
      break;
    case PVValHelper::norm_dx:
      returnType = std::make_tuple("norm_dx", "d_{x}/#sigma_{d_{x}}", "");
      break;
    case PVValHelper::norm_dy:
      returnType = std::make_tuple("norm_dy", "d_{y}/#sigma_{d_{y}}", "");
      break;
    case PVValHelper::norm_dz:
      returnType = std::make_tuple("norm_dz", "d_{z}/#sigma_{d_{z}}", "");
      break;
    case PVValHelper::norm_IP2D:
      returnType = std::make_tuple("norm_IP2D", "IP_{2D}/#sigma_{IP_{2D}}", "");
      break;
    case PVValHelper::norm_resz:
      returnType = std::make_tuple("norm_resz", "z_{trk}-z_{vtx}/#sigma_{res_{z}}", "");
      break;
    case PVValHelper::norm_IP3D:
      returnType = std::make_tuple("norm_IP3D", "IP_{3D}/#sigma_{IP_{3D}}", "");
      break;
    case PVValHelper::norm_d3D:
      returnType = std::make_tuple("norm_d3D", "d_{3D}/#sigma_{d_{3D}}", "");
      break;

    default:
      edm::LogWarning("PVValidationHelpers") << " getTypeString() unknown residual type: " << type << std::endl;
  }

  return returnType;
}

//*************************************************************
PVValHelper::plotLabels PVValHelper::getVarString(PVValHelper::plotVariable var)
//*************************************************************
{
  PVValHelper::plotLabels returnVar;
  switch (var) {
    case PVValHelper::phi:
      returnVar = std::make_tuple("phi", "#phi", "[rad]");
      break;
    case PVValHelper::eta:
      returnVar = std::make_tuple("eta", "#eta", "");
      break;
    case PVValHelper::pT:
      returnVar = std::make_tuple("pT", "p_{T}", "[GeV]");
      break;
    case PVValHelper::pTCentral:
      returnVar = std::make_tuple("pTCentral", "p_{T} |#eta|<1.", "[GeV]");
      break;
    case PVValHelper::ladder:
      returnVar = std::make_tuple("ladder", "ladder number", "");
      break;
    case PVValHelper::modZ:
      returnVar = std::make_tuple("modZ", "module number", "");
      break;
    default:
      edm::LogWarning("PVValidationHelpers") << " getVarString() unknown plot variable: " << var << std::endl;
  }

  return returnVar;
}

//*************************************************************
std::vector<float> PVValHelper::generateBins(int n, float start, float range)
//*************************************************************
{
  std::vector<float> v(n);
  float interval = range / (n - 1);
  std::iota(v.begin(), v.end(), 1.);

  //std::cout<<" interval:"<<interval<<std::endl;
  //for(float &a : v) { std::cout<< a << " ";  }
  //std::cout<< "\n";

  std::for_each(begin(v), end(v), [&](float& a) { a = start + ((a - 1) * interval); });

  return v;
}

//*************************************************************
Measurement1D PVValHelper::getMedian(TH1F* histo)
//*************************************************************
{
  double median = 0.;
  double q = 0.5;  // 0.5 quantile for "median"
  // protect against empty histograms
  if (histo->Integral() != 0) {
    histo->GetQuantiles(1, &median, &q);
  }

  Measurement1D result(median, median / TMath::Sqrt(histo->GetEntries()));

  return result;
}

//*************************************************************
Measurement1D PVValHelper::getMAD(TH1F* histo)
//*************************************************************
{
  int nbins = histo->GetNbinsX();
  double median = getMedian(histo).value();
  double x_lastBin = histo->GetBinLowEdge(nbins + 1);
  const char* HistoName = histo->GetName();
  std::string Finalname = "resMed_";
  Finalname.append(HistoName);
  TH1F* newHisto = new TH1F(Finalname.c_str(), Finalname.c_str(), nbins, 0., x_lastBin);
  double* residuals = new double[nbins];
  const float* weights = histo->GetArray();

  for (int j = 0; j < nbins; j++) {
    residuals[j] = std::abs(median - histo->GetBinCenter(j + 1));
    newHisto->Fill(residuals[j], weights[j]);
  }

  double theMAD = (PVValHelper::getMedian(newHisto).value()) * 1.4826;

  delete[] residuals;
  residuals = nullptr;
  newHisto->Delete("");

  Measurement1D result(theMAD, theMAD / histo->GetEntries());
  return result;
}

//*************************************************************
std::pair<Measurement1D, Measurement1D> PVValHelper::fitResiduals(TH1* hist)
//*************************************************************
{
  //float fitResult(9999);
  if (hist->GetEntries() < 1) {
    return std::make_pair(Measurement1D(0., 0.), Measurement1D(0., 0.));
  };

  float mean = hist->GetMean();
  float sigma = hist->GetRMS();

  TF1 func("tmp", "gaus", mean - 1.5 * sigma, mean + 1.5 * sigma);
  if (0 == hist->Fit(&func, "QNR")) {  // N: do not blow up file by storing fit!
    mean = func.GetParameter(1);
    sigma = func.GetParameter(2);
    // second fit: three sigma of first fit around mean of first fit
    func.SetRange(mean - 2 * sigma, mean + 2 * sigma);
    // I: integral gives more correct results if binning is too wide
    // L: Likelihood can treat empty bins correctly (if hist not weighted...)
    if (0 == hist->Fit(&func, "Q0LR")) {
      if (hist->GetFunction(func.GetName())) {  // Take care that it is later on drawn:
        hist->GetFunction(func.GetName())->ResetBit(TF1::kNotDraw);
      }
    }
  }

  float res_mean = func.GetParameter(1);
  float res_width = func.GetParameter(2);

  float res_mean_err = func.GetParError(1);
  float res_width_err = func.GetParError(2);

  Measurement1D resultM(res_mean, res_mean_err);
  Measurement1D resultW(res_width, res_width_err);

  std::pair<Measurement1D, Measurement1D> result;

  result = std::make_pair(resultM, resultW);
  return result;
}
