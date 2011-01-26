#include <stdexcept>
#include "HiggsAnalysis/CombinedLimit/interface/FeldmanCousins.h"
#include "HiggsAnalysis/CombinedLimit/interface/Combine.h"
#include <RooRealVar.h>
#include <RooArgSet.h>
#include <RooArgSet.h>
#include <RooWorkspace.h>
#include <RooDataHist.h>
#include <RooStats/ModelConfig.h>
#include <RooStats/FeldmanCousins.h>
#include <RooStats/PointSetInterval.h>

FeldmanCousins::FeldmanCousins() :
    LimitAlgo("FeldmanCousins specific options") 
{
    options_.add_options()
        ("lowerLimit", "Compute the lower limit instead of the upper limit")
        ("toysFactor", boost::program_options::value<float>()->default_value(1),  "Increase the toys per point by this factor w.r.t. the minimum from adaptive sampling")
    ;
}

void FeldmanCousins::applyOptions(const boost::program_options::variables_map &vm) 
{
    lowerLimit_ = vm.count("lowerLimit");
    rAbsAccuracy_ = vm.count("rAbsAcc") ? vm["rAbsAcc"].as<double>() : 0.1;
    rRelAccuracy_ = vm.count("rRelAcc") ? vm["rRelAcc"].as<double>() : 0.05;
    toysFactor_ = vm.count("toysFactor") ? vm["toysFactor"].as<float>() : 1.0;
}

bool FeldmanCousins::run(RooWorkspace *w, RooAbsData &data, double &limit, const double *hint) {
  RooRealVar *r = w->var("r");
  RooArgSet  poi(*r);

  if ((hint != 0) && (*hint > r->getMin())) {
    r->setMax(std::min<double>(3*(*hint), r->getMax()));
  }

  RooStats::ModelConfig modelConfig("sb_model", w);
  modelConfig.SetPdf(*w->pdf("model_s"));
  modelConfig.SetObservables(*w->set("observables"));
  modelConfig.SetParametersOfInterest(poi);
  if (withSystematics) modelConfig.SetNuisanceParameters(*w->set("nuisances"));
  modelConfig.SetSnapshot(poi);

  RooStats::FeldmanCousins fc(data, modelConfig);
  fc.FluctuateNumDataEntries(w->pdf("model_b")->canBeExtended());
  fc.UseAdaptiveSampling(true);
  fc.SetConfidenceLevel(cl);
  fc.AdditionalNToysFactor(toysFactor_);

  fc.SetNBins(10);
  double fcMid = 0, fcErr = 0; 
  do { 
      if (verbose > 1) std::cout << "scan in range [" << r->getMin() << ", " << r->getMax() << "]" << std::endl;
      std::auto_ptr<RooStats::PointSetInterval> fcInterval((RooStats::PointSetInterval *)fc.GetInterval());
      if (fcInterval.get() == 0) return false;
      if (verbose > 1) fcInterval->GetParameterPoints()->Print("V"); 
      RooDataHist* parameterScan = (RooDataHist*) fc.GetPointsToScan();
      int found = -1; 
      if (lowerLimit_) {
          for(int i=0; i<parameterScan->numEntries(); ++i){
              const RooArgSet * tmpPoint = parameterScan->get(i);
              if (!fcInterval->IsInInterval(*tmpPoint)) found = i;
              else break;
          }
      } else {
          for(int i=0; i<parameterScan->numEntries(); ++i){
              const RooArgSet * tmpPoint = parameterScan->get(i);
              bool inside = fcInterval->IsInInterval(*tmpPoint);
              if (inside) found = i;
              else if (found != -1) break;
          }
          if (found == -1) {
            if (verbose) std::cout << "Points are either all inside or all outside the bound." << std::endl;
            return false;
          }
      }
      double fcBefore = (found > -1 ? parameterScan->get(found)->getRealValue(r->GetName()) : r->getMin());
      double fcAfter  = (found < parameterScan->numEntries()-1 ? 
              parameterScan->get(found+1)->getRealValue(r->GetName()) : r->getMax());
      fcMid = 0.5*(fcAfter+fcBefore);
      fcErr = 0.5*(fcAfter-fcBefore);
      if (verbose > 0) std::cout << "  would be r < " << fcMid << " +/- "<<fcErr << std::endl;
      r->setMin(std::max(r->getMin(), fcMid-3*fcErr)); 
      r->setMax(std::min(r->getMax(), fcMid+3*fcErr));
      if (fcErr < 4*std::max(rAbsAccuracy_, rRelAccuracy_ * fcMid)) { // make last scan more precise
          fc.AdditionalNToysFactor(4*toysFactor_);
      }
  } while (fcErr > std::max(rAbsAccuracy_, rRelAccuracy_ * fcMid));

  limit = fcMid;
  if (verbose > -1) {
      std::cout << "\n -- FeldmanCousins++ -- \n";
      std::cout << "Limit: r " << (lowerLimit_ ? "> " : "< ") << fcMid << " +/- "<<fcErr << "\n";
  }
  return true;
}
