// -*- C++ -*-
#include "Rivet/Analysis.hh"
#include "Rivet/Tools/Logging.hh"
#include "Rivet/Tools/BinnedHistogram.hh"
#include "Rivet/Projections/FinalState.hh"
#include "Rivet/Projections/FastJets.hh"
#include "Rivet/RivetAIDA.hh"
#include "LWH/Histogram1D.h"

namespace Rivet {


   // CMS Ratio of the 3-jet to 2-jet Cross Sections by Tomo
   class CMS_2011_S9088458 : public Analysis {
   public:

     CMS_2011_S9088458() : Analysis("CMS_2011_S9088458") {}


     void init() {
       FinalState fs;
       FastJets akt(fs, FastJets::ANTIKT, 0.5);
       addProjection(akt, "antikT");

       _h_dijet = bookHistogram1D("dijet", binEdges(1, 1, 1));
       _h_trijet = bookHistogram1D("trijet", binEdges(1, 1, 1));
       _h_r32 = bookDataPointSet(1, 1, 1);
     }


     void analyze(const Event & event) {
       const double weight = event.weight();

       Jets highpT_jets;
       double HT = 0;
       foreach(const Jet & jet, applyProjection<JetAlg>(event, "antikT").jetsByPt(50.0*GeV)) {
         if (fabs(jet.momentum().eta()) < 2.5) {
           highpT_jets.push_back(jet);
           HT += jet.momentum().pT();
         }
       }
       if (highpT_jets.size() < 2) vetoEvent;

       if (highpT_jets.size() >= 2) _h_dijet->fill(HT/TeV, weight) ;
       if (highpT_jets.size() >= 3) _h_trijet->fill(HT/TeV, weight) ;
     }


     void finalize() {
       vector<double> yval_R32, yerr_R32;
       for (size_t i = 0;  i < 30; ++i) {
         double yval, yerr;
         if (_h_dijet->binHeight(i)==0.0 || _h_trijet->binHeight(i)==0.0) {
           yval = 0.0;
           yerr = 0.0;
         }
         else {
           yval =  _h_trijet->binHeight(i)/_h_dijet->binHeight(i);
           yerr = sqrt(_h_dijet->binError(i)*_h_dijet->binError(i)/(_h_dijet->binHeight(i) * _h_dijet->binHeight(i)) +
                       _h_trijet->binError(i)*_h_trijet->binError(i)/(_h_trijet->binHeight(i) * _h_trijet->binHeight(i))) * yval;
         }
         yval_R32.push_back(yval);
         yerr_R32.push_back(yerr);
       }
       _h_r32->setCoordinate(1, yval_R32, yerr_R32);
       histogramFactory().destroy(_h_dijet);
       histogramFactory().destroy(_h_trijet);
     }


   private:

     AIDA::IHistogram1D *_h_dijet, *_h_trijet;
     AIDA::IDataPointSet *_h_r32;

  };

  // This global object acts as a hook for the plugin system
//AK  DECLARE_RIVET_PLUGIN(CMS_2011_S9088458);
  AnalysisBuilder<CMS_2011_S9088458> plugin_CMS_2011_S9088458;

}


