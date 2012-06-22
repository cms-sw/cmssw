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
 
   CMS_2011_S9088458() : Analysis("CMS_2011_S9088458") {
    setBeams(PROTON, PROTON);
    setNeedsCrossSection(true);
   }
 
 
// ======================================================== init

   void init() {
     FinalState fs;
     FastJets akt(fs, FastJets::ANTIKT, 0.5);
     addProjection(akt, "antikT");
     
     if(fuzzyEquals(sqrtS(), 7000*GeV, 1E-3)){ 

       _h_dijet = bookHistogram1D("dijet", binEdges(1, 1, 1)); //.reset (new LWH::Histogram1D(binEdges(1,1,1)));
       _h_trijet = bookHistogram1D("trijet", binEdges(1, 1, 1)); //.reset(new LWH::Histogram1D(binEdges(1,1,1)));
       _h_r32 = bookDataPointSet(1, 1, 1); 

     }
   }
 
 
// ======================================================== analyze

   void analyze(const Event & event) {
     const double weight = event.weight();

     const Jets& jets = applyProjection<JetAlg>(event, "antikT").jetsByPt();
     if (jets.size() < 2) vetoEvent;
     Jets highpT_jets;

     unsigned int i;
     for(i=0; i< jets.size(); i++) {
       if (jets[i].momentum().pT() > 50*GeV && fabs(jets[i].momentum().eta()) < 2.5) highpT_jets.push_back(jets[i]);
     }

     if (highpT_jets.size() < 2) vetoEvent;

     double HT = 0;
     for(i=0; i< highpT_jets.size(); i++) 
       HT = HT + highpT_jets[i].momentum().pT();
       

     if (highpT_jets.size() >= 2) _h_dijet->fill(HT/TeV, weight) ;
     if (highpT_jets.size() >= 3) _h_trijet->fill(HT/TeV, weight) ;


   }
 
// ======================================================== finalize 
 
   void finalize() {
     vector<double> yval_R32, yerr_R32;
     for (size_t i = 0;  i < 30; ++i) {
        const double yval = _h_trijet->binHeight(i) / _h_dijet->binHeight(i);
        yval_R32.push_back(yval);
        const double yerr = sqrt(_h_dijet->binError(i)*_h_dijet->binError(i)/(_h_dijet->binHeight(i) * _h_dijet->binHeight(i)) +
                                 _h_trijet->binError(i)*_h_trijet->binError(i)/(_h_trijet->binHeight(i) * _h_trijet->binHeight(i))) * yval;
        yerr_R32.push_back(yerr);                         
      }
      _h_r32->setCoordinate(1, yval_R32, yerr_R32); 
   }
 


    private:

    AIDA::IHistogram1D* _h_dijet, *_h_trijet;
    AIDA::IDataPointSet* _h_r32;


 }; 
  
   // This global object acts as a hook for the plugin system
   AnalysisBuilder<CMS_2011_S9088458> plugin_CMS_2011_S9088458;
 
}


