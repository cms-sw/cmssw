// -*- C++ -*-
#include "Rivet/Analysis.hh"
#include "Rivet/RivetAIDA.hh"
#include "Rivet/Tools/Logging.hh"
#include "Rivet/Projections/FinalState.hh"
#include "Rivet/Projections/FastJets.hh"
#include "EventShape_rivet.hh"

const int njetptmn=5; // Threshold of leading pt
double leadingPtThreshold[njetptmn]={110.0, 170.0, 250.0, 320.0, 390.0};
using namespace std;
using namespace CLHEP;
/// @todo Include more projections as required, e.g. ChargedFinalState, FastJets, ZFinder...

namespace Rivet {


  class CMS_SMP_12_022 : public Analysis {
  public:

    /// @name Constructors etc.

    /// Constructor
    CMS_SMP_12_022()
      : Analysis("CMS_SMP_12_022")
    {    }

  public:
    
    /// @name Analysis methods
    
    /// Book histograms and initialise projections before the run
    void init() {
      nevt = 0;

      /// @todo Initialise and register projections here
      const FastJets jets(FinalState(-2.6, 2.6, 0.0*GeV), FastJets::ANTIKT, 0.5);
      addProjection(jets, "Jets");
      
      /// @todo Book histograms here, e.g.:
      // _h_XXXX = bookProfile1D(1, 1, 1);
      // _h_YYYY = bookHistogram1D(2, 1, 1);
	for (int ij=0; ij<njetptmn; ij++) {
	  _h_thrustc[ij] = bookHistogram1D(1, 1, ij+1);
	  _h_minorc[ij] = bookHistogram1D(1, 2, ij+1);
	  _h_broadt[ij] = bookHistogram1D(1, 3, ij+1);
	  
	  _alow1[ij] = _h_thrustc[ij]->axis().binLowerEdge(0);
	  _alow2[ij] = _h_minorc[ij]->axis().binLowerEdge(0);
	  _alow3[ij] = _h_broadt[ij]->axis().binLowerEdge(0);
	  
	  _ahgh1[ij] = _h_thrustc[ij]->axis().binUpperEdge(_h_thrustc[ij]->axis().bins()-1);
	  _ahgh2[ij] = _h_minorc[ij]->axis().binUpperEdge(_h_minorc[ij]->axis().bins()-1);
	  _ahgh3[ij] = _h_broadt[ij]->axis().binUpperEdge(_h_broadt[ij]->axis().bins()-1);
	  
	  cout <<"alow "<< _alow1[ij]<<" "<<_alow2[ij]<<" "<<_alow3[ij]<<" "<<_ahgh1[ij]<<" "<<_ahgh2[ij]<<" "<<_ahgh3[ij]<<endl;
	}
    }

    /// Perform the per-event analysis
    void analyze(const Event& event) {
      nevt++;
      const double weight = event.weight();
      
      /// @todo Do the event by event analysis here
	
	const Jets& jets = applyProjection<FastJets>(event, "Jets").jetsByPt(30.0*GeV);
	
	if (jets.size()<2) vetoEvent;
	double leadingpt = jets[0].momentum().pT()/GeV;
	if (jets.size() < 2 ||
	    fabs(jets[0].momentum().eta()) >= 1.3 ||
	    fabs(jets[1].momentum().eta()) >= 1.3 ||
	    leadingpt < 110.0) {
	  
	  vetoEvent;
	}
	
	std::vector<HepLorentzVector> momenta;
	foreach (const Jet& ij, jets) {
	  if (fabs(ij.momentum().eta()) < 1.3) {
	    HepLorentzVector tmp4v(ij.momentum().px(), ij.momentum().py(), ij.momentum().pz(), ij.momentum().E());
	    momenta.push_back(tmp4v);
	  }
	}
	
	EventShape_rivet  eventshape(momenta);
	eventvar = eventshape.getEventShapes();
	
	if (nevt%10000==1) cout <<"njets "<<jets.size()<<" "<< leadingpt<<" "<<eventvar[0]<<" "<<eventvar[1]<<" "<<eventvar[2]<<" "<<eventvar[3]<<" "<<nevt<<" "<<weight<<" "<<endl;

	if (eventvar[nevtvar]<0) vetoEvent; // Jets are not only one hemesphere
	
	for (int ij=njetptmn-1; ij>=0; ij--) {
	  
	  if (leadingpt > leadingPtThreshold[ij]) {
	    if (eventvar[0]>=_alow1[ij] &&  eventvar[0]<=_ahgh1[ij]) {_h_thrustc[ij]->fill(eventvar[0], weight);}
	    if (eventvar[1]>=_alow2[ij] &&  eventvar[1]<=_ahgh2[ij]) {_h_minorc[ij]->fill(eventvar[1], weight);}
	    if (eventvar[2]>=_alow3[ij] &&  eventvar[2]<=_ahgh3[ij] && eventvar[nevtvar]>=3) {_h_broadt[ij]->fill(eventvar[2], weight);}
	    break;
	  }
	}
    }
    /// Normalise histograms etc., after the run
    void finalize() {
      /// @todo Normalise, scale and otherwise manipulate histograms here
      
      // scale(_h_YYYY, crossSection()/sumOfWeights()); # norm to cross section
      // normalize(_h_YYYY); # normalize to unity
      for (int ij=0; ij<njetptmn; ij++) {
	normalize(_h_thrustc[ij]);
	normalize(_h_minorc[ij]);
	normalize(_h_broadt[ij]);
      }
    }
    
  private:
    
    // Data members like post-cuts event weight counters go here
    
  private:
    /// @name Histograms

    AIDA::IProfile1D *_h_XXXX;
    AIDA::IHistogram1D *_h_YYYY;

    AIDA::IHistogram1D* _h_thrustc[njetptmn];
    AIDA::IHistogram1D* _h_minorc[njetptmn];
    AIDA::IHistogram1D* _h_broadt[njetptmn];
    
    vector<double>eventvar;
    int nevt;
    
    
    double _alow1[njetptmn], _alow2[njetptmn], _alow3[njetptmn];
    double _ahgh1[njetptmn], _ahgh2[njetptmn], _ahgh3[njetptmn];
  };

  // The hook for the plugin system
  DECLARE_RIVET_PLUGIN(CMS_SMP_12_022);

}
