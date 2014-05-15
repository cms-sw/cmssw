#include "Rivet/Analysis.hh"
#include "Rivet/RivetAIDA.hh"
#include "Rivet/Tools/Logging.hh"
#include "Rivet/Tools/BinnedHistogram.hh"
#include "Rivet/Projections/FinalState.hh"
#include "Rivet/Projections/FastJets.hh"
#include "Rivet/Projections/ZFinder.hh"
#include "Rivet/Tools/ParticleIdUtils.hh"
#include "Rivet/Math/Vector4.hh"
#include "Rivet/Projections/Thrust.hh"

namespace Rivet {

  class CMS_EWK_11_021 : public Analysis {
  public:

    //Constructor:
    CMS_EWK_11_021()
      : Analysis("CMS_EWK_11_021")
    {
      setBeams(PROTON, PROTON);
      setNeedsCrossSection(true);
    }


    void init()
      {

      //full final state
      const FinalState fs(-5.0,5.0);
      addProjection(fs, "FS");
      //Z finders for electrons and muons
      const ZFinder zfe(fs, -2.4, 2.4, 20*GeV, 11, 71*GeV, 111.*GeV, 0.1, true, false);
      const ZFinder zfm(fs, -2.4, 2.4, 20*GeV, 13, 71*GeV, 111.*GeV, 0.1, true, false);
      addProjection(zfe, "ZFE");
      addProjection(zfm, "ZFM");
      //jets  
      const FastJets jets(fs, FastJets::ANTIKT, 0.5);
      addProjection(jets, "JETS");

      //Histograms with data
      _histDeltaPhiZJ1    = bookHistogram1D(1, 1, 1); //10, 1, 1); 
      _histDeltaPhiZJ1_2  = bookHistogram1D(2, 1, 1); //1, 1, 1);  
      _histDeltaPhiZJ3    = bookHistogram1D(3, 1, 1); //4, 1, 1);  
      _histDeltaPhiZJ1_3  = bookHistogram1D(4, 1, 1);  //8, 1, 1);
      _histDeltaPhiZJ2_3  = bookHistogram1D(5, 1, 1); //14, 1, 1);
      _histDeltaPhiJ1J2_3 = bookHistogram1D(6, 1, 1); //3, 1, 1);
      _histDeltaPhiJ1J3_3 = bookHistogram1D(7, 1, 1); //5, 1, 1);   	
      _histDeltaPhiJ2J3_3 = bookHistogram1D(8, 1, 1); //12, 1, 1);   	
      _histTransvThrust   = bookHistogram1D(9, 1, 1); //17, 1, 1);
      // Boosted regime
      _histBoostedDeltaPhiZJ1    = bookHistogram1D(10, 1, 1); //9, 1, 1);
      _histBoostedDeltaPhiZJ1_2  = bookHistogram1D(11, 1, 1); //2, 1, 1);
      _histBoostedDeltaPhiZJ3    = bookHistogram1D(12, 1, 1); //18, 1, 1);
      _histBoostedDeltaPhiZJ1_3  = bookHistogram1D(13, 1, 1); //6, 1, 1);
      _histBoostedDeltaPhiZJ2_3  = bookHistogram1D(14, 1, 1); //7, 1, 1);
      _histBoostedDeltaPhiJ1J2_3 = bookHistogram1D(15, 1, 1);  //15, 1, 1);
      _histBoostedDeltaPhiJ1J3_3 = bookHistogram1D(16, 1, 1); //13, 1, 1);    
      _histBoostedDeltaPhiJ2J3_3 = bookHistogram1D(17, 1, 1); //11, 1, 1);    
      _histBoostedTransvThrust   = bookHistogram1D(18, 1, 1); //16, 1, 1);

      }

    void analyze(const Event& event){
      const double weight = event.weight();
      //apply the Z finders
      const ZFinder& zfe = applyProjection<ZFinder>(event, "ZFE");
      const ZFinder& zfm = applyProjection<ZFinder>(event, "ZFM");

      //if no Z found, veto
      if (zfe.empty() && zfm.empty())
        vetoEvent;

      //Choose the Z candidate
      const ParticleVector& z = !zfm.empty() ? zfm.bosons() : zfe.bosons();
      const ParticleVector& clusteredConstituents = !zfm.empty() ? zfm.constituents() : zfe.constituents();
      
      //determine whether we are in boosted regime
      bool is_boosted = false;
      if (z[0].momentum().pT()>150*GeV)
        is_boosted = true;

      //build the jets
      const FastJets& jetfs = applyProjection<FastJets>(event, "JETS");
      Jets jets = jetfs.jetsByPt(50.*GeV, MAXDOUBLE, -2.5, 2.5);

      //clean the jets against the lepton candidates, as in the paper, with a DeltaR cut of 0.4 against the clustered leptons
      std::vector<const Jet*> cleanedJets;
      for (unsigned int i = 0; i < jets.size(); ++i){
        bool isolated = true; 
        for (unsigned j = 0; j < clusteredConstituents.size(); ++j){
          if (deltaR(clusteredConstituents[j].momentum().vector3(), jets[i].momentum().vector3()) < 0.4){
            isolated=false;
            break;
          }
        }
        if (isolated)
          cleanedJets.push_back(&jets[i]);
      }

      unsigned int Njets = cleanedJets.size();
      //require at least 1 jet
      if (Njets < 1)
        vetoEvent;

      //now compute Thrust
      // Collect Z and jets transverse momenta to calculate transverse thrust
      std::vector<Vector3> momenta;
      momenta.clear();
      Vector3 mom = z[0].momentum().p();
      mom.setZ(0.0);
      momenta.push_back(mom);

      for (unsigned int i = 0; i < cleanedJets.size(); ++i){
        Vector3 mj = cleanedJets[i]->momentum().vector3();
        mj.setZ(0.0);
        momenta.push_back(mj);
      }
      if (momenta.size() <= 2){ 
        // We need to use a ghost so that Thrust.calc() doesn't return 1.
        momenta.push_back(Vector3(0.0000001,0.0000001,0.));
      }

      Thrust thrust;
      thrust.calc(momenta);    
      _histTransvThrust->fill(max(log( 1-thrust.thrust() ), -14.), weight); // d17


      if (is_boosted) _histBoostedTransvThrust->fill(max(log( 1-thrust.thrust() ), -14.), weight); // d16 
    
      double PhiZ = z[0].momentum().phi();
      double PhiJet1 = cleanedJets[0]->phi();

      _histDeltaPhiZJ1->fill(deltaPhi(PhiJet1,PhiZ), weight); // d10 300*0.10472*

      if (is_boosted) _histBoostedDeltaPhiZJ1->fill(deltaPhi(PhiJet1,PhiZ), weight); // d09 300*0.10472*
 
      if (Njets > 1){
        double PhiJet2 = cleanedJets[1]->phi();

	_histDeltaPhiZJ1_2->fill(deltaPhi(PhiJet1,PhiZ), weight);	// d01 10*0.10472

        if (is_boosted){
      	  _histBoostedDeltaPhiZJ1_2->fill(deltaPhi(PhiJet1,PhiZ), weight);	// d02 30*0.10472*
        } 

        if (Njets > 2){
          double PhiJet3 = cleanedJets[2]->phi();
	  _histDeltaPhiZJ1_3->fill(deltaPhi(PhiJet1,PhiZ), weight);	// d08 0.10472*
	  _histDeltaPhiZJ2_3->fill(deltaPhi(PhiJet2,PhiZ), weight); // d14 10*0.10472*
	  _histDeltaPhiJ1J2_3->fill(deltaPhi(PhiJet1,PhiJet2), weight);	// d03 100*0.10472*
	  _histDeltaPhiJ1J3_3->fill(deltaPhi(PhiJet1,PhiJet3), weight);	// d05 10*0.10472*
	  _histDeltaPhiJ2J3_3->fill(deltaPhi(PhiJet2,PhiJet3), weight);	// d12 0.10472*
	  _histDeltaPhiZJ3->fill(deltaPhi(PhiZ,PhiJet3), weight);	// d04 0.10472*


          if (is_boosted) {
  	       _histBoostedDeltaPhiZJ1_3->fill(deltaPhi(PhiJet1,PhiZ), weight); // d06 0.21416*
	       _histBoostedDeltaPhiZJ2_3->fill(deltaPhi(PhiJet2,PhiZ), weight); // d07 10*0.21416*
	       _histBoostedDeltaPhiJ1J2_3->fill(deltaPhi(PhiJet1,PhiJet2), weight); // d15 100*0.21416*
    	       _histBoostedDeltaPhiJ1J3_3->fill(deltaPhi(PhiJet1,PhiJet3), weight); // d13 10*0.21416*
	       _histBoostedDeltaPhiJ2J3_3->fill(deltaPhi(PhiJet2,PhiJet3), weight);	// d11 0.21416*
	       _histBoostedDeltaPhiZJ3->fill(deltaPhi(PhiZ,PhiJet3), weight); // d18 0.21416*
          }
        }
      }
    }  

    void normalizeNoOverflows(AIDA::IHistogram1D* plot, double integral){
      double factor=1.;
      if (plot->sumBinHeights()>0 && plot->sumAllBinHeights()>0)
        factor = plot->sumAllBinHeights()/plot->sumBinHeights();
      normalize(plot, factor*integral);  
    }

    void finalize() 
      {
      normalizeNoOverflows(_histDeltaPhiZJ1,1.);
      normalizeNoOverflows(_histDeltaPhiZJ3,1.);
      normalizeNoOverflows(_histDeltaPhiZJ1_2,1.);
      normalizeNoOverflows(_histDeltaPhiZJ1_3,1.);
      normalizeNoOverflows(_histDeltaPhiZJ2_3,1.);
      normalizeNoOverflows(_histDeltaPhiJ1J2_3,1.);
      normalize(_histTransvThrust,1.);
      normalizeNoOverflows(_histDeltaPhiJ1J3_3, 1.);
      normalizeNoOverflows(_histDeltaPhiJ2J3_3, 1.);
      // Boosted
      normalizeNoOverflows(_histBoostedDeltaPhiZJ1,1.);
      normalizeNoOverflows(_histBoostedDeltaPhiZJ3,1.);
      normalizeNoOverflows(_histBoostedDeltaPhiZJ1_2,1.);
      normalizeNoOverflows(_histBoostedDeltaPhiZJ1_3,1.);
      normalizeNoOverflows(_histBoostedDeltaPhiZJ2_3,1.);
      normalizeNoOverflows(_histBoostedDeltaPhiJ1J2_3,1.);
      normalize(_histBoostedTransvThrust,1.);
      normalizeNoOverflows(_histBoostedDeltaPhiJ1J3_3, 1.);
      normalizeNoOverflows(_histBoostedDeltaPhiJ2J3_3, 1.);
     }


  private:

    AIDA::IHistogram1D* _histDeltaPhiZJ1;    
    AIDA::IHistogram1D* _histDeltaPhiZJ3;    
    AIDA::IHistogram1D* _histDeltaPhiZJ1_2;  
    AIDA::IHistogram1D* _histDeltaPhiZJ1_3;  
    AIDA::IHistogram1D* _histDeltaPhiZJ2_3;  
    AIDA::IHistogram1D* _histDeltaPhiJ1J2_3; 
    AIDA::IHistogram1D* _histTransvThrust; 
    AIDA::IHistogram1D* _histDeltaPhiJ1J3_3;
    AIDA::IHistogram1D* _histDeltaPhiJ2J3_3;
    // Boosted
    AIDA::IHistogram1D* _histBoostedDeltaPhiZJ1;   
    AIDA::IHistogram1D* _histBoostedDeltaPhiZJ3;    
    AIDA::IHistogram1D* _histBoostedDeltaPhiZJ1_2;  
    AIDA::IHistogram1D* _histBoostedDeltaPhiZJ1_3;  
    AIDA::IHistogram1D* _histBoostedDeltaPhiZJ2_3;  
    AIDA::IHistogram1D* _histBoostedDeltaPhiJ1J2_3; 
    AIDA::IHistogram1D* _histBoostedTransvThrust;
    AIDA::IHistogram1D* _histBoostedDeltaPhiJ1J3_3;
    AIDA::IHistogram1D* _histBoostedDeltaPhiJ2J3_3;  
  };
  
  AnalysisBuilder<CMS_EWK_11_021> plugin_CMS_EWK_11_021;
  
}
