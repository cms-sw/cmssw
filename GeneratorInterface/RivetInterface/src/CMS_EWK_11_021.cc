#include "Rivet/Analysis.hh"
#include "Rivet/RivetAIDA.hh"
#include "Rivet/Tools/Logging.hh"
#include "Rivet/Tools/BinnedHistogram.hh"
#include "Rivet/Projections/FinalState.hh"
#include "Rivet/Projections/FastJets.hh"
#include "Rivet/Projections/ChargedFinalState.hh"
#include "Rivet/Projections/InvMassFinalState.hh"
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

      const FinalState fs(-5.0,5.0);
      addProjection(fs, "FS");

      // Histograms without data
      _histMll            = bookHistogram1D("Mll", 60, 50., 130.);
      _histNjets          = bookHistogram1D("Njets", 5, -0.5, 4.5);
      _histPtll           = bookHistogram1D("Ptll", 50, 0., 1000.);
      _histMjj            = bookHistogram1D("Mjj", 70, 0., 1400.);   
      _histPtJet1         = bookHistogram1D("PtJet1", 45, 50., 500.);
      _histPtJet2         = bookHistogram1D("PtJet2", 45, 50., 500.);
      _histPtJet3         = bookHistogram1D("PtJet3", 45, 50., 500.);
      _histPtJet4         = bookHistogram1D("PtJet4", 45, 50., 500.);
      _histDeltaPhiZJ2    = bookHistogram1D("DeltaPhiZJ2", 32, 0., 3.15);
      _histDeltaPhiJ1J2_2 = bookHistogram1D("DeltaPhiJ1J2_2", 32, 0., 3.15);
      _histSumDeltaPhi    = bookHistogram1D("SumDeltaPhi", 32, 0, 6.30);
      // Boosted regime
      _histBoostedDeltaPhiZJ2    = bookHistogram1D("BoostedDeltaPhiZJ2", 32, 0., 3.15);
      _histBoostedDeltaPhiJ1J2_2 = bookHistogram1D("BoostedDeltaPhiJ1J2_2", 32, 0., 3.15);
      _histBoostedSumDeltaPhi    = bookHistogram1D("BoostedSumDeltaPhi", 32, 0, 6.30);


      //Histograms with data
      _histDeltaPhiZJ1    = bookHistogram1D(10, 1, 1);
      _histDeltaPhiZJ3    = bookHistogram1D(4, 1, 1);
      _histDeltaPhiZJ1_2  = bookHistogram1D(1, 1, 1);
      _histDeltaPhiZJ1_3  = bookHistogram1D(8, 1, 1);
      _histDeltaPhiZJ2_3  = bookHistogram1D(14, 1, 1);
      _histDeltaPhiJ1J2_3 = bookHistogram1D(3, 1, 1);
      _histDeltaPhiJ1J3_3 = bookHistogram1D(5, 1, 1);   	
      _histDeltaPhiJ2J3_3 = bookHistogram1D(12, 1, 1);   	
      _histTransvThrust   = bookHistogram1D(17, 1, 1);
      // Boosted regime
      _histBoostedDeltaPhiZJ1    = bookHistogram1D(9, 1, 1);
      _histBoostedDeltaPhiZJ3    = bookHistogram1D(18, 1, 1);
      _histBoostedDeltaPhiZJ1_2  = bookHistogram1D(2, 1, 1);
      _histBoostedDeltaPhiZJ1_3  = bookHistogram1D(6, 1, 1);
      _histBoostedDeltaPhiZJ2_3  = bookHistogram1D(7, 1, 1);
      _histBoostedDeltaPhiJ1J2_3 = bookHistogram1D(15, 1, 1);
      _histBoostedDeltaPhiJ1J3_3 = bookHistogram1D(13, 1, 1);    
      _histBoostedDeltaPhiJ2J3_3 = bookHistogram1D(11, 1, 1);    
      _histBoostedTransvThrust   = bookHistogram1D(16, 1, 1);

      }

    // Function that finds all photons, and particles and antiparticles of a specified type:
    void FindPhoParParbar(ParticleVector FPtcls, int Par, vector<unsigned int>& PhoIndex, vector<unsigned int>& ParIndex, vector<unsigned int>& ParbarIndex)
      {
      for(unsigned int i(0); i!=FPtcls.size(); ++i)
        {
        FourMomentum p(FPtcls[i].momentum());
        double eta = p.eta();
        int Id = FPtcls[i].pdgId();

        if( Id==22 ) PhoIndex.push_back(i);
        if( Id==Par && fabs(eta) < 2.4 ) ParIndex.push_back(i); // && (fabs(eta) < 1.4442 || fabs(eta) > 1.5660)
        if( Id==-Par && fabs(eta) < 2.4 ) ParbarIndex.push_back(i); // && (fabs(eta) < 1.4442 || fabs(eta) > 1.5660)  
        }
        return;
      }



    //Function that convolutes a particle (or antiparticle) with the surrounding photons:
    void ConvPar(vector<unsigned int> PhoIndex, vector<unsigned int> ParIndex, FourMomentum& par_conv, vector<unsigned int>& par_del, ParticleVector FPtcls, bool& is_par)
      {

      vector<unsigned int> tmp_par_del;
      FourMomentum tmp_par_conv;
      FourMomentum tmp_test_conv;

      for(unsigned int i(0); i != ParIndex.size(); ++i)
        {
        double R;
        double mass;
        FourMomentum par_4p (FPtcls.at( ParIndex[i]).momentum());
        double par_eta (par_4p.eta());
        double par_phi (par_4p.phi());
        
        tmp_par_del.clear();
        tmp_par_del.push_back(ParIndex[i]);
        tmp_par_conv = par_4p;
        tmp_test_conv = par_4p;
       
        for(unsigned int j(0); j != PhoIndex.size(); ++j)
          {
          FourMomentum pho_4p (FPtcls.at(PhoIndex[j]).momentum());
          double pho_eta (pho_4p.eta());
          double pho_phi (pho_4p.phi());
    
          if (fabs(par_eta) < 1.4442) R = 0.05;
          else R=0.07;

          if (sqrt( (par_eta-pho_eta)*(par_eta-pho_eta) + (par_phi-pho_phi)*(par_phi-pho_phi)) < R )
            {
            tmp_par_conv += pho_4p;
            tmp_par_del.push_back( PhoIndex[j]);

            if (tmp_test_conv.mass2() < 0. ) mass = 0;
            else mass = tmp_test_conv.mass();

            double E_sum = tmp_test_conv.E() + pho_4p.E();
            double px_tmp = tmp_test_conv.px() * sqrt( E_sum*E_sum - mass*mass) / tmp_test_conv.p().mod();
            double py_tmp = tmp_test_conv.py() * sqrt( E_sum*E_sum - mass*mass) / tmp_test_conv.p().mod();
            double pz_tmp = tmp_test_conv.pz() * sqrt( E_sum*E_sum - mass*mass) / tmp_test_conv.p().mod();

            tmp_test_conv.setPx(px_tmp);
            tmp_test_conv.setPy(py_tmp);
            tmp_test_conv.setPz(pz_tmp);
            tmp_test_conv.setE(E_sum);
            }
          }

        if ((tmp_par_conv.pT() > 20.0) && (tmp_par_conv.pT() > par_conv.pT()))
          {
          par_conv = tmp_par_conv;
          par_del = tmp_par_del;
          is_par = true;
          }
        }
      return;
      }


    void GetPtEtaPhi(FourMomentum p1, double& pt, double& eta, double& phi)
      {
      pt = p1.pT();
      eta = p1.eta();
      phi = p1.phi();
      return;
      }


    void analyze(const Event& event)
      {
      const double weight = event.weight();
      bool is_ele = false;
      bool is_pos = false;
      bool is_mu = false;
      bool is_mubar = false;
      bool Zee = false;
      bool Zmm = false;
      bool is_boosted = false;

      vector<unsigned int> pho_index;

      vector<unsigned int> ele_index;
      vector<unsigned int> pos_index;
      vector<unsigned int> ele_del;
      vector<unsigned int> pos_del;

      vector<unsigned int> mu_index;
      vector<unsigned int> mubar_index;
      vector<unsigned int> mu_del;
      vector<unsigned int> mubar_del;
  
      vector<unsigned int> total_del;


      FourMomentum ele_conv (0,0,0,0);  
      FourMomentum pos_conv (0,0,0,0);

      FourMomentum mu_conv (0,0,0,0);  
      FourMomentum mubar_conv (0,0,0,0);

      ParticleVector final_ptcls;
      std::vector<fastjet::PseudoJet> vecs;

      const FinalState& fs = applyProjection<FinalState>(event, "FS");
      final_ptcls = fs.particlesByPt();

      // Find all electrons (muons), positrons (antimuons) and photons
      FindPhoParParbar(final_ptcls, 11, pho_index, ele_index, pos_index);
      pho_index.clear();
      FindPhoParParbar(final_ptcls, 13, pho_index, mu_index, mubar_index);      

      // Convolute electrons with surrounding photons
      ConvPar(pho_index, ele_index, ele_conv, ele_del, final_ptcls, is_ele);
      // Convolute muons with surrounding photons
      ConvPar(pho_index, mu_index, mu_conv, mu_del, final_ptcls, is_mu);

      if ((!is_ele) && (!is_mu)) vetoEvent;

      // Convolute positrons with surrounding photons
      ConvPar(pho_index, pos_index, pos_conv, pos_del, final_ptcls, is_pos);
      // Convolute anti muons with surrounding photons
      ConvPar(pho_index, mubar_index, mubar_conv, mubar_del, final_ptcls, is_mubar);


      if ((!is_pos) && (!is_mubar)) vetoEvent;

      FourMomentum Z_momentum;
      FourMomentum Zee_momentum(add(pos_conv, ele_conv));
      FourMomentum Zmm_momentum(add(mubar_conv, mu_conv));
 
      if (Zee_momentum.mass2() >= 0. )
        {
        if (Zee_momentum.mass() > 71.11 && Zee_momentum.mass() < 111.) { Z_momentum = Zee_momentum; Zee = true; }
        }

      if ( (!Zee) && Zmm_momentum.mass2() >= 0. )
        {
        if ( Zmm_momentum.mass() > 71.11 && Zmm_momentum.mass() < 111.) { Z_momentum = Zmm_momentum; Zmm = true; }
        else vetoEvent;
        }      
      
      if ( (!Zee) && (!Zmm) ) vetoEvent;

      if (Z_momentum.pT() > 150) is_boosted = true;

      double par_pt, par_eta, par_phi;
      double parbar_pt, parbar_eta, parbar_phi;

      if (Zee)
        {
        GetPtEtaPhi(ele_conv, par_pt, par_eta, par_phi);
        GetPtEtaPhi(pos_conv, parbar_pt, parbar_eta, parbar_phi);
        
        total_del.reserve(ele_del.size()+pos_del.size() );
        total_del.insert(total_del.end(), ele_del.begin(), ele_del.end());
        total_del.insert(total_del.end(), pos_del.begin(), pos_del.end());
        sort(total_del.begin(), total_del.end());
        }

      if (Zmm)
        {
        GetPtEtaPhi(mu_conv, par_pt, par_eta, par_phi);
        GetPtEtaPhi(mubar_conv, parbar_pt, parbar_eta, parbar_phi);
        
        total_del.reserve(mu_del.size()+mubar_del.size() );
        total_del.insert(total_del.end(), mu_del.begin(), mu_del.end());
        total_del.insert(total_del.end(), mubar_del.begin(), mubar_del.end());
        sort(total_del.begin(), total_del.end());
        }

      for(unsigned int i(total_del.size()); i !=0; --i)
        final_ptcls.erase(final_ptcls.begin()+total_del.at(i-1));

    _histMll->fill( Z_momentum.mass(), weight);

     
      for(unsigned int i(0); i != final_ptcls.size(); ++i)
        {
        if (fabs(final_ptcls[i].pdgId()) == 12 || fabs(final_ptcls[i].pdgId()) == 14 || fabs(final_ptcls[i].pdgId()) == 16 ) continue;
        if (PID::threeCharge (final_ptcls[i].pdgId()) != 0)
          {
          if (final_ptcls[i].momentum().E() < 0.25) continue;
          }

        fastjet::PseudoJet pseudoJet(final_ptcls[i].momentum().px(), final_ptcls[i].momentum().py(), final_ptcls[i].momentum().pz(), final_ptcls[i].momentum().E());
        pseudoJet.set_user_index(i);
        vecs.push_back(pseudoJet);
        }

      vector<fastjet::PseudoJet> jet_list;
      fastjet::ClusterSequence cseq(vecs, fastjet::JetDefinition(fastjet::antikt_algorithm, 0.5));
      vector<fastjet::PseudoJet> jets = sorted_by_pt(cseq.inclusive_jets(50.0));                  // In draft, the jet-treshold is 50 GeV

      for(unsigned int i(0); i<jets.size(); ++i)
        {
        double j_eta = jets[i].eta();
        double j_phi = jets[i].phi();
        double j_pt = jets[i].pt();

        if (fabs(j_eta) <2.5) 
          {
          if(j_pt > 50.0)
            {
            if (deltaR(par_pt, par_phi, j_eta, j_phi) > 0.4 && deltaR(parbar_pt, parbar_phi, j_eta, j_phi) > 0.4)
              jet_list.push_back(jets[i]);
            continue;
            }
          }
        }

      double Njets = jet_list.size();

      if (Njets) 
        {

        // Collect Z and jets transverse momenta to calculate transverse thrust
        std::vector<Vector3> momenta;
        momenta.clear();
        Vector3 mom = Z_momentum.p();
        mom.setZ(0.0);
        momenta.push_back(mom);

        for (unsigned int i(0); i != jet_list.size(); ++i)
          {
	  double momX = 0;
	  double momY = 0;
            for (unsigned int j(0); j != cseq.constituents(jet_list[i]).size(); ++j)
            {
            std::valarray<double> mom_4 = cseq.constituents(jet_list[i])[j].four_mom();
            momX += mom_4[0];        
            momY += mom_4[1];
            }
          mom.setX(momX);        
          mom.setY(momY);
          mom.setZ(0.0);
          momenta.push_back(mom);
          }

        if (momenta.size() <= 2) 
          {
          // We need to use a ghost so that Thrust.calc() doesn't return 1.
          momenta.push_back(Vector3(0.0000001,0.0000001,0.0000001));
          }

        Thrust thrust;
        thrust.calc(momenta);    
        _histTransvThrust->fill(max(log( 1-thrust.thrust() ), -14.), weight); // d17


        if (is_boosted) _histBoostedTransvThrust->fill(max(log( 1-thrust.thrust() ), -14.), weight); // d16 
    
        double Ptll = Z_momentum.pT();
        double PhiZ = Z_momentum.phi();
        double PtJet1 = jet_list[0].pt();
        double PhiJet1 = jet_list[0].phi();

        _histNjets->fill(Njets, weight);
        _histPtll->fill(Ptll, weight);
        _histPtJet1->fill(PtJet1, weight);
        _histDeltaPhiZJ1->fill(deltaPhi(PhiJet1,PhiZ), weight); // d10 300*0.10472*

        if (is_boosted) _histBoostedDeltaPhiZJ1->fill(deltaPhi(PhiJet1,PhiZ), weight); // d09 300*0.10472*
 
        if (Njets > 1)
          {
          FourMomentum J1_4p (jet_list[0].E(), jet_list[0].px(), jet_list[0].py(), jet_list[0].pz());
          FourMomentum J2_4p (jet_list[1].E(), jet_list[1].px(), jet_list[1].py(), jet_list[1].pz());
          FourMomentum pJ1J2(add(J1_4p,J2_4p)); 

          double Mjj;
          //if (pJ1J2.mass2() < 0. ) Mjj = 0;
          //else 
          Mjj = pJ1J2.mass(); // pJ1J2.mass() gives error message.
          double PtJet2 = jet_list[1].pt();
          double PhiJet2 = jet_list[1].phi();

          _histMjj->fill(Mjj, weight);
          _histPtJet2->fill(PtJet2, weight);
          _histDeltaPhiZJ2->fill(deltaPhi(PhiJet2, PhiZ), weight);
	  _histDeltaPhiZJ1_2->fill(deltaPhi(PhiJet1,PhiZ), weight);	// d01 10*0.10472
	  _histDeltaPhiJ1J2_2->fill(deltaPhi(PhiJet1,PhiJet2), weight);

          if (is_boosted)
            {
            _histBoostedDeltaPhiZJ2->fill(deltaPhi(PhiJet2, PhiZ), weight);
      	    _histBoostedDeltaPhiZJ1_2->fill(deltaPhi(PhiJet1,PhiZ), weight);	// d02 30*0.10472*
	    _histBoostedDeltaPhiJ1J2_2->fill(deltaPhi(PhiJet1,PhiJet2), weight);
            } 

           if (Njets > 2)
             {
             double PtJet3 = jet_list[2].pt();
             double PhiJet3 = jet_list[2].phi();
             double SumDeltaPhi = deltaPhi(PhiJet1,PhiJet2) + deltaPhi(PhiJet1,PhiJet3) + deltaPhi(PhiJet2,PhiJet3);
	     _histPtJet3->fill(PtJet3, weight);
	     _histDeltaPhiZJ1_3->fill(deltaPhi(PhiJet1,PhiZ), weight);	// d08 0.10472*
	     _histDeltaPhiZJ2_3->fill(deltaPhi(PhiJet2,PhiZ), weight); // d14 10*0.10472*
	     _histDeltaPhiJ1J2_3->fill(deltaPhi(PhiJet1,PhiJet2), weight);	// d03 100*0.10472*
	     _histDeltaPhiJ1J3_3->fill(deltaPhi(PhiJet1,PhiJet3), weight);	// d05 10*0.10472*
	     _histDeltaPhiJ2J3_3->fill(deltaPhi(PhiJet2,PhiJet3), weight);	// d12 0.10472*
	     _histDeltaPhiZJ3->fill(deltaPhi(PhiZ,PhiJet3), weight);	// d04 0.10472*
	     _histSumDeltaPhi->fill(SumDeltaPhi, weight);


             if (is_boosted)
               {
  	       _histBoostedDeltaPhiZJ1_3->fill(deltaPhi(PhiJet1,PhiZ), weight); // d06 0.21416*
	       _histBoostedDeltaPhiZJ2_3->fill(deltaPhi(PhiJet2,PhiZ), weight); // d07 10*0.21416*
	       _histBoostedDeltaPhiJ1J2_3->fill(deltaPhi(PhiJet1,PhiJet2), weight); // d15 100*0.21416*
    	       _histBoostedDeltaPhiJ1J3_3->fill(deltaPhi(PhiJet1,PhiJet3), weight); // d13 10*0.21416*
	       _histBoostedDeltaPhiJ2J3_3->fill(deltaPhi(PhiJet2,PhiJet3), weight);	// d11 0.21416*
	       _histBoostedDeltaPhiZJ3->fill(deltaPhi(PhiZ,PhiJet3), weight); // d18 0.21416*
	       _histBoostedSumDeltaPhi->fill(SumDeltaPhi, weight);

               }

	      if(Njets>3)
                {
	        double PtJet4  = jet_list[3].pt();
	        _histPtJet4->fill(PtJet4, weight);

                }
              }
            }
          }
      }

    void finalize() 
      {
      normalize(_histMll,1.);
      normalize(_histNjets,1.);
      normalize(_histPtll,1.);
      normalize(_histMjj,1.);   
      normalize(_histPtJet1,1.);
      normalize(_histPtJet2,1.);
      normalize(_histPtJet3,1.);
      normalize(_histPtJet4,1.);
      normalize(_histDeltaPhiZJ1,1.); // d10 300.
      normalize(_histDeltaPhiZJ2,1.);
      normalize(_histDeltaPhiZJ3,1.); // d04
      normalize(_histDeltaPhiZJ1_2,1.); // d01 10.
      normalize(_histDeltaPhiZJ1_3,1.); // d08 100.
      normalize(_histDeltaPhiZJ2_3,1.); // d14
      normalize(_histDeltaPhiJ1J2_2,1.);
      normalize(_histDeltaPhiJ1J2_3,1.); // d03 100.
      normalize(_histSumDeltaPhi,1.);
      normalize(_histTransvThrust,1); // d17. They have apparently remembered to multiply by the binsize.
      normalize(_histDeltaPhiJ1J3_3, 1.); // d05 10.
      normalize(_histDeltaPhiJ2J3_3, 1.); // d12
      // Boosted
      normalize(_histBoostedDeltaPhiZJ1,1.); // d09 300.
      normalize(_histBoostedDeltaPhiZJ2,1.);
      normalize(_histBoostedDeltaPhiZJ3,1.); // d18
      normalize(_histBoostedDeltaPhiZJ1_2,1.); // d02 30.
      normalize(_histBoostedDeltaPhiZJ1_3,1.); // d06 300.
      normalize(_histBoostedDeltaPhiZJ2_3,1.); // d07 10.
      normalize(_histBoostedDeltaPhiJ1J2_2,1.);
      normalize(_histBoostedDeltaPhiJ1J2_3,1.); // d15 100.
      normalize(_histBoostedSumDeltaPhi,1.);
      normalize(_histBoostedTransvThrust,1); // d16. They have apparently remembered to multiply by the binsize.
      normalize(_histBoostedDeltaPhiJ1J3_3, 1.); // d13 10.
      normalize(_histBoostedDeltaPhiJ2J3_3, 1.); // d11
     }


  private:

    AIDA::IHistogram1D* _histMll;            
    AIDA::IHistogram1D* _histNjets;          
    AIDA::IHistogram1D* _histPtll;           
    AIDA::IHistogram1D* _histMjj;            
    AIDA::IHistogram1D* _histPtJet1;         
    AIDA::IHistogram1D* _histPtJet2;         
    AIDA::IHistogram1D* _histPtJet3;         
    AIDA::IHistogram1D* _histPtJet4;         
    AIDA::IHistogram1D* _histDeltaPhiZJ1;    
    AIDA::IHistogram1D* _histDeltaPhiZJ2;    
    AIDA::IHistogram1D* _histDeltaPhiZJ3;    
    AIDA::IHistogram1D* _histDeltaPhiZJ1_2;  
    AIDA::IHistogram1D* _histDeltaPhiZJ1_3;  
    AIDA::IHistogram1D* _histDeltaPhiZJ2_3;  
    AIDA::IHistogram1D* _histDeltaPhiJ1J2_2; 
    AIDA::IHistogram1D* _histDeltaPhiJ1J2_3; 
    AIDA::IHistogram1D* _histSumDeltaPhi;
    AIDA::IHistogram1D* _histTransvThrust; 
    AIDA::IHistogram1D* _histDeltaPhiJ1J3_3;
    AIDA::IHistogram1D* _histDeltaPhiJ2J3_3;
    // Boosted
    AIDA::IHistogram1D* _histBoostedDeltaPhiZJ1;   
    AIDA::IHistogram1D* _histBoostedDeltaPhiZJ2;    
    AIDA::IHistogram1D* _histBoostedDeltaPhiZJ3;    
    AIDA::IHistogram1D* _histBoostedDeltaPhiZJ1_2;  
    AIDA::IHistogram1D* _histBoostedDeltaPhiZJ1_3;  
    AIDA::IHistogram1D* _histBoostedDeltaPhiZJ2_3;  
    AIDA::IHistogram1D* _histBoostedDeltaPhiJ1J2_2; 
    AIDA::IHistogram1D* _histBoostedDeltaPhiJ1J2_3; 
    AIDA::IHistogram1D* _histBoostedSumDeltaPhi;
    AIDA::IHistogram1D* _histBoostedTransvThrust;
    AIDA::IHistogram1D* _histBoostedDeltaPhiJ1J3_3;
    AIDA::IHistogram1D* _histBoostedDeltaPhiJ2J3_3;  
  };
  
  AnalysisBuilder<CMS_EWK_11_021> plugin_CMS_EWK_11_021;
  
}



   
