// -*- C++ -*-
#include "Rivet/Analysis.hh"
#include "Rivet/RivetAIDA.hh"
#include "Rivet/Tools/Logging.hh"
#include "Rivet/Projections/FinalState.hh"
#include "Rivet/Projections/FastJets.hh"
#include "fastjet/tools/Filter.hh"
#include "fastjet/tools/Pruner.hh"
#include "Rivet/Projections/WFinder.hh"
#include "Rivet/Projections/ZFinder.hh"
#include "Rivet/Particle.hh"
#include "Rivet/Tools/ParticleIdUtils.hh"

namespace Rivet {
    
    
    class CMS_2013_I1224539 : public Analysis {
    public:
        
        /// @name Constructors etc.
        //@{
        
        /// Constructor
        CMS_2013_I1224539()
        : Analysis("CMS_2013_I1224539")
        {    }
        
        //@}
        
        
    public:
        
        /// @name Analysis methods
        //@{
        
        /// Book histograms and initialise projections before the run
        void init() {
            
            FinalState fs(-2.4, 2.4, 0.0*GeV);
            addProjection(fs, "FS");
            
            // find W's, w PT > 125, MET > 50
            WFinder wfinder(fs, -2.4, 2.4, 125.0*GeV, ELECTRON, 50.0*GeV, 1000.0*GeV, 50.0*GeV, 0.2, true, false, 80.4, true);
            addProjection(wfinder, "WFinder");
            // find Z's, z pT > 125
            ZFinder zfinder(fs, -2.4, 2.4, 125.0*GeV, ELECTRON, 80.0*GeV, 100.0*GeV, 0.2, true, true);
            addProjection(zfinder, "ZFinder");            
            
            // jet collections
            addProjection(FastJets(fs, FastJets::ANTIKT, 0.7), "JetsAK7");
            addProjection(FastJets(fs, FastJets::CAM, 0.8), "JetsCA8");
            addProjection(FastJets(fs, FastJets::CAM, 1.2), "JetsCA12");
            
            // GG Rivet. No 2-d histograms. Boooo. 
            // dijets
            for( unsigned i = 0; i < N_PT_BINS_dj; ++i ) {
                _h_ungroomedAvgJetMass_dj[i] = bookHistogram1D(i+1+0*N_PT_BINS_dj,1,1);
            }
            for( unsigned i = 0; i < N_PT_BINS_dj; ++i ) {
                _h_filteredAvgJetMass_dj[i] = bookHistogram1D(i+1+1*N_PT_BINS_dj,1,1);
            }
            for( unsigned i = 0; i < N_PT_BINS_dj; ++i ) {
                _h_trimmedAvgJetMass_dj[i] = bookHistogram1D(i+1+2*N_PT_BINS_dj,1,1);
            }
            for( unsigned i = 0; i < N_PT_BINS_dj; ++i ) {
                _h_prunedAvgJetMass_dj[i] = bookHistogram1D(i+1+3*N_PT_BINS_dj,1,1);
            }
            // wjets
            int wjetsOffset = 28;
            for( unsigned i = 0; i < N_PT_BINS_vj; ++i ) {
                _h_ungroomedJetMass_AK7_wj[i] = bookHistogram1D(wjetsOffset+i+1+0*N_PT_BINS_vj,1,1);
            }
            for( unsigned i = 0; i < N_PT_BINS_vj; ++i ) {
                _h_filteredJetMass_AK7_wj[i] = bookHistogram1D(wjetsOffset+i+1+1*N_PT_BINS_vj,1,1);
            }
            for( unsigned i = 0; i < N_PT_BINS_vj; ++i ) {
                _h_trimmedJetMass_AK7_wj[i] = bookHistogram1D(wjetsOffset+i+1+2*N_PT_BINS_vj,1,1);
            }
            for( unsigned i = 0; i < N_PT_BINS_vj; ++i ) {
                _h_prunedJetMass_AK7_wj[i] = bookHistogram1D(wjetsOffset+i+1+3*N_PT_BINS_vj,1,1);
            }
            for( unsigned i = 0; i < N_PT_BINS_vj; ++i ) {
                _h_prunedJetMass_CA8_wj[i] = bookHistogram1D(wjetsOffset+i+1+4*N_PT_BINS_vj,1,1);
            }
            for( unsigned i = 1; i < N_PT_BINS_vj; ++i ) {
                _h_filteredJetMass_CA12_wj[i] = bookHistogram1D(wjetsOffset+i+5*N_PT_BINS_vj,1,1);
            }
            // zjets offset
            int zjetsOffset = 51;
            for( unsigned i = 0; i < N_PT_BINS_vj; ++i ) {
                _h_ungroomedJetMass_AK7_zj[i] = bookHistogram1D(zjetsOffset+i+1+0*N_PT_BINS_vj,1,1);
            }
            for( unsigned i = 0; i < N_PT_BINS_vj; ++i ) {
                _h_filteredJetMass_AK7_zj[i] = bookHistogram1D(zjetsOffset+i+1+1*N_PT_BINS_vj,1,1);
            }
            for( unsigned i = 0; i < N_PT_BINS_vj; ++i ) {
                _h_trimmedJetMass_AK7_zj[i] = bookHistogram1D(zjetsOffset+i+1+2*N_PT_BINS_vj,1,1);
            }
            for( unsigned i = 0; i < N_PT_BINS_vj; ++i ) {
                _h_prunedJetMass_AK7_zj[i] = bookHistogram1D(zjetsOffset+i+1+3*N_PT_BINS_vj,1,1);
            }
            for( unsigned i = 0; i < N_PT_BINS_vj; ++i ) {
                _h_prunedJetMass_CA8_zj[i] = bookHistogram1D(zjetsOffset+i+1+4*N_PT_BINS_vj,1,1);
            }
            for( unsigned i = 1; i < N_PT_BINS_vj; ++i ) {
                _h_filteredJetMass_CA12_zj[i] = bookHistogram1D(zjetsOffset+i+5*N_PT_BINS_vj,1,1);
            }
            
            
        }
        
        
        /// Perform the per-event analysis
        void analyze(const Event& event) {
            
            bool wjEvent = false;
            bool zjEvent = false;
                        
            // Hacking 2-d histograms. 
            double ptBins_dj[N_PT_BINS_dj+1] = {
                220.0,
                300.0,
                450.0,
                500.0,
                600.0,
                800.0,
                1000.0,
                1500.0};            
            double ptBins_vj[N_PT_BINS_vj+1] = {
                125.0,
                150.0,
                220.0,
                300.0,
                450.0};
            
            const double weight = event.weight() > 0 ? event.weight() : 1;
            
            // Get the W "projection"
            const WFinder& wfinder = applyProjection<WFinder>(event, "WFinder");
            Particle l;
            if (wfinder.bosons().size() == 1) {
                wjEvent = true;
                l=wfinder.constituentLeptons()[0];
            }
                        
            // Get the Z "projection"
            Particle l1;
            Particle l2;
            const ZFinder& zfinder = applyProjection<ZFinder>(event, "ZFinder");
            if (zfinder.bosons().size() == 1) {
                zjEvent = true;
                l1=zfinder.constituents()[0];
                l2=zfinder.constituents()[1];                
            }
            
            // Get the Jet "projections"
            Jets jetsAK7 = applyProjection<JetAlg>(event, "JetsAK7").jetsByPt(20*GeV);
            Jets jetsCA8 = applyProjection<JetAlg>(event, "JetsCA8").jetsByPt(20*GeV);
            Jets jetsCA12 = applyProjection<JetAlg>(event, "JetsCA12").jetsByPt(20*GeV);
            
            // Get the pseudojets. 
            const PseudoJets& psjetsAK7 = applyProjection<FastJets>(event, "JetsAK7").pseudoJetsByPt( 50.0*GeV );
            const PseudoJets& psjetsCA8 = applyProjection<FastJets>(event, "JetsCA8").pseudoJetsByPt( 50.0*GeV );
            const PseudoJets& psjetsCA12 = applyProjection<FastJets>(event, "JetsCA12").pseudoJetsByPt( 50.0*GeV );             
                        
            // Define the FJ3 grooming algorithms
            double rFilt = 0.3;
            int nFilt = 3;
            double rTrim = 0.2;
            double trimPtFracMin = 0.03;
            double zCut = 0.1;
            double RcutFactor = 0.5;
            
            fastjet::Filter filter( fastjet::Filter(fastjet::JetDefinition(fastjet::cambridge_algorithm, rFilt), fastjet::SelectorNHardest(nFilt)));
            fastjet::Filter trimmer( fastjet::Filter(fastjet::JetDefinition(fastjet::kt_algorithm, rTrim), fastjet::SelectorPtFractionMin(trimPtFracMin)));
            fastjet::Pruner pruner(fastjet::cambridge_algorithm, zCut, RcutFactor);      
                        
            // -----------------------
            // is W+jet event
            if (!zjEvent && wjEvent && (l.momentum().pT() > 80)){
                // Look at events with >= 1 AK7 jets
                if (!psjetsAK7.empty() && psjetsAK7.size() > 0) {
                    
                    // Get the leading jet
                    const fastjet::PseudoJet& j0 = psjetsAK7[0];
                    // Find the pt
                    double ptJ = j0.pt();
                    // Find the histogram bin that this belongs to. 
                    unsigned int njetBin = N_PT_BINS_vj;
                    for ( unsigned int ibin = 0; ibin < N_PT_BINS_vj; ++ibin ) {
                        if ( ptJ >= ptBins_vj[ibin] && ptJ < ptBins_vj[ibin+1] ) {
                            njetBin = ibin;
                            break;
                        }
                    }
                    
                    if ( njetBin >= N_PT_BINS_vj ) 
                        return;
                    
                    // Now run the substructure algs...
                    fastjet::PseudoJet filtered0 = filter(j0);                     
                    fastjet::PseudoJet trimmed0 = trimmer(j0); 
                    fastjet::PseudoJet pruned0 = pruner(j0); 
                    
                    // ... and fill the hists
                    _h_ungroomedJetMass_AK7_wj[njetBin]->fill( j0.m() / GeV, weight);
                    _h_filteredJetMass_AK7_wj[njetBin]->fill( filtered0.m() / GeV, weight);
                    _h_trimmedJetMass_AK7_wj[njetBin]->fill( trimmed0.m() / GeV, weight);
                    _h_prunedJetMass_AK7_wj[njetBin]->fill( pruned0.m() / GeV, weight);
                    
                }
                // Look at events with >= 1 CA8 jets
                if (!psjetsCA8.empty() && psjetsCA8.size() > 0) {
                    
                    // Get the leading jet
                    const fastjet::PseudoJet& j0 = psjetsCA8[0];
                    // Find the pt
                    double ptJ = j0.pt();
                    // Find the histogram bin that this belongs to. 
                    unsigned int njetBin = N_PT_BINS_vj;
                    for ( unsigned int ibin = 0; ibin < N_PT_BINS_vj; ++ibin ) {
                        if ( ptJ >= ptBins_vj[ibin] && ptJ < ptBins_vj[ibin+1] ) {
                            njetBin = ibin;
                            break;
                        }
                    }
                    
                    if ( njetBin >= N_PT_BINS_vj ) 
                        return;
                    
                    // Now run the substructure algs...
                    fastjet::PseudoJet pruned0 = pruner(j0); 
                    
                    // ... and fill the hists
                    _h_prunedJetMass_CA8_wj[njetBin]->fill( pruned0.m() / GeV, weight);
                    
                }
                // Look at events with >= 1 CA12 jets
                if (!psjetsCA12.empty() && psjetsCA12.size() > 0) {
                    
                    // Get the leading jet
                    const fastjet::PseudoJet& j0 = psjetsCA12[0];
                    // Find the pt
                    double ptJ = j0.pt();
                    // Find the histogram bin that this belongs to. 
                    unsigned int njetBin = N_PT_BINS_vj;
                    for ( unsigned int ibin = 1; ibin < N_PT_BINS_vj; ++ibin ) {
                        if ( ptJ >= ptBins_vj[ibin] && ptJ < ptBins_vj[ibin+1] ) {
                            njetBin = ibin;
                            break;
                        }
                    }
                    
                    if ( njetBin >= N_PT_BINS_vj ) 
                        return;
                    
                    // Now run the substructure algs...
                    fastjet::PseudoJet filtered0 = filter(j0); 
                    
                    // ... and fill the hists
                    _h_filteredJetMass_CA12_wj[njetBin]->fill( filtered0.m() / GeV, weight);
                    
                }
            }
            
            // -----------------------
            // is Z+jet event
            if (zjEvent && !wjEvent && (l1.momentum().pT() > 30) && (l2.momentum().pT() > 30)){
                
                // Look at events with >= 1 AK7 jets
                if (!psjetsAK7.empty() && psjetsAK7.size() > 0) {
                    
                    // Get the leading jet
                    const fastjet::PseudoJet& j0 = psjetsAK7[0];
                    // Find the pt
                    double ptJ = j0.pt();
                    // Find the histogram bin that this belongs to. 
                    unsigned int njetBin = N_PT_BINS_vj;
                    for ( unsigned int ibin = 0; ibin < N_PT_BINS_vj; ++ibin ) {
                        if ( ptJ >= ptBins_vj[ibin] && ptJ < ptBins_vj[ibin+1] ) {
                            njetBin = ibin;
                            break;
                        }
                    }
                    
                    if ( njetBin >= N_PT_BINS_vj ) 
                        return;
                    
                    // Now run the substructure algs...
                    fastjet::PseudoJet filtered0 = filter(j0);                     
                    fastjet::PseudoJet trimmed0 = trimmer(j0); 
                    fastjet::PseudoJet pruned0 = pruner(j0); 
                    
                    // ... and fill the hists
                    _h_ungroomedJetMass_AK7_zj[njetBin]->fill( j0.m() / GeV, weight);
                    _h_filteredJetMass_AK7_zj[njetBin]->fill( filtered0.m() / GeV, weight);
                    _h_trimmedJetMass_AK7_zj[njetBin]->fill( trimmed0.m() / GeV, weight);
                    _h_prunedJetMass_AK7_zj[njetBin]->fill( pruned0.m() / GeV, weight);
                    
                }
                // Look at events with >= 1 CA8 jets
                if (!psjetsCA8.empty() && psjetsCA8.size() > 0) {
                    
                    // Get the leading jet
                    const fastjet::PseudoJet& j0 = psjetsCA8[0];
                    // Find the pt
                    double ptJ = j0.pt();
                    // Find the histogram bin that this belongs to. 
                    unsigned int njetBin = N_PT_BINS_vj;
                    for ( unsigned int ibin = 0; ibin < N_PT_BINS_vj; ++ibin ) {
                        if ( ptJ >= ptBins_vj[ibin] && ptJ < ptBins_vj[ibin+1] ) {
                            njetBin = ibin;
                            break;
                        }
                    }
                    
                    if ( njetBin >= N_PT_BINS_vj ) 
                        return;
                    
                    // Now run the substructure algs...
                    fastjet::PseudoJet pruned0 = pruner(j0); 
                    
                    // ... and fill the hists
                    _h_prunedJetMass_CA8_zj[njetBin]->fill( pruned0.m() / GeV, weight);
                    
                }
                // Look at events with >= 1 CA12 jets
                if (!psjetsCA12.empty() && psjetsCA12.size() > 0) {
                    
                    // Get the leading jet
                    const fastjet::PseudoJet& j0 = psjetsCA12[0];
                    // Find the pt
                    double ptJ = j0.pt();
                    // Find the histogram bin that this belongs to. 
                    unsigned int njetBin = N_PT_BINS_vj;
                    for ( unsigned int ibin = 1; ibin < N_PT_BINS_vj; ++ibin ) {
                        if ( ptJ >= ptBins_vj[ibin] && ptJ < ptBins_vj[ibin+1] ) {
                            njetBin = ibin;
                            break;
                        }
                    }
                    
                    if ( njetBin >= N_PT_BINS_vj ) 
                        return;
                    
                    // Now run the substructure algs...
                    fastjet::PseudoJet filtered0 = filter(j0); 
                    
                    // ... and fill the hists
                    _h_filteredJetMass_CA12_zj[njetBin]->fill( filtered0.m() / GeV, weight);
                    
                }
                
            }
            // -----------------------
            // is dijet event
            if (!zjEvent && !wjEvent){
                
                // Look at events with >= 2 jets
                if (!psjetsAK7.empty() && psjetsAK7.size() > 1) {
                    
                    // Get the leading two jets
                    const fastjet::PseudoJet& j0 = psjetsAK7[0];
                    const fastjet::PseudoJet& j1 = psjetsAK7[1];
                    
                    // Find their average pt
                    double ptAvg = (j0.pt() + j1.pt()) * 0.5;
                    
                    // Find the histogram bin that this belongs to. 
                    unsigned int njetBin = N_PT_BINS_dj;
                    for ( unsigned int ibin = 0; ibin < N_PT_BINS_dj; ++ibin ) {
                        if ( ptAvg >= ptBins_dj[ibin] && ptAvg < ptBins_dj[ibin+1] ) {
                            njetBin = ibin;
                            break;
                        }
                    }
                    
                    if ( njetBin >= N_PT_BINS_dj ) 
                        return;
                    
                    // Now run the substructure algs...
                    fastjet::PseudoJet filtered0 = filter(j0); 
                    fastjet::PseudoJet filtered1 = filter(j1); 
                    
                    fastjet::PseudoJet trimmed0 = trimmer(j0); 
                    fastjet::PseudoJet trimmed1 = trimmer(j1); 
                    
                    fastjet::PseudoJet pruned0 = pruner(j0); 
                    fastjet::PseudoJet pruned1 = pruner(j1); 
                    
                    // ... and fill the hists
                    _h_ungroomedAvgJetMass_dj[njetBin]->fill( (j0.m() + j1.m()) * 0.5 / GeV, weight);
                    _h_filteredAvgJetMass_dj[njetBin]->fill( (filtered0.m() + filtered1.m()) * 0.5 / GeV, weight);
                    _h_trimmedAvgJetMass_dj[njetBin]->fill( (trimmed0.m() + trimmed1.m()) * 0.5 / GeV, weight);
                    _h_prunedAvgJetMass_dj[njetBin]->fill( (pruned0.m() + pruned1.m()) * 0.5 / GeV, weight); 
                    
                }
                
                
            }
            
        }
        /// Normalise histograms etc., after the run
        void finalize() {
            for ( unsigned int i = 0; i < N_PT_BINS_dj; ++i ) {
                normalize( _h_ungroomedAvgJetMass_dj[i], 1000. );
                normalize( _h_filteredAvgJetMass_dj[i], 1000. );
                normalize( _h_trimmedAvgJetMass_dj[i], 1000. );
                normalize( _h_prunedAvgJetMass_dj[i], 1000. );
            }            
            for ( unsigned int i = 0; i < N_PT_BINS_vj; ++i ) {
                normalize( _h_ungroomedJetMass_AK7_wj[i], 1000. );
                normalize( _h_filteredJetMass_AK7_wj[i], 1000. );
                normalize( _h_trimmedJetMass_AK7_wj[i], 1000. );
                normalize( _h_prunedJetMass_AK7_wj[i], 1000. );
                normalize( _h_prunedJetMass_CA8_wj[i], 1000. ); 
                
                normalize( _h_ungroomedJetMass_AK7_zj[i], 1000. );
                normalize( _h_filteredJetMass_AK7_zj[i], 1000. );
                normalize( _h_trimmedJetMass_AK7_zj[i], 1000. );
                normalize( _h_prunedJetMass_AK7_zj[i], 1000. );
                normalize( _h_prunedJetMass_CA8_zj[i], 1000. );                  
            }
            for ( unsigned int i = 1; i < N_PT_BINS_vj; ++i ) {
                normalize( _h_filteredJetMass_CA12_wj[i], 1000. );                
                normalize( _h_filteredJetMass_CA12_zj[i], 1000. );                
            }
            
        }
        
        //@}
        
        
    private:
        
        /// @name Histograms
        //@{
        enum {
            PT_220_300_dj=0,
            PT_300_450_dj,
            PT_450_500_dj,
            PT_500_600_dj,
            PT_600_800_dj,
            PT_800_1000_dj,
            PT_1000_1500_dj,
            N_PT_BINS_dj
        } BINS_dj;        
        enum {
            PT_125_150_vj=0,
            PT_150_220_vj,            
            PT_220_300_vj,
            PT_300_450_vj,
            N_PT_BINS_vj
        } BINS_vj;
        
        AIDA::IHistogram1D * _h_ungroomedJet0pt, * _h_ungroomedJet1pt;
        
        AIDA::IHistogram1D * _h_ungroomedAvgJetMass_dj[N_PT_BINS_dj];
        AIDA::IHistogram1D * _h_filteredAvgJetMass_dj[N_PT_BINS_dj];
        AIDA::IHistogram1D * _h_trimmedAvgJetMass_dj[N_PT_BINS_dj];
        AIDA::IHistogram1D * _h_prunedAvgJetMass_dj[N_PT_BINS_dj];        
        
        AIDA::IHistogram1D * _h_ungroomedJetMass_AK7_wj[N_PT_BINS_vj];
        AIDA::IHistogram1D * _h_filteredJetMass_AK7_wj[N_PT_BINS_vj];
        AIDA::IHistogram1D * _h_trimmedJetMass_AK7_wj[N_PT_BINS_vj];
        AIDA::IHistogram1D * _h_prunedJetMass_AK7_wj[N_PT_BINS_vj];
        AIDA::IHistogram1D * _h_prunedJetMass_CA8_wj[N_PT_BINS_vj];        
        AIDA::IHistogram1D * _h_filteredJetMass_CA12_wj[N_PT_BINS_vj-1];        
        
        AIDA::IHistogram1D * _h_ungroomedJetMass_AK7_zj[N_PT_BINS_vj];
        AIDA::IHistogram1D * _h_filteredJetMass_AK7_zj[N_PT_BINS_vj];
        AIDA::IHistogram1D * _h_trimmedJetMass_AK7_zj[N_PT_BINS_vj];
        AIDA::IHistogram1D * _h_prunedJetMass_AK7_zj[N_PT_BINS_vj];
        AIDA::IHistogram1D * _h_prunedJetMass_CA8_zj[N_PT_BINS_vj];        
        AIDA::IHistogram1D * _h_filteredJetMass_CA12_zj[N_PT_BINS_vj-1];         
        
        //@}
        
        
    };
    
    
    
    // The hook for the plugin system
    DECLARE_RIVET_PLUGIN(CMS_2013_I1224539);
    
}
