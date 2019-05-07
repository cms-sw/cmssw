#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

#include "DataFormats/BTauReco/interface/SeedingTrackFeatures.h"
#include "DataFormats/BTauReco/interface/TrackPairFeatures.h"

#include "RecoBTag/FeatureTools/interface/TrackPairInfoBuilder.h"
#include "RecoBTag/FeatureTools/interface/SeedingTrackInfoBuilder.h"

#include "DataFormats/PatCandidates/interface/Jet.h"

#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/IPTools/interface/IPTools.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "RecoVertex/VertexTools/interface/VertexDistance3D.h"

#include "RecoBTag/TrackProbability/interface/HistogramProbabilityEstimator.h"

#include "RecoBTag/FeatureTools/interface/SeedingTracksConverter.h"


namespace btagbtvdeep {
    
        void seedingTracksToFeatures(edm::Handle<edm::View<reco::Candidate> > tracks,
                                        const reco::Jet & jet,
                                        const reco::Vertex & pv,                                            
                                        edm::ESHandle<TransientTrackBuilder> & track_builder,
                                        HistogramProbabilityEstimator* probabilityEstimator,
                                        bool computeProbabilities,
                                        std::vector<btagbtvdeep::SeedingTrackFeatures> & seedingT_features_vector
                                        ) 
        {
            
            GlobalVector jetdirection(jet.px(),jet.py(),jet.pz());
            GlobalPoint pvp(pv.x(),pv.y(),pv.z());
                        
            std::vector<reco::TransientTrack> selectedTracks;          
            std::vector<float> masses;
                
            for(typename edm::View<reco::Candidate>::const_iterator track = tracks->begin(); track != tracks->end(); ++track) {
                
                unsigned int k = track - tracks->begin();
                
                if(track->bestTrack() != nullptr && track->pt()>0.5) 
                { 

                    if (std::fabs(track->vz()-pv.z())<0.5)
                    {
                        selectedTracks.push_back(track_builder->build(tracks->ptrAt(k)));
                        masses.push_back(track->mass());
                                                
                    }
                    
                }
                
            }
                
            std::multimap<double,std::pair<btagbtvdeep::SeedingTrackInfoBuilder,std::vector<btagbtvdeep::TrackPairFeatures>>> sortedSeedsMap;
            std::multimap<double,btagbtvdeep::TrackPairInfoBuilder> sortedNeighboursMap;
            
            std::vector<btagbtvdeep::TrackPairFeatures> tp_features_vector;            
            
            sortedSeedsMap.clear();
            seedingT_features_vector.clear();
            
            unsigned int selTrackCount=0;

            for(auto const& it : selectedTracks){
                
                selTrackCount+=1;
                sortedNeighboursMap.clear();
                tp_features_vector.clear();        
                
                if (reco::deltaR(it.track(), jet) > 0.4) continue;
                
                std::pair<bool,Measurement1D> ip = IPTools::absoluteImpactParameter3D(it, pv);
                std::pair<bool,Measurement1D> ip2d = IPTools::absoluteTransverseImpactParameter(it, pv);
                std::pair<double, Measurement1D> jet_dist =IPTools::jetTrackDistance(it, jetdirection, pv); 
                TrajectoryStateOnSurface closest = IPTools::closestApproachToJet(it.impactPointState(),pv, jetdirection,it.field());
                float length=999;
                if (closest.isValid()) length=(closest.globalPosition() - pvp).mag();
                
                
                if (!(ip.first && ip.second.value() >= 0.0 &&
                      ip.second.significance() >= 1.0 && 
                      ip.second.value() <= 9999. &&
                      ip.second.significance() <= 9999. &&
                      it.track().normalizedChi2()<5. &&
                      std::fabs(it.track().dxy(pv.position())) < 2 &&
                      std::fabs(it.track().dz(pv.position())) < 17 &&
                      jet_dist.second.value()<0.07 &&
                      length<5. )) 
                    continue;                
                
                
                btagbtvdeep::SeedingTrackInfoBuilder seedInfo;
                seedInfo.buildSeedingTrackInfo(&(it), pv, jet, masses[selTrackCount-1], probabilityEstimator, computeProbabilities);

                unsigned int neighbourTrackCount=0;
                
                for(auto const& tt : selectedTracks){
                    
                    neighbourTrackCount+=1;

                    if(tt==it) continue;
                    if(std::fabs(pv.z()-tt.track().vz())>0.1) continue;

                    btagbtvdeep::TrackPairInfoBuilder trackPairInfo;
                    trackPairInfo.buildTrackPairInfo(&(it),&(tt),pv,masses[neighbourTrackCount-1],jetdirection);
                    sortedNeighboursMap.insert(std::make_pair(trackPairInfo.pca_distance(), trackPairInfo));
                    
                }
      
                int max_counter=0;

                for(auto const& im: sortedNeighboursMap){
                                
                    if(max_counter>=20) break;
                    btagbtvdeep::TrackPairFeatures tp_features;
                    
                    int logOffset=0;
                    auto const& tp = im.second;
                    
                    tp_features.pt=(tp.track_pt()==0) ? 0: 1.0/tp.track_pt();
                    tp_features.eta=tp.track_eta();
                    tp_features.phi=tp.track_phi();
                    tp_features.mass=tp.track_candMass();
                    tp_features.dz=(logOffset+log(fabs(tp.track_dz())))*((tp.track_dz() < 0) ? -1 : (tp.track_dz() > 0));
                    tp_features.dxy=(logOffset+log(fabs(tp.track_dxy())))*((tp.track_dxy() < 0) ? -1 : (tp.track_dxy() > 0));
                    tp_features.ip3D=log(tp.track_ip3d());
                    tp_features.sip3D=log(tp.track_ip3dSig());
                    tp_features.ip2D=log(tp.track_ip2d());
                    tp_features.sip2D=log(tp.track_ip2dSig());
                    tp_features.distPCA=log(tp.pca_distance());
                    tp_features.dsigPCA=log(tp.pca_significance());
                    tp_features.x_PCAonSeed=tp.pcaSeed_x();
                    tp_features.y_PCAonSeed=tp.pcaSeed_y();
                    tp_features.z_PCAonSeed=tp.pcaSeed_z();
                    tp_features.xerr_PCAonSeed=tp.pcaSeed_xerr();
                    tp_features.yerr_PCAonSeed=tp.pcaSeed_yerr();
                    tp_features.zerr_PCAonSeed=tp.pcaSeed_zerr();
                    tp_features.x_PCAonTrack=tp.pcaTrack_x();
                    tp_features.y_PCAonTrack=tp.pcaTrack_y();
                    tp_features.z_PCAonTrack=tp.pcaTrack_z();
                    tp_features.xerr_PCAonTrack=tp.pcaTrack_xerr();
                    tp_features.yerr_PCAonTrack=tp.pcaTrack_yerr();
                    tp_features.zerr_PCAonTrack=tp.pcaTrack_zerr();
                    tp_features.dotprodTrack=tp.dotprodTrack();
                    tp_features.dotprodSeed=tp.dotprodSeed();
                    tp_features.dotprodTrackSeed2D=tp.dotprodTrackSeed2D();
                    tp_features.dotprodTrackSeed3D=tp.dotprodTrackSeed3D();
                    tp_features.dotprodTrackSeedVectors2D=tp.dotprodTrackSeed2DV();
                    tp_features.dotprodTrackSeedVectors3D=tp.dotprodTrackSeed3DV();
                    tp_features.pvd_PCAonSeed=log(tp.pcaSeed_dist());
                    tp_features.pvd_PCAonTrack=log(tp.pcaTrack_dist());
                    tp_features.dist_PCAjetAxis=log(tp.pca_jetAxis_dist());
                    tp_features.dotprod_PCAjetMomenta=tp.pca_jetAxis_dotprod();
                    tp_features.deta_PCAjetDirs=log(tp.pca_jetAxis_dEta());
                    tp_features.dphi_PCAjetDirs=tp.pca_jetAxis_dPhi();
                    
                    
                    max_counter=max_counter+1;
                    tp_features_vector.push_back(tp_features);             
                    
                    
                } 

                sortedSeedsMap.insert(std::make_pair(-seedInfo.sip3d_Signed(), std::make_pair(seedInfo,tp_features_vector)));
            
            }


            int max_counter_seed=0;

            for(auto const& im: sortedSeedsMap){
                
                if(max_counter_seed>=10) break;
                
                btagbtvdeep::SeedingTrackFeatures seed_features;            
                
                int logOffset=0;
                auto const& seed = im.second.first;
                
                seed_features.nearTracks=im.second.second;
                seed_features.pt=(seed.pt()==0) ? 0: 1.0/seed.pt();
                seed_features.eta=seed.eta();
                seed_features.phi=seed.phi();
                seed_features.mass=seed.mass();
                seed_features.dz=(logOffset+log(fabs(seed.dz())))*((seed.dz() < 0) ? -1 : (seed.dz() > 0));
                seed_features.dxy=(logOffset+log(fabs(seed.dxy())))*((seed.dxy() < 0) ? -1 : (seed.dxy() > 0));
                seed_features.ip3D=log(seed.ip3d());
                seed_features.sip3D=log(seed.sip3d());
                seed_features.ip2D=log(seed.ip2d());
                seed_features.sip2D=log(seed.sip2d());
                seed_features.signedIp3D=(logOffset+log(fabs(seed.ip3d_Signed())))*((seed.ip3d_Signed() < 0) ? -1 : (seed.ip3d_Signed() > 0));
                seed_features.signedSip3D=(logOffset+log(fabs(seed.sip3d_Signed())))*((seed.sip3d_Signed() < 0) ? -1 : (seed.sip3d_Signed() > 0));
                seed_features.signedIp2D=(logOffset+log(fabs(seed.ip2d_Signed())))*((seed.ip2d_Signed() < 0) ? -1 : (seed.ip2d_Signed() > 0));
                seed_features.signedSip2D=(logOffset+log(fabs(seed.sip2d_Signed())))*((seed.sip2d_Signed() < 0) ? -1 : (seed.sip2d_Signed() > 0));
                seed_features.trackProbability3D=seed.trackProbability3D();
                seed_features.trackProbability2D=seed.trackProbability2D();
                seed_features.chi2reduced=seed.chi2reduced();
                seed_features.nPixelHits=seed.nPixelHits();
                seed_features.nHits=seed.nHits();
                seed_features.jetAxisDistance=log(seed.jetAxisDistance());
                seed_features.jetAxisDlength=log(seed.jetAxisDlength());
                
        
                max_counter_seed=max_counter_seed+1;
                seedingT_features_vector.push_back(seed_features);

                
            }
            
            
            if (sortedSeedsMap.size()<10){
                
                for (unsigned int i=sortedSeedsMap.size(); i<10; i++){
                    
                    
                    std::vector<btagbtvdeep::TrackPairFeatures> tp_features_zeropad(20);
                    btagbtvdeep::SeedingTrackFeatures seed_features_zeropad;
                    seed_features_zeropad.nearTracks=tp_features_zeropad;
                    seedingT_features_vector.push_back(seed_features_zeropad);
                    
                }    
                
            }          
          
    }
      
    
}
