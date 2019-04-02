#ifndef RecoBTag_FeatureTools_SeedingTracksConverter_h
#define RecoBTag_FeatureTools_SeedingTracksConverter_h

#include "RecoBTag/FeatureTools/interface/deep_helpers.h"
#include "RecoBTag/FeatureTools/interface/TrackInfoBuilder.h"

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

namespace btagbtvdeep {
    
        static void seedingTracksToFeatures(edm::Handle<edm::View<reco::Candidate> > tracks,
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
                
                unsigned int k=track - tracks->begin();
                
                if((*tracks)[k].bestTrack() != nullptr &&  (*tracks)[k].pt()>0.5) 
                { 
                    
                    if (std::fabs(pv.z()-track_builder->build(tracks->ptrAt(k)).track().vz())<0.5)
                    {
                        selectedTracks.push_back(track_builder->build(tracks->ptrAt(k)));
                        masses.push_back(tracks->ptrAt(k)->mass());
                                                
                    }
                    
                }
                
            }
                
            std::multimap<double,std::pair<btagbtvdeep::SeedingTrackInfoBuilder,std::vector<btagbtvdeep::TrackPairFeatures>>> sortedSeedsMap;
            std::multimap<double,btagbtvdeep::TrackPairInfoBuilder> sortedNeighboursMap;
            
            std::vector<btagbtvdeep::TrackPairFeatures> tp_features_vector;            
            
            sortedSeedsMap.clear();
            seedingT_features_vector.clear();
            
            //for(auto const& it : selectedTracks){
            for(std::vector<reco::TransientTrack>::const_iterator it = selectedTracks.begin(); it != selectedTracks.end(); it++){
                
                sortedNeighboursMap.clear();
                tp_features_vector.clear();        
                
                if (reco::deltaR(it->track(), jet) > 0.4) continue;
                
                std::pair<bool,Measurement1D> ip = IPTools::absoluteImpactParameter3D(*it, pv);        
                std::pair<bool,Measurement1D> ip2d = IPTools::absoluteTransverseImpactParameter(*it, pv);
                std::pair<double, Measurement1D> jet_dist =IPTools::jetTrackDistance(*it, jetdirection, pv); 
                TrajectoryStateOnSurface closest = IPTools::closestApproachToJet(it->impactPointState(),pv, jetdirection,it->field());
                float length=999;
                if (closest.isValid()) length=(closest.globalPosition() - pvp).mag();
                
                
                if (!(ip.first && ip.second.value() >= 0.0 && 
                      ip.second.significance() >= 1.0 && 
                      ip.second.value() <= 9999. && 
                      ip.second.significance() <= 9999. && 
                      it->track().normalizedChi2()<5. && 
                      std::fabs(it->track().dxy(pv.position())) < 2 && 
                      std::fabs(it->track().dz(pv.position())) < 17 && 
                      jet_dist.second.value()<0.07 && 
                      length<5. )) 
                    continue;                
                
                
                btagbtvdeep::SeedingTrackInfoBuilder seedInfo;
                seedInfo.buildSeedingTrackInfo(&(*it), pv, jet, masses[it-selectedTracks.begin()], probabilityEstimator, computeProbabilities);
                
                for(std::vector<reco::TransientTrack>::const_iterator tt = selectedTracks.begin();tt!=selectedTracks.end(); ++tt )   {
                    
                    if(*tt==*it) continue;
                    if(std::fabs(pv.z()-tt->track().vz())>0.1) continue;

                    btagbtvdeep::TrackPairInfoBuilder trackPairInfo;
                    trackPairInfo.buildTrackPairInfo(&(*it),&(*tt),pv, masses[tt-selectedTracks.begin()],jetdirection);
                    sortedNeighboursMap.insert(std::make_pair(trackPairInfo.pca_distance(), trackPairInfo));
                    
                }
      
                int max_counter=0;
                
                //for(std::multimap<double,btagbtvdeep::TrackPairInfoBuilder>::const_iterator im = sortedNeighboursMap.begin(); im != sortedNeighboursMap.end(); im++){
                for(auto const& im: sortedNeighboursMap){
                
                                
                    if(max_counter>=20) break;
                    btagbtvdeep::TrackPairFeatures tp_features;
                                
                    
                    int logOffset=0;
                    
                    tp_features.pt=(im.second.track_pt()==0) ? 0: 1.0/im.second.track_pt();  //im.second.track_pt();
                    tp_features.eta=im.second.track_eta();
                    tp_features.phi=im.second.track_phi();
                    tp_features.mass=im.second.track_candMass();
                    tp_features.dz=(logOffset+log(fabs(im.second.track_dz())))*((im.second.track_dz() < 0) ? -1 : (im.second.track_dz() > 0));  //im.second.track_dz();
                    tp_features.dxy=(logOffset+log(fabs(im.second.track_dxy())))*((im.second.track_dxy() < 0) ? -1 : (im.second.track_dxy() > 0));  //im.second.track_dxy();
                    tp_features.ip3D=log(im.second.track_ip3d());
                    tp_features.sip3D=log(im.second.track_ip3dSig());
                    tp_features.ip2D=log(im.second.track_ip2d());
                    tp_features.sip2D=log(im.second.track_ip2dSig());
                    tp_features.distPCA=log(im.second.pca_distance());
                    tp_features.dsigPCA=log(im.second.pca_significance());     
                    tp_features.x_PCAonSeed=im.second.pcaSeed_x();
                    tp_features.y_PCAonSeed=im.second.pcaSeed_y();
                    tp_features.z_PCAonSeed=im.second.pcaSeed_z();      
                    tp_features.xerr_PCAonSeed=im.second.pcaSeed_xerr();
                    tp_features.yerr_PCAonSeed=im.second.pcaSeed_yerr();
                    tp_features.zerr_PCAonSeed=im.second.pcaSeed_zerr();      
                    tp_features.x_PCAonTrack=im.second.pcaTrack_x();
                    tp_features.y_PCAonTrack=im.second.pcaTrack_y();
                    tp_features.z_PCAonTrack=im.second.pcaTrack_z();      
                    tp_features.xerr_PCAonTrack=im.second.pcaTrack_xerr();
                    tp_features.yerr_PCAonTrack=im.second.pcaTrack_yerr();
                    tp_features.zerr_PCAonTrack=im.second.pcaTrack_zerr(); 
                    tp_features.dotprodTrack=im.second.dotprodTrack();
                    tp_features.dotprodSeed=im.second.dotprodSeed();
                    tp_features.dotprodTrackSeed2D=im.second.dotprodTrackSeed2D();
                    tp_features.dotprodTrackSeed3D=im.second.dotprodTrackSeed3D();
                    tp_features.dotprodTrackSeedVectors2D=im.second.dotprodTrackSeed2DV();
                    tp_features.dotprodTrackSeedVectors3D=im.second.dotprodTrackSeed3DV();      
                    tp_features.pvd_PCAonSeed=log(im.second.pcaSeed_dist());
                    tp_features.pvd_PCAonTrack=log(im.second.pcaTrack_dist());
                    tp_features.dist_PCAjetAxis=log(im.second.pca_jetAxis_dist());
                    tp_features.dotprod_PCAjetMomenta=im.second.pca_jetAxis_dotprod();
                    tp_features.deta_PCAjetDirs=log(im.second.pca_jetAxis_dEta());
                    tp_features.dphi_PCAjetDirs=im.second.pca_jetAxis_dPhi();               
                    
                    
                    max_counter=max_counter+1;
                    tp_features_vector.push_back(tp_features);             
                    
                    
                } 
                
                sortedSeedsMap.insert(std::make_pair(-seedInfo.sip3d_Signed(), std::make_pair(seedInfo,tp_features_vector)));
                
                    
            }
            
            
            
            
            int max_counter_seed=0;
                
            //for(std::multimap<double,std::pair<btagbtvdeep::SeedingTrackInfoBuilder,std::vector<btagbtvdeep::TrackPairFeatures>>>::const_iterator im = sortedSeedsMap.begin(); im != sortedSeedsMap.end(); im++){
            for(auto const& im: sortedSeedsMap){
                
                if(max_counter_seed>=10) break;
                
                btagbtvdeep::SeedingTrackFeatures seed_features;            
                
                int logOffset=0;
                
                seed_features.nearTracks=im.second.second;
                seed_features.pt=(im.second.first.pt()==0) ? 0: 1.0/im.second.first.pt();
                seed_features.eta=im.second.first.eta();
                seed_features.phi=im.second.first.phi();
                seed_features.mass=im.second.first.mass();                
                seed_features.dz=(logOffset+log(fabs(im.second.first.dz())))*((im.second.first.dz() < 0) ? -1 : (im.second.first.dz() > 0));
                seed_features.dxy=(logOffset+log(fabs(im.second.first.dxy())))*((im.second.first.dxy() < 0) ? -1 : (im.second.first.dxy() > 0));
                seed_features.ip3D=log(im.second.first.ip3d());
                seed_features.sip3D=log(im.second.first.sip3d());
                seed_features.ip2D=log(im.second.first.ip2d());
                seed_features.sip2D=log(im.second.first.sip2d()); 
                seed_features.signedIp3D=(logOffset+log(fabs(im.second.first.ip3d_Signed())))*((im.second.first.ip3d_Signed() < 0) ? -1 : (im.second.first.ip3d_Signed() > 0));
                seed_features.signedSip3D=(logOffset+log(fabs(im.second.first.sip3d_Signed())))*((im.second.first.sip3d_Signed() < 0) ? -1 : (im.second.first.sip3d_Signed() > 0));
                seed_features.signedIp2D=(logOffset+log(fabs(im.second.first.ip2d_Signed())))*((im.second.first.ip2d_Signed() < 0) ? -1 : (im.second.first.ip2d_Signed() > 0));
                seed_features.signedSip2D=(logOffset+log(fabs(im.second.first.sip2d_Signed())))*((im.second.first.sip2d_Signed() < 0) ? -1 : (im.second.first.sip2d_Signed() > 0));
                seed_features.trackProbability3D=im.second.first.trackProbability3D();
                seed_features.trackProbability2D=im.second.first.trackProbability2D();
                seed_features.chi2reduced=im.second.first.chi2reduced();
                seed_features.nPixelHits=im.second.first.nPixelHits();
                seed_features.nHits=im.second.first.nHits();
                seed_features.jetAxisDistance=log(im.second.first.jetAxisDistance());
                seed_features.jetAxisDlength=log(im.second.first.jetAxisDlength());
                
        
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

#endif //RecoBTag_FeatureTools_SeedingTracksConverter_h
