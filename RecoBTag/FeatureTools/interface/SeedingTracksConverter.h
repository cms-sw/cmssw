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
class HistogramProbabilityEstimator;

namespace btagbtvdeep {
  
  class SeedingTracksConverter {

    public:

      static void SeedingTracksToFeatures(std::vector<btagbtvdeep::SeedingTrackFeatures> & seedingT_features_vector,
                                            edm::Handle<edm::View<pat::PackedCandidate> > tracks,
                                            const reco::Jet & jet,
                                            const reco::Vertex & pv,                                            
                                            edm::ESHandle<TransientTrackBuilder> & track_builder,
                                            
                                            HistogramProbabilityEstimator* m_probabilityEstimator,
                                            bool m_computeProbabilities
                                            
                                            ) 
      {

            GlobalVector jetdirection(jet.px(),jet.py(),jet.pz());
            GlobalPoint pvp(pv.x(),pv.y(),pv.z());
                        
            std::vector<reco::TransientTrack> selectedTracks;          
            std::vector<float> masses;
                
            for(typename edm::View<pat::PackedCandidate>::const_iterator track = tracks->begin(); track != tracks->end(); ++track) {
                
                    unsigned int k=track - tracks->begin(); 
                    
                    if((*tracks)[k].bestTrack() != nullptr &&  (*tracks)[k].pt()>0.5) {
                        if (std::fabs(pv.z()-track_builder->build(tracks->ptrAt(k)).track().vz())<0.5){
                            selectedTracks.push_back(track_builder->build(tracks->ptrAt(k)));
                            masses.push_back(tracks->ptrAt(k)->mass());
                            
                            
                        }
                    }
            }
                
            std::multimap<double,std::pair<btagbtvdeep::SeedingTrackInfoBuilder,std::vector<btagbtvdeep::TrackPairFeatures>>> SortedSeedsMap;
            std::multimap<double,btagbtvdeep::TrackPairInfoBuilder> SortedNeighboursMap;
            
            std::vector<btagbtvdeep::TrackPairFeatures> tp_features_vector;            
            
            SortedSeedsMap.clear();
            seedingT_features_vector.clear();
            
            for(std::vector<reco::TransientTrack>::const_iterator it = selectedTracks.begin(); it != selectedTracks.end(); it++){
                
                SortedNeighboursMap.clear();
                tp_features_vector.clear();        
                
                if (reco::deltaR(it->track(), jet) > 0.4) continue;
                
                std::pair<bool,Measurement1D> ip = IPTools::absoluteImpactParameter3D(*it, pv);        
                std::pair<bool,Measurement1D> ip2d = IPTools::absoluteTransverseImpactParameter(*it, pv);
                std::pair<double, Measurement1D> jet_dist =IPTools::jetTrackDistance(*it, jetdirection, pv); 
                TrajectoryStateOnSurface closest = IPTools::closestApproachToJet(it->impactPointState(),pv, jetdirection,it->field());
                float length=999;
                if (closest.isValid()) length=(closest.globalPosition() - pvp).mag();
                
                if (!(ip.first && ip.second.value() >= 0.0 && ip.second.significance() >= 1.0 && ip.second.value() <= 9999. && ip.second.significance() <= 9999. &&
                    it->track().normalizedChi2()<5. && std::fabs(it->track().dxy(pv.position())) < 2 && std::fabs(it->track().dz(pv.position())) < 17 && jet_dist.second.value()<0.07 && length<5. )) continue;                
                
                
                btagbtvdeep::SeedingTrackInfoBuilder seedInfo;
                seedInfo.buildSeedingTrackInfo(&(*it), pv, jet, masses[it-selectedTracks.begin()], m_probabilityEstimator, m_computeProbabilities);
                
                for(std::vector<reco::TransientTrack>::const_iterator tt = selectedTracks.begin();tt!=selectedTracks.end(); ++tt )   {

                if(*tt==*it) continue;
                if(std::fabs(pv.z()-tt->track().vz())>0.1) continue;

                btagbtvdeep::TrackPairInfoBuilder trackPairInfo;
                trackPairInfo.buildTrackPairInfo(&(*it),&(*tt),pv, masses[tt-selectedTracks.begin()],jetdirection);
                SortedNeighboursMap.insert(std::make_pair(trackPairInfo.get_pca_distance(), trackPairInfo));
                    
                }
      
                int max_counter=0;
                
                for(std::multimap<double,btagbtvdeep::TrackPairInfoBuilder>::const_iterator im = SortedNeighboursMap.begin(); im != SortedNeighboursMap.end(); im++){       
                
                                
                    if(max_counter>=20) break;
                    btagbtvdeep::TrackPairFeatures tp_features;
                                
                    
                    int N=0;
                    
                    tp_features.nearTracks_pt=(im->second.get_track_pt()==0) ? 0: 1.0/im->second.get_track_pt();  //im->second.get_track_pt();
                    tp_features.nearTracks_eta=im->second.get_track_eta();
                    tp_features.nearTracks_phi=im->second.get_track_phi();
                    tp_features.nearTracks_mass=im->second.get_track_candMass();
                    tp_features.nearTracks_dz=(N+log(fabs(im->second.get_track_dz())))*((im->second.get_track_dz() < 0) ? -1 : (im->second.get_track_dz() > 0));  //im->second.get_track_dz();
                    tp_features.nearTracks_dxy=(N+log(fabs(im->second.get_track_dxy())))*((im->second.get_track_dxy() < 0) ? -1 : (im->second.get_track_dxy() > 0));  //im->second.get_track_dxy();
                    tp_features.nearTracks_3D_ip=log(im->second.get_track_ip3d());
                    tp_features.nearTracks_3D_sip=log(im->second.get_track_ip3dSig());
                    tp_features.nearTracks_2D_ip=log(im->second.get_track_ip2d());
                    tp_features.nearTracks_2D_sip=log(im->second.get_track_ip2dSig());
                    tp_features.nearTracks_PCAdist=log(im->second.get_pca_distance());
                    tp_features.nearTracks_PCAdsig=log(im->second.get_pca_significance());     
                    tp_features.nearTracks_PCAonSeed_x=im->second.get_pcaSeed_x();
                    tp_features.nearTracks_PCAonSeed_y=im->second.get_pcaSeed_y();
                    tp_features.nearTracks_PCAonSeed_z=im->second.get_pcaSeed_z();      
                    tp_features.nearTracks_PCAonSeed_xerr=im->second.get_pcaSeed_xerr();
                    tp_features.nearTracks_PCAonSeed_yerr=im->second.get_pcaSeed_yerr();
                    tp_features.nearTracks_PCAonSeed_zerr=im->second.get_pcaSeed_zerr();      
                    tp_features.nearTracks_PCAonTrack_x=im->second.get_pcaTrack_x();
                    tp_features.nearTracks_PCAonTrack_y=im->second.get_pcaTrack_y();
                    tp_features.nearTracks_PCAonTrack_z=im->second.get_pcaTrack_z();      
                    tp_features.nearTracks_PCAonTrack_xerr=im->second.get_pcaTrack_xerr();
                    tp_features.nearTracks_PCAonTrack_yerr=im->second.get_pcaTrack_yerr();
                    tp_features.nearTracks_PCAonTrack_zerr=im->second.get_pcaTrack_zerr(); 
                    tp_features.nearTracks_dotprodTrack=im->second.get_dotprodTrack();
                    tp_features.nearTracks_dotprodSeed=im->second.get_dotprodSeed();
                    tp_features.nearTracks_dotprodTrackSeed2D=im->second.get_dotprodTrackSeed2D();
                    tp_features.nearTracks_dotprodTrackSeed3D=im->second.get_dotprodTrackSeed3D();
                    tp_features.nearTracks_dotprodTrackSeedVectors2D=im->second.get_dotprodTrackSeed2DV();
                    tp_features.nearTracks_dotprodTrackSeedVectors3D=im->second.get_dotprodTrackSeed3DV();      
                    tp_features.nearTracks_PCAonSeed_pvd=log(im->second.get_pcaSeed_dist());
                    tp_features.nearTracks_PCAonTrack_pvd=log(im->second.get_pcaTrack_dist());
                    tp_features.nearTracks_PCAjetAxis_dist=log(im->second.get_pca_jetAxis_dist());
                    tp_features.nearTracks_PCAjetMomenta_dotprod=im->second.get_pca_jetAxis_dotprod();
                    tp_features.nearTracks_PCAjetDirs_DEta=log(im->second.get_pca_jetAxis_dEta());
                    tp_features.nearTracks_PCAjetDirs_DPhi=im->second.get_pca_jetAxis_dPhi();               
                    
                    
                    max_counter=max_counter+1;
                    tp_features_vector.push_back(tp_features);             
                    
                    
                } 
                
                SortedSeedsMap.insert(std::make_pair(-seedInfo.get_sip3d_Signed(), std::make_pair(seedInfo,tp_features_vector)));
                
                    
            }
            
            
            
            
            int max_counter_seed=0;
                
            for(std::multimap<double,std::pair<btagbtvdeep::SeedingTrackInfoBuilder,std::vector<btagbtvdeep::TrackPairFeatures>>>::const_iterator im = SortedSeedsMap.begin(); im != SortedSeedsMap.end(); im++){
                    
                if(max_counter_seed>=10) break;
                
                btagbtvdeep::SeedingTrackFeatures seed_features;            
                
                int N=0;
                
                seed_features.seed_nearTracks=im->second.second;
                seed_features.seed_pt=(im->second.first.get_pt()==0) ? 0: 1.0/im->second.first.get_pt();
                seed_features.seed_eta=im->second.first.get_eta();
                seed_features.seed_phi=im->second.first.get_phi();
                seed_features.seed_mass=im->second.first.get_mass();                
                seed_features.seed_dz=(N+log(fabs(im->second.first.get_dz())))*((im->second.first.get_dz() < 0) ? -1 : (im->second.first.get_dz() > 0));
                seed_features.seed_dxy=(N+log(fabs(im->second.first.get_dxy())))*((im->second.first.get_dxy() < 0) ? -1 : (im->second.first.get_dxy() > 0));
                seed_features.seed_3D_ip=log(im->second.first.get_ip3d());
                seed_features.seed_3D_sip=log(im->second.first.get_sip3d());
                seed_features.seed_2D_ip=log(im->second.first.get_ip2d());
                seed_features.seed_2D_sip=log(im->second.first.get_sip2d()); 
                seed_features.seed_3D_signedIp=(N+log(fabs(im->second.first.get_ip3d_Signed())))*((im->second.first.get_ip3d_Signed() < 0) ? -1 : (im->second.first.get_ip3d_Signed() > 0));
                seed_features.seed_3D_signedSip=(N+log(fabs(im->second.first.get_sip3d_Signed())))*((im->second.first.get_sip3d_Signed() < 0) ? -1 : (im->second.first.get_sip3d_Signed() > 0));
                seed_features.seed_2D_signedIp=(N+log(fabs(im->second.first.get_ip2d_Signed())))*((im->second.first.get_ip2d_Signed() < 0) ? -1 : (im->second.first.get_ip2d_Signed() > 0));
                seed_features.seed_2D_signedSip=(N+log(fabs(im->second.first.get_sip2d_Signed())))*((im->second.first.get_sip2d_Signed() < 0) ? -1 : (im->second.first.get_sip2d_Signed() > 0));
                seed_features.seed_3D_TrackProbability=im->second.first.get_trackProbability3D();
                seed_features.seed_2D_TrackProbability=im->second.first.get_trackProbability2D();
                seed_features.seed_chi2reduced=im->second.first.get_chi2reduced();
                seed_features.seed_nPixelHits=im->second.first.get_nPixelHits();
                seed_features.seed_nHits=im->second.first.get_nHits();
                seed_features.seed_jetAxisDistance=log(im->second.first.get_jetAxisDistance());
                seed_features.seed_jetAxisDlength=log(im->second.first.get_jetAxisDlength());
                
        
                max_counter_seed=max_counter_seed+1;
                seedingT_features_vector.push_back(seed_features);

                
            }
            
            
            if (SortedSeedsMap.size()<10){
                
                for (unsigned int i=SortedSeedsMap.size(); i<10; i++){
                    
                    
                    std::vector<btagbtvdeep::TrackPairFeatures> tp_features_zeropad(20);
                    btagbtvdeep::SeedingTrackFeatures seed_features_zeropad;
                    seed_features_zeropad.seed_nearTracks=tp_features_zeropad;
                    seedingT_features_vector.push_back(seed_features_zeropad);
                    
                }    
                
            }          
          
    }
      
};
    
}

#endif //RecoBTag_FeatureTools_SeedingTracksConverter_h
