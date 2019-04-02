#ifndef RecoBTag_FeatureTools_TrackPairInfoBuilder_h
#define RecoBTag_FeatureTools_TrackPairInfoBuilder_h

#include "DataFormats/GeometrySurface/interface/Line.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/IPTools/interface/IPTools.h"
#include "TrackingTools/PatternTools/interface/TwoTrackMinimumDistance.h"


namespace btagbtvdeep{

class TrackPairInfoBuilder{
public:
    TrackPairInfoBuilder():
    
    
        track_pt_(0),
        track_eta_(0),
        track_phi_(0),
        track_dz_(0),
        track_dxy_(0),
                
        pca_distance_(0),
        pca_significance_(0),
        
        pcaSeed_x_(0),
        pcaSeed_y_(0),
        pcaSeed_z_(0),
        pcaSeed_xerr_(0),
        pcaSeed_yerr_(0),
        pcaSeed_zerr_(0),
        pcaTrack_x_(0),
        pcaTrack_y_(0),
        pcaTrack_z_(0),
        pcaTrack_xerr_(0),
        pcaTrack_yerr_(0),
        pcaTrack_zerr_(0),
        
        dotprodTrack_(0),
        dotprodSeed_(0),
        pcaSeed_dist_(0),
        pcaTrack_dist_(0),
        
        track_candMass_(0),
        track_ip2d_(0),
        track_ip2dSig_(0),
        track_ip3d_(0),
        track_ip3dSig_(0),
        
        dotprodTrackSeed2D_(0),
        dotprodTrackSeed2DV_(0),
        dotprodTrackSeed3D_(0),
        dotprodTrackSeed3DV_(0),
        
        pca_jetAxis_dist_(0),
        pca_jetAxis_dotprod_(0),
        pca_jetAxis_dEta_(0),
        pca_jetAxis_dPhi_(0)
        

{


}

    void buildTrackPairInfo(const reco::TransientTrack * it , const reco::TransientTrack * tt, const reco::Vertex & pv, float mass, GlobalVector jetdirection ){
        
        GlobalPoint pvp(pv.x(),pv.y(),pv.z());
        
        VertexDistance3D distanceComputer;
        TwoTrackMinimumDistance dist;
        
        auto const& iImpactState = it->impactPointState();
        auto const& tImpactState = tt->impactPointState();
        
        if(dist.calculate(tImpactState,iImpactState)) {
            
            GlobalPoint ttPoint          = dist.points().first;
            GlobalError ttPointErr       = tImpactState.cartesianError().position();
            GlobalPoint seedPosition     = dist.points().second;
            GlobalError seedPositionErr  = iImpactState.cartesianError().position();
            
            Measurement1D m = distanceComputer.distance(VertexState(seedPosition,seedPositionErr), VertexState(ttPoint, ttPointErr));
            
            GlobalPoint cp(dist.crossingPoint()); 

            GlobalVector PairMomentum((Basic3DVector<float>) (it->track().momentum()+tt->track().momentum()));
//             it->track().px()+tt->track().px(), it->track().py()+tt->track().py(), it->track().pz()+tt->track().pz()
            GlobalVector  PCA_pv(cp-pvp);

            float PCAseedFromPV =  (dist.points().second-pvp).mag();
            float PCAtrackFromPV =  (dist.points().first-pvp).mag();               
            float distance = dist.distance();

            GlobalVector trackDir2D(tImpactState.globalDirection().x(),tImpactState.globalDirection().y(),0.); 
            GlobalVector seedDir2D(iImpactState.globalDirection().x(),iImpactState.globalDirection().y(),0.); 
            GlobalVector trackPCADir2D(dist.points().first.x()-pvp.x(),dist.points().first.y()-pvp.y(),0.); 
            GlobalVector seedPCADir2D(dist.points().second.x()-pvp.x(),dist.points().second.y()-pvp.y(),0.); 

            float dotprodTrack = (dist.points().first-pvp).unit().dot(tImpactState.globalDirection().unit());
            float dotprodSeed = (dist.points().second-pvp).unit().dot(iImpactState.globalDirection().unit());                    
    
            std::pair<bool,Measurement1D> t_ip = IPTools::absoluteImpactParameter3D(*tt,pv);        
            std::pair<bool,Measurement1D> t_ip2d = IPTools::absoluteTransverseImpactParameter(*tt,pv);             

            Line::PositionType pos(pvp);
            Line::DirectionType dir(jetdirection);
            Line::DirectionType pairMomentumDir(PairMomentum);
            Line jetLine(pos,dir);   
            Line PCAMomentumLine(cp,pairMomentumDir);

           
            track_pt_=tt->track().pt();
            track_eta_=tt->track().eta();
            track_phi_=tt->track().phi();
            track_dz_=tt->track().dz(pv.position());
            track_dxy_=tt->track().dxy(pv.position());


            pca_distance_=distance;
            pca_significance_=m.significance();

            pcaSeed_x_=seedPosition.x();
            pcaSeed_y_=seedPosition.y();
            pcaSeed_z_=seedPosition.z();
            pcaSeed_xerr_=seedPositionErr.cxx();
            pcaSeed_yerr_=seedPositionErr.cyy();
            pcaSeed_zerr_=seedPositionErr.czz();
            pcaTrack_x_=ttPoint.x();
            pcaTrack_y_=ttPoint.y();
            pcaTrack_z_=ttPoint.z();
            pcaTrack_xerr_=ttPointErr.cxx();
            pcaTrack_yerr_=ttPointErr.cyy();
            pcaTrack_zerr_=ttPointErr.czz();

            dotprodTrack_=dotprodTrack;
            dotprodSeed_=dotprodSeed;
            pcaSeed_dist_=PCAseedFromPV;
            pcaTrack_dist_=PCAtrackFromPV;

            track_candMass_=mass;  
            track_ip2d_=t_ip2d.second.value();
            track_ip2dSig_=t_ip2d.second.significance();
            track_ip3d_=t_ip.second.value();
            track_ip3dSig_=t_ip.second.significance();

            dotprodTrackSeed2D_=trackDir2D.unit().dot(seedDir2D.unit());
            dotprodTrackSeed3D_=iImpactState.globalDirection().unit().dot(tImpactState.globalDirection().unit());
            dotprodTrackSeed2DV_=trackPCADir2D.unit().dot(seedPCADir2D.unit());
            dotprodTrackSeed3DV_=(dist.points().second-pvp).unit().dot((dist.points().first-pvp).unit());

            pca_jetAxis_dist_=jetLine.distance(cp).mag();
            pca_jetAxis_dotprod_=PairMomentum.unit().dot(jetdirection.unit());
            pca_jetAxis_dEta_=std::fabs(PCA_pv.eta()-jetdirection.eta());
            pca_jetAxis_dPhi_=std::fabs(PCA_pv.phi()-jetdirection.phi());


    }
    }

   
    const float track_pt() const {return track_pt_;}
    const float track_eta() const {return track_eta_;}
    const float track_phi() const {return track_phi_;}
    const float track_dz() const {return track_dz_;}
    const float track_dxy() const {return track_dxy_;}
    const float pca_distance() const {return pca_distance_;}
    const float pca_significance() const {return pca_significance_; }   
    const float pcaSeed_x() const {return pcaSeed_x_;}
    const float pcaSeed_y() const {return pcaSeed_y_;}
    const float pcaSeed_z() const {return pcaSeed_z_;}
    const float pcaSeed_xerr() const {return pcaSeed_xerr_;}
    const float pcaSeed_yerr() const {return pcaSeed_yerr_;}
    const float pcaSeed_zerr() const {return pcaSeed_zerr_;}
    const float pcaTrack_x() const {return pcaTrack_x_;}
    const float pcaTrack_y() const {return pcaTrack_y_;}
    const float pcaTrack_z() const {return pcaTrack_z_;}
    const float pcaTrack_xerr() const {return pcaTrack_xerr_;}
    const float pcaTrack_yerr() const {return pcaTrack_yerr_;}
    const float pcaTrack_zerr() const {return pcaTrack_zerr_;}    
    const float dotprodTrack() const {return dotprodTrack_;}
    const float dotprodSeed() const {return dotprodSeed_;}
    const float pcaSeed_dist() const {return pcaSeed_dist_;}
    const float pcaTrack_dist() const {return pcaTrack_dist_;}    
    const float track_candMass() const {return track_candMass_;}
    const float track_ip2d() const {return track_ip2d_;}
    const float track_ip2dSig() const {return track_ip2dSig_;}
    const float track_ip3d() const {return track_ip3d_;}
    const float track_ip3dSig() const {return track_ip3dSig_; }   
    const float dotprodTrackSeed2D() const {return dotprodTrackSeed2D_;}
    const float dotprodTrackSeed2DV() const {return dotprodTrackSeed2DV_;}
    const float dotprodTrackSeed3D() const {return dotprodTrackSeed3D_;}
    const float dotprodTrackSeed3DV() const {return dotprodTrackSeed3DV_;}    
    const float pca_jetAxis_dist() const {return pca_jetAxis_dist_;}
    const float pca_jetAxis_dotprod() const {return pca_jetAxis_dotprod_;}
    const float pca_jetAxis_dEta() const {return pca_jetAxis_dEta_;}
    const float pca_jetAxis_dPhi() const {return pca_jetAxis_dPhi_;}
    
    


private:

    float track_pt_;
    float track_eta_;
    float track_phi_;
    float track_dz_;
    float track_dxy_;
    float pca_distance_;
    float pca_significance_;    
    float pcaSeed_x_;
    float pcaSeed_y_;
    float pcaSeed_z_;
    float pcaSeed_xerr_;
    float pcaSeed_yerr_;
    float pcaSeed_zerr_;
    float pcaTrack_x_;
    float pcaTrack_y_;
    float pcaTrack_z_;
    float pcaTrack_xerr_;
    float pcaTrack_yerr_;
    float pcaTrack_zerr_;    
    float dotprodTrack_;
    float dotprodSeed_;
    float pcaSeed_dist_;
    float pcaTrack_dist_;    
    float track_candMass_;
    float track_ip2d_;
    float track_ip2dSig_;
    float track_ip3d_;
    float track_ip3dSig_;    
    float dotprodTrackSeed2D_;
    float dotprodTrackSeed2DV_;
    float dotprodTrackSeed3D_;
    float dotprodTrackSeed3DV_;    
    float pca_jetAxis_dist_;
    float pca_jetAxis_dotprod_;
    float pca_jetAxis_dEta_;
    float pca_jetAxis_dPhi_;

};

}

#endif //RecoBTag_FeatureTools_TrackPairInfoBuilder_h
