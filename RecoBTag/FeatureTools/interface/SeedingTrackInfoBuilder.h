#ifndef RecoBTag_FeatureTools_SeedingTrackInfoBuilder_h
#define RecoBTag_FeatureTools_SeedingTrackInfoBuilder_h

#include "DataFormats/GeometrySurface/interface/Line.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/IPTools/interface/IPTools.h"
#include "TrackingTools/PatternTools/interface/TwoTrackMinimumDistance.h"

#include "RecoBTag/TrackProbability/interface/HistogramProbabilityEstimator.h"
class HistogramProbabilityEstimator;


namespace btagbtvdeep{
    
class SeedingTrackInfoBuilder{
public:
    SeedingTrackInfoBuilder():
    
    
        pt_(0),
        eta_(0),
        phi_(0),
        mass_(0),
        dz_(0),
        dxy_(0),
        ip3D_(0),
        sip3D_(0),
        ip2D_(0),
        sip2D_(0),
        ip3D_signed_(0),
        sip3D_signed_(0),
        ip2D_signed_(0),
        sip2D_signed_(0),
        chi2reduced_(0),
        nPixelHits_(0),
        nHits_(0),
        jetAxisDistance_(0),
        jetAxisDlength_(0),
        trackProbability3D_(0),
        trackProbability2D_(0)


{
}

    void buildSeedingTrackInfo(const reco::TransientTrack * it , const reco::Vertex & pv,  const reco::Jet & jet,/*GlobalVector jetdirection,*/ float mass, 
                              HistogramProbabilityEstimator* m_probabilityEstimator, bool m_computeProbabilities=false){
        
        GlobalPoint pvp(pv.x(),pv.y(),pv.z());
        GlobalVector jetdirection(jet.px(),jet.py(),jet.pz());
        
        pt_=it->track().pt();
        eta_=it->track().eta();
        phi_=it->track().phi();
        dz_=it->track().dz(pv.position());
        dxy_=it->track().dxy(pv.position());
        mass_=mass;        
        
        std::pair<bool,Measurement1D> ipSigned = IPTools::signedImpactParameter3D(*it,jetdirection, pv);        
        std::pair<bool,Measurement1D> ip2dSigned = IPTools::signedTransverseImpactParameter(*it,jetdirection, pv);  
        std::pair<bool,Measurement1D> ip = IPTools::absoluteImpactParameter3D(*it, pv);        
        std::pair<bool,Measurement1D> ip2d = IPTools::absoluteTransverseImpactParameter(*it, pv);        
        
        ip3D_=ip.second.value();
        sip3D_=ip.second.significance();
        ip2D_=ip2d.second.value();
        sip2D_=ip2d.second.significance();
        ip3D_signed_=ipSigned.second.value();
        sip3D_signed_=ipSigned.second.significance();
        ip2D_signed_=ip2dSigned.second.value();
        sip2D_signed_=ip2dSigned.second.significance();

        chi2reduced_=it->track().normalizedChi2();
        nPixelHits_=it->track().hitPattern().numberOfValidPixelHits();
        nHits_=it->track().hitPattern().numberOfValidHits();
        
        std::pair<double, Measurement1D> jet_distance =IPTools::jetTrackDistance(*it, jetdirection, pv);
        jetAxisDistance_=std::fabs(jet_distance.second.value());

        TrajectoryStateOnSurface closest = IPTools::closestApproachToJet(it->impactPointState(),pv,jetdirection,it->field());
        if (closest.isValid()) jetAxisDlength_=(closest.globalPosition() - pvp).mag(); 
        else jetAxisDlength_=-99;
        
        trackProbability3D_=0.5;
        trackProbability2D_=0.5;
        
        if (m_computeProbabilities) {
            
            //probability with 3D ip
            std::pair<bool,double> probability = m_probabilityEstimator->probability(false,0,ip.second.significance(),it->track(),jet,pv);
            double prob3D=(probability.first ? probability.second : -1.);
                        
            //probability with 2D ip                  
            probability = m_probabilityEstimator->probability(false,1,ip2d.second.significance(),it->track(),jet,pv);
            double prob2D=(probability.first ? probability.second : -1.);
                                
            trackProbability3D_=prob3D;
            trackProbability2D_=prob2D;            
            
        } 
        
        if (!(trackProbability3D_==trackProbability3D_)) trackProbability3D_=0.5;
        if (!(trackProbability2D_==trackProbability2D_)) trackProbability2D_=0.5;


    }
    
    
    const float get_pt() const {return pt_;}
    const float get_eta() const {return eta_;}
    const float get_phi() const {return phi_;}
    const float get_mass() const {return mass_;}
    const float get_dz() const {return dz_;}
    const float get_dxy() const {return dxy_;}
    const float get_ip3d() const {return ip3D_;}
    const float get_sip3d() const {return sip3D_;}
    const float get_ip2d() const {return ip2D_;}
    const float get_sip2d() const {return sip2D_;}
    const float get_ip3d_Signed() const {return ip3D_signed_;}
    const float get_sip3d_Signed() const {return sip3D_signed_;}
    const float get_ip2d_Signed() const {return ip2D_signed_;}
    const float get_sip2d_Signed() const {return sip2D_signed_;}
    const float get_chi2reduced() const {return chi2reduced_;}
    const float get_nPixelHits() const {return nPixelHits_;}
    const float get_nHits() const {return nHits_;}
    const float get_jetAxisDistance() const {return jetAxisDistance_;}
    const float get_jetAxisDlength() const {return jetAxisDlength_;}
    const float get_trackProbability3D() const {return trackProbability3D_;}
    const float get_trackProbability2D() const {return trackProbability2D_;}   


private:

    float pt_;
    float eta_;
    float phi_;
    float mass_;
    float dz_;
    float dxy_;
    float ip3D_;
    float sip3D_;
    float ip2D_;
    float sip2D_;
    float ip3D_signed_;
    float sip3D_signed_;
    float ip2D_signed_;
    float sip2D_signed_;
    float chi2reduced_;
    float nPixelHits_;
    float nHits_;
    float jetAxisDistance_;
    float jetAxisDlength_;
    float trackProbability3D_;
    float trackProbability2D_;

    

};
}

#endif //RecoBTag_FeatureTools_SeedingTrackInfoBuilder_h 
