#include "AnalysisDataFormats/TrackInfo/interface/RecoTracktoTP.h"


// Constructors
RecoTracktoTP::RecoTracktoTP() 
{
    SetBeamSpot(math::XYZPoint(-9999.0, -9999.0, -9999.0)); 
    SetTrackingParticlePCA(GlobalPoint(-9999.0, -9999.0, -9999.0));
    SetTrackingParticleMomentumPCA(GlobalVector(-9999.0, -9999.0, -9999.0));
}


RecoTracktoTP::~RecoTracktoTP()
{
}


TrackingParticle RecoTracktoTP::TPMother(unsigned short i) const
{
    std::vector<TrackingParticle>  result;

    if( TP().parentVertex().isNonnull())
    {
        if(TP().parentVertex()->nSourceTracks() > 0)
        {
        for(TrackingParticleRefVector::iterator si = TP().parentVertex()->sourceTracks_begin();
            si != TP().parentVertex()->sourceTracks_end(); ++si)
            {
            for(TrackingParticleRefVector::iterator di = TP().parentVertex()->daughterTracks_begin();
                di != TP().parentVertex()->daughterTracks_end(); ++di)
                {
                    if(si != di)
                    {
                        result.push_back(**si);
                        break;
                    }
                }
                if(result.size()) break;
            }
        }
        else
        {
            return TrackingParticle();
        }
    }
    else
    {
        return TrackingParticle();
    }

    return i < result.size() ? result[i] : TrackingParticle();
}


int RecoTracktoTP::numTPMothers() const
{
    int count = 0;
    for(TrackingParticleRefVector::iterator si = TP().parentVertex()->sourceTracks_begin();
        si != TP().parentVertex()->sourceTracks_end(); ++si)
    {
        for(TrackingParticleRefVector::iterator di = TP().parentVertex()->daughterTracks_begin();
            di != TP().parentVertex()->daughterTracks_end(); ++di)
        {
            if(si != di) count++;
            break;
        }
        if(count>0) break;
    }
    return count;
}



