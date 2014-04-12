#include <DataFormats/TrackReco/interface/TrackFwd.h>
#include <DataFormats/Math/interface/LorentzVectorFwd.h>
#include <TVector3.h>
#include <vector>

class EventShape 
{
  public:
  
    static math::XYZTLorentzVectorF thrust(const reco::TrackCollection&);
    static float sphericity(const reco::TrackCollection&);
    static float aplanarity(const reco::TrackCollection&);
    static float planarity(const reco::TrackCollection&);

    EventShape(reco::TrackCollection&);

    math::XYZTLorentzVectorF thrust() const;
    float sphericity() const;
    float aplanarity() const;
    float planarity() const;
    
  private:

     std::vector<TVector3> p;
     std::vector<float> eigenvalues;
     
};
