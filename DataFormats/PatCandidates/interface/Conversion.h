#ifndef DataFormats_PatCandidates_Conversion_h
#define DataFormats_PatCandidates_Conversion_h

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronCore.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronCoreFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/PatCandidates/interface/Lepton.h"

// Define typedefs for convenience
namespace pat {
  class Conversion;
  typedef std::vector<Conversion>              ConversionCollection;
  typedef edm::Ref<ConversionCollection>       ConversionRef;
  typedef edm::RefVector<ConversionCollection> ConversionRefVector;
}


//we use index to match with electron. However, can we do this with gsfTrack instead of index?
namespace pat {
  class Conversion {
  public:
    Conversion () {}
    Conversion ( int index );
    virtual ~Conversion() {}

    const double vtxProb() const {return vtxProb_;}
    void setVtxProb(double vtxProb);
    const double lxy() const {return lxy_;}
    void setLxy( double lxy );
    const int nHitsMax() const {return nHitsMax_;}
    void setNHitsMax( int nHitsMax );
    const int index() const {return index_; }     

  private:
    double vtxProb_;
    double lxy_;
    int nHitsMax_;

    //electron index matched with conversion
    int index_;

  };
}

#endif
