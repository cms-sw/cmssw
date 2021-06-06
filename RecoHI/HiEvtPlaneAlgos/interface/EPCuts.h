#ifndef RecoHI_HiEvtPlaneAlgos_EPCuts_h
#define RecoHI_HiEvtPlaneAlgos_EPCuts_h

namespace hi {

  enum class EP_ERA { ppReco, HIReco, Pixel, GenMC };

  struct TrackStructure {
    int centbin;
    float eta;
    float phi;
    float et;
    float pt;
    int charge;
    int pdgid;
    int hits;
    int algos;
    int collection;
    float dz;
    float dxy;
    float dzError;
    float dxyError;
    float ptError;
    bool highPurity;
    float dzSig;
    float dxySig;
    float normalizedChi2;
    float dzError_Pix;
    float chi2layer;
    int numberOfValidHits;
    int pixel;
  };

  class EPCuts {
  public:
    explicit EPCuts(EP_ERA cutEra = EP_ERA::ppReco,
                    double pterror = 0.1,
                    double dzerror = 3.0,
                    double dxyerror = 3.0,
                    double chi2perlayer = 0.18,
                    double dzError_Pix = 10.0,
                    double chi2Pix = 40.,
                    int numberOfValidHits = 11) {
      cutera_ = cutEra;
      pterror_ = pterror;
      dzerror_ = dzerror;
      dxyerror_ = dxyerror;
      chi2perlayer_ = chi2perlayer;
      dzerror_Pix_ = dzError_Pix;
      chi2Pix_ = chi2Pix;
      numberOfValidHits_ = numberOfValidHits;
    }

    bool isGoodHF(const TrackStructure& track) const {
      if (track.pdgid != 1 && track.pdgid != 2)
        return false;
      if (std::abs(track.eta) < 3 || std::abs(track.eta) > 5)
        return false;
      return true;
    }

    bool isGoodCastor(const TrackStructure& track) const { return true; }

    bool isGoodTrack(const TrackStructure& track) const {
      if (cutera_ == EP_ERA::ppReco)
        return trackQuality_ppReco(track);
      if (cutera_ == EP_ERA::HIReco)
        return trackQuality_HIReco(track);
      if (cutera_ == EP_ERA::Pixel)
        return trackQuality_Pixel(track);
      return false;
    }

    bool trackQuality_ppReco(const TrackStructure& track) const {
      if (track.charge == 0)
        return false;
      if (!track.highPurity)
        return false;
      if (track.ptError > pterror_ * track.pt)
        return false;
      if (track.numberOfValidHits < numberOfValidHits_)
        return false;
      if (track.chi2layer > chi2perlayer_)
        return false;
      if (std::abs(track.dxy) > dxyerror_ * track.dxyError)
        return false;
      if (std::abs(track.dz) > dzerror_ * track.dzError)
        return false;
      return true;
    }

    bool trackQuality_HIReco(const TrackStructure& track) const {
      if (track.charge == 0)
        return false;
      if (!track.highPurity)
        return false;
      if (track.numberOfValidHits < numberOfValidHits_)
        return false;
      if (track.ptError > pterror_ * track.pt)
        return false;
      if (std::abs(track.dxy) > dxyerror_ * track.dxyError)
        return false;
      if (std::abs(track.dz) > dzerror_ * track.dzError)
        return false;
      if (track.chi2layer > chi2perlayer_)
        return false;
      //if (track.algos != 4 && track.algos != 5 && track.algos != 6 && track.algos != 7)
      if (track.algos != reco::TrackBase::initialStep && track.algos != reco::TrackBase::lowPtTripletStep &&
          track.algos != reco::TrackBase::pixelPairStep && track.algos != reco::TrackBase::detachedTripletStep)
        return false;
      return true;
    }

    bool trackQuality_Pixel(const TrackStructure& track) const {
      if (track.charge == 0)
        return false;
      if (!track.highPurity)
        return false;
      bool bPix = false;
      int nHits = track.numberOfValidHits;
      if (track.ptError > pterror_ * track.pt)
        return false;
      if (track.pt < 2.4 and (nHits <= 6))
        bPix = true;
      if (not bPix) {
        if (nHits < numberOfValidHits_)
          return false;
        if (track.chi2layer > chi2perlayer_)
          return false;
        if (track.ptError > pterror_ * track.pt)
          return false;
        int algo = track.algos;
        if (track.pt > 2.4 && algo != reco::TrackBase::initialStep && algo != reco::TrackBase::lowPtTripletStep &&
            algo != reco::TrackBase::pixelPairStep && algo != reco::TrackBase::detachedTripletStep)
          return false;
        if (std::abs(track.dxy) > dxyerror_ * track.dxyError)
          return false;
        if (std::abs(track.dz) > dzerror_ * track.dzError)
          return false;
      } else {
        if (track.chi2layer > chi2Pix_)
          return false;
        if (std::abs(track.dz) > dzerror_Pix_ * track.dzError)
          return false;
      }
      return true;
    }

    bool trackQuality_GenMC(const TrackStructure& track) const {
      if (track.charge == 0)
        return false;
      if (std::abs(track.eta) > 2.4)
        return false;
      return true;
    }

  private:
    EP_ERA cutera_;
    double pterror_;
    double dzerror_;
    double dxyerror_;
    double chi2perlayer_;
    double dzerror_Pix_;
    double chi2Pix_;
    int numberOfValidHits_;
  };
}  // namespace hi
#endif
