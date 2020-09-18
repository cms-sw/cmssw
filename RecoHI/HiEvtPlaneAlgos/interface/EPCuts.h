#ifndef __EPCuts__
#define __EPCuts__

namespace hi {

  enum class EP_ERA { ppReco, HIReco, Pixel, GenMC };

  struct TrackStructure {
    int centbin;
    double eta;
    double phi;
    double et;
    double pt;
    int charge;
    int pdgid;
    int hits;
    int algos;
    int collection;
    double dz;
    double dxy;
    double dzError;
    double dxyError;
    double ptError;
    bool highPurity;
    double dzSig;
    double dxySig;
    double normalizedChi2;
    double dzError_Pix;
    double chi2layer;
    int numberOfValidHits;
    int pixel;
  };

  class EPCuts {
  private:
    EP_ERA cutera_;
    double pterror_;
    double dzerror_;
    double dxyerror_;
    double chi2perlayer_;
    double dzerror_Pix_;
    double chi2Pix_;
    int numberOfValidHits_;

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
    ~EPCuts() {}

    bool isGoodHF(TrackStructure track) {
      if (track.pdgid != 1 && track.pdgid != 2)
        return false;
      if (fabs(track.eta) < 3 || fabs(track.eta) > 5)
        return false;
      return true;
    }

    bool isGoodCastor(TrackStructure track) { return true; }

    bool isGoodTrack(TrackStructure track) {
      if (cutera_ == EP_ERA::ppReco)
        return TrackQuality_ppReco(track);
      if (cutera_ == EP_ERA::HIReco)
        return TrackQuality_HIReco(track);
      if (cutera_ == EP_ERA::Pixel)
        return TrackQuality_Pixel(track);
      return false;
    }

    bool TrackQuality_ppReco(TrackStructure track) {
      if (track.charge == 0)
        return false;
      if (!track.highPurity)
        return false;
      if (track.ptError / track.pt > pterror_)
        return false;
      if (track.numberOfValidHits < numberOfValidHits_)
        return false;
      if (track.chi2layer > chi2perlayer_)
        return false;
      if (fabs(track.dxy / track.dxyError) > dxyerror_)
        return false;
      if (fabs(track.dz / track.dzError) > dzerror_)
        return false;
      return true;
    }

    bool TrackQuality_HIReco(TrackStructure track) {
      if (track.charge == 0)
        return false;
      if (!track.highPurity)
        return false;
      if (track.numberOfValidHits < numberOfValidHits_)
        return false;
      if (track.ptError / track.pt > pterror_)
        return false;
      if (fabs(track.dxy / track.dxyError) > dxyerror_)
        return false;
      if (fabs(track.dz / track.dzError) > dzerror_)
        return false;
      if (track.chi2layer > chi2perlayer_)
        return false;
      if (track.algos != 4 && track.algos != 5 && track.algos != 6 && track.algos != 7)
        return false;
      return true;
    }

    bool TrackQuality_Pixel(TrackStructure track) {
      if (track.charge == 0)
        return false;
      if (!track.highPurity)
        return false;
      bool bPix = false;
      int nHits = track.numberOfValidHits;
      if (track.ptError / track.pt > pterror_)
        return false;
      if (track.pt < 2.4 and (nHits == 3 or nHits == 4 or nHits == 5 or nHits == 6))
        bPix = true;
      if (not bPix) {
        if (nHits < numberOfValidHits_)
          return false;
        if (track.chi2layer > chi2perlayer_)
          return false;
        if (track.ptError / track.pt > pterror_)
          return false;
        int algo = track.algos;
        if (track.pt > 2.4 && algo != 4 && algo != 5 && algo != 6 && algo != 7)
          return false;
        if (fabs(track.dxy / track.dxyError) > dxyerror_)
          return false;
        if (fabs(track.dz / track.dzError) > dzerror_)
          return false;
      } else {
        if (track.chi2layer > chi2Pix_)
          return false;
        if (fabs(track.dz / track.dzError) > dzerror_Pix_)
          return false;
      }
      return true;
    }

    bool TrackQuality_GenMC(TrackStructure track) {
      if (track.charge == 0)
        return false;
      if (fabs(track.eta) > 2.4)
        return false;
      return true;
    }
  };
}  // namespace hi
#endif
