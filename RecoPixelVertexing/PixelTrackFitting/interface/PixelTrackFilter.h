#ifndef PixelTrackFitting_PixelTrackFilter_H
#define PixelTrackFitting_PixelTrackFilter_H

namespace reco { class Track; };

class PixelTrackFilter {
public:
  virtual ~PixelTrackFilter() {}
  virtual bool operator()(const reco::Track*) const = 0;
};
#endif
