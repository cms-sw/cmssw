#ifndef Fireworks_Core_FWDetailView_h
#define Fireworks_Core_FWDetailView_h

class TEveElementList;
class FWModelId;

class FWDetailView {
public:
     virtual void build (TEveElementList **, const FWModelId &) = 0;
     virtual ~FWDetailView () { }

public:
     Double_t rotation_center[3]; // la di hack di da
};

#endif
