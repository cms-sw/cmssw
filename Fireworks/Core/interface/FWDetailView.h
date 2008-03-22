#ifndef Fireworks_Core_FWDetailView_h
#define Fireworks_Core_FWDetailView_h

class TEveElementList;
class TGTextView;
class FWModelId;

class FWDetailView {
public:
     virtual void 	build (TEveElementList **, const FWModelId &) = 0;
     virtual 		~FWDetailView () { }
     void		setTextView (TGTextView *v) { text_view = v; }

public:
     TGTextView	*text_view;
     Double_t 	rotation_center[3]; 
};

#endif
