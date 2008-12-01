// -*- C++ -*-
#ifndef Fireworks_Core_FWDetailView_h
#define Fireworks_Core_FWDetailView_h

class TEveElementList;
class TGTextView;
class TGLViewer;
class FWModelId;

class FWDetailView {
public:
     virtual void 	build (TEveElementList **, const FWModelId &) = 0;
     virtual 		~FWDetailView () { }
     void		setTextView (TGTextView *v) { text_view = v; }
     void		setViewer (TGLViewer *v) { viewer = v; }

public:
     TGTextView	*text_view;
     TGLViewer	*viewer;
     Double_t 	rotation_center[3]; 
};

#endif
