#ifndef MEClickableCanvas_hh
#define MEClickableCanvas_hh

#include <TRootEmbeddedCanvas.h>

class MECanvasHolder;

class MEClickableCanvas : public TRootEmbeddedCanvas
{
public:

  MEClickableCanvas( const char *name, 
		     const TGWindow *p, 
		     UInt_t w, UInt_t h,
		     MECanvasHolder* gui );
  virtual ~MEClickableCanvas() {}
  
  Bool_t HandleContainerDoubleClick(Event_t *event);

private:
  
  MECanvasHolder* _gui;

  ClassDef( MEClickableCanvas, 0 ) //
};

#endif
