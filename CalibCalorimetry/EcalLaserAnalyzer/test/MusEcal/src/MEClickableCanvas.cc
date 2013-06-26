#include <iostream>
#include <assert.h>
using namespace std;

#include "MEClickableCanvas.hh"
#include "MusEcalGUI.hh"

ClassImp( MEClickableCanvas ) 

MEClickableCanvas::MEClickableCanvas( const char *name, 
				      const TGWindow *p, 
				      UInt_t w, UInt_t h,
				      MECanvasHolder* gui )
  : TRootEmbeddedCanvas( name, p, w, h ), _gui(gui) 
{
}

Bool_t 
MEClickableCanvas::HandleContainerDoubleClick(Event_t *event)
{
  bool k = TRootEmbeddedCanvas::HandleContainerDoubleClick(event);

  TPad* fPad1 = (TPad*) GetCanvas();
  assert( fPad1!=0 );
  //  fPad1->cd();

  int px = fPad1->GetEventX();
  int py = fPad1->GetEventY();

  _gui->setPxAndPy( px, py );

  return k;
}

