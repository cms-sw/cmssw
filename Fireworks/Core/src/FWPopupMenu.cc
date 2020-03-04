#include "TGMenu.h"
#include "KeySymbols.h"

class FWPopupMenu : public TGPopupMenu {
public:
  FWPopupMenu(const TGWindow* p = nullptr, UInt_t w = 10, UInt_t h = 10, UInt_t options = 0)
      : TGPopupMenu(p, w, h, options) {
    AddInput(kKeyPressMask);
  }

  // virtual void	PlaceMenu(Int_t x, Int_t y, Bool_t stick_mode, Bool_t grab_pointer)
  // {
  //    TGPopupMenu::PlaceMenu(x, y, stick_mode, grab_pointer);
  //    gVirtualX->GrabKey(fId, 0l, kAnyModifier, kTRUE);
  // }

  void PoppedUp() override {
    TGPopupMenu::PoppedUp();
    gVirtualX->SetInputFocus(fId);
    gVirtualX->GrabKey(fId, 0l, kAnyModifier, kTRUE);
  }

  void PoppedDown() override {
    gVirtualX->GrabKey(fId, 0l, kAnyModifier, kFALSE);
    TGPopupMenu::PoppedDown();
  }

  Bool_t HandleKey(Event_t* event) override {
    if (event->fType != kGKeyPress)
      return kTRUE;

    UInt_t keysym;
    char tmp[2];
    gVirtualX->LookupString(event, tmp, sizeof(tmp), keysym);

    TGMenuEntry* ce = fCurrent;

    switch (keysym) {
      case kKey_Up: {
        if (ce)
          ce = (TGMenuEntry*)GetListOfEntries()->Before(ce);
        while (ce && ((ce->GetType() == kMenuSeparator) || (ce->GetType() == kMenuLabel) ||
                      !(ce->GetStatus() & kMenuEnableMask))) {
          ce = (TGMenuEntry*)GetListOfEntries()->Before(ce);
        }
        if (!ce)
          ce = (TGMenuEntry*)GetListOfEntries()->Last();
        Activate(ce);
        break;
      }
      case kKey_Down: {
        if (ce)
          ce = (TGMenuEntry*)GetListOfEntries()->After(ce);
        while (ce && ((ce->GetType() == kMenuSeparator) || (ce->GetType() == kMenuLabel) ||
                      !(ce->GetStatus() & kMenuEnableMask))) {
          ce = (TGMenuEntry*)GetListOfEntries()->After(ce);
        }
        if (!ce)
          ce = (TGMenuEntry*)GetListOfEntries()->First();
        Activate(ce);
        break;
      }
      case kKey_Enter:
      case kKey_Return: {
        Event_t ev;
        ev.fType = kButtonRelease;
        ev.fWindow = fId;
        return HandleButton(&ev);
      }
      case kKey_Escape: {
        fCurrent = nullptr;
        void* dummy = nullptr;
        return EndMenu(dummy);
      }
      default: {
        break;
      }
    }

    return kTRUE;
  }
};
