#include "MyHcalClasses.h"
#include "TObject.h"
#include "TCanvas.h"
#include "TH2.h"

class TCanvas;
class TH2;

class HcalVisualSelector {
public:
  class Callbacks {
  public:
    virtual ~Callbacks() { }
    virtual void plot(const MyHcalDetId& id) = 0;
    virtual MyHcalSubdetector getSubdet(int ieta, int depth) = 0;
  };

  HcalVisualSelector(Callbacks* cb,
		     int ieta_lo=-41, int ieta_hi=41, int iphi_lo=1, int iphi_hi=72);

  void fill(const MyHcalDetId& id, double value);
  void onEvent(int event, int x, int y, TObject *selected);
  void Update();
private:
  Callbacks* m_cb;
  TCanvas* m_canvas;
  TH2* m_hist[4];  
};


