#include "MyHcalClasses.h"
#include "TObject.h"
#include "TCanvas.h"
#include "TH2.h"

class TCanvas;
class TH2;

class HcalElectronicsSelector {
public:
  class Callbacks {
  public:
    virtual ~Callbacks() { }
    virtual void plot(const MyElectronicsId& id) = 0;
   
  };

  HcalElectronicsSelector(Callbacks* cb,
			  int htrChan_lo=0, int htrChan_hi=24, int fpga_lo=-31, int fpga_hi=31,int crate=0);

  void fill(const MyElectronicsId& id, double value);
  void onEvent(int event, int x, int y, TObject *selected);//add crate
  void Update();
private:
  Callbacks* m_cb;
  TCanvas* m_canvas;
  TH2* m_hist;
  int m_crate;
};



