#include "DQMServices/DQMGUI/interface/DQMRenderPlugin.h"
#include "TROOT.h"
#include "TStyle.h"
#include "TH2.h"

class EcalPedestalsPCLRenderPlugin : public DQMRenderPlugin {
public:
  virtual bool applies(const VisDQMObject &dqmObject, const VisDQMImgInfo &) {
    return (dqmObject.name.substr(0, 33) == "AlCaReco/EcalPedestalsPCL/Summary");
  }

  virtual void preDraw(TCanvas *, const VisDQMObject &dqmObject, const VisDQMImgInfo &, VisDQMRenderInfo &renderInfo) {
    TH2 *obj(dynamic_cast<TH2 *>(dqmObject.object));

    // apply colz to all 2D histos
    if (obj) {
      obj->SetStats(false);
      renderInfo.drawOptions = "colz";
    }
  }
};

static EcalPedestalsPCLRenderPlugin instance;
