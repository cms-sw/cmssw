/*!
  \file DQMMessageLoggerRenderPlugin
  \brief Display Plugin for Quality Histograms
  \author E. Nesvold
  \version $Revision: 1.5 $
  \date $Date: 2011/09/09 11:53:42 $
*/

#include "DQM/DQMRenderPlugin.h"
#include "utils.h"
#include "TCanvas.h"
#include "TH1F.h"
#include "TH1.h"
#include "TStyle.h"
#include "TPad.h"
#include <cassert>

class DQMMessageLoggerRenderPlugin : public DQMRenderPlugin{

public:
  virtual bool applies( const VisDQMObject &o, const VisDQMImgInfo &)
    {

      if(o.name.find("MessageLogger") != std::string::npos){
        if(o.name.find ("Errors") != std::string::npos){
          if(o.name.find("categoriesErrorsFound")!=std::string::npos){
            return true;
          }else if(o.name.find("categories_errors")!=std::string::npos){
            return true;
          }else if(o.name.find("modulesErrorsFound")!=std::string::npos){
            return true;
          }else if(o.name.find("modules_errors")!=std::string::npos){
            return true;
          }else if(o.name.find("total_errors")!=std::string::npos){
	    return true;
	  }
        }
        if (o.name.find("Warnings") != std::string::npos){
          if(o.name.find("modulesWarningsFound") != std::string::npos){
            return true;
          }else if(o.name.find("modules_warnings")!=std::string::npos){
            return true;
          }else if(o.name.find("categoriesWarningsFound")!=std::string::npos){
            return true;
          }else if(o.name.find("categories_warnings")!=std::string::npos){
            return true;
          }else if(o.name.find("total warnings")!=std::string::npos){
	    return true;
	  }
        }

      }
      return false;
    }

  virtual void preDraw(TCanvas *c, const VisDQMObject &o, const VisDQMImgInfo &, VisDQMRenderInfo &)
    {

      if (dynamic_cast<TH1F*>(o.object)){
        preDrawTH1(c, o);
      }

    }

  virtual void postDraw(TCanvas *, const VisDQMObject &, const VisDQMImgInfo &){};

private:
  void preDrawTH1(TCanvas *c, const VisDQMObject &o)
    {

      TH1F *obj = dynamic_cast<TH1F*>(o.object);

      assert(obj);

      c->cd();

      //obj->SetTitle(kFALSE);

      Double_t bm = 0.4;

      if(o.name.find("categoriesErrorsFound")!=std::string::npos){
	if(obj->ComputeIntegral() > 0.5){
	  gPad->SetBottomMargin(bm);
	}
	gStyle->SetOptStat("e");
      }else if(o.name.find("categoriesWarningsFound")!=std::string::npos){

	if(obj->ComputeIntegral() > 0.5){
	  gPad->SetBottomMargin(bm);
	}
	gStyle->SetOptStat("e");
      }else if(o.name.find("modulesErrorsFound")!=std::string::npos){
	if(obj->ComputeIntegral() > 0.5){
	  gPad->SetBottomMargin(bm);
	}
	gStyle->SetOptStat("e");
      }else if(o.name.find("modulesWarningsFound")!=std::string::npos){
	if(obj->ComputeIntegral() > 0.5){
	  gPad->SetBottomMargin(bm);
	}
	gStyle->SetOptStat("e" );
      }else if(o.name.find("categories_errors")!=std::string::npos){
	if(obj->ComputeIntegral() > 0.5){
	  gPad->SetBottomMargin(bm);
	}

	gStyle->SetOptStat("e" );
      }else if(o.name.find("categories_warnings")!=std::string::npos){
	if(obj->ComputeIntegral() > 0.5){
	  gPad->SetBottomMargin(bm);
	}

	gStyle->SetOptStat("e" );
      }else if(o.name.find("modules_errors")!=std::string::npos){
	if(obj->ComputeIntegral() > 0.5){
	  gPad->SetBottomMargin(bm);
	}

	gStyle->SetOptStat("e" );
      }else if(o.name.find("modules_warnings")!=std::string::npos){
	if(obj->ComputeIntegral() > 0.5){
	  gPad->SetBottomMargin(bm);
	}
	gStyle->SetOptStat("e" );
      }else if(o.name.find("total warnings")!=std::string::npos){
	gStyle->SetOptStat("emr");
      }else if(o.name.find("total_errors")!=std::string::npos){
	gStyle->SetOptStat("emr");
      }

    }
};

static DQMMessageLoggerRenderPlugin instance;
