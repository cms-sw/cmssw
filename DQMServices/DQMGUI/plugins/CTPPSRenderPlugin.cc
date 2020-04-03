#include "DQM/DQMRenderPlugin.h"

#include "TH1F.h"
#include "TH2F.h"
#include "TCanvas.h"
#include "TStyle.h"

#include <cassert>

//*****************************************************
// 11.08.17 V.Popov: changed: TH2 palette to 1; OptStat(0)
//*****************************************************
class CTPPSRenderPlugin : public DQMRenderPlugin
{
	public:
		virtual bool applies(const VisDQMObject &o, const VisDQMImgInfo &) override
		{
			if ((o.name.find( "CTPPS/" ) != std::string::npos))
				return true;

			return false;
		}

		virtual void preDraw(TCanvas *c, const VisDQMObject &o, const VisDQMImgInfo &, VisDQMRenderInfo & ) override
		{
			c->cd();

			if (dynamic_cast<TH1F*>(o.object))
				preDrawTH1F(c, o);

			if (dynamic_cast<TH2F*>(o.object))
				preDrawTH2F(c, o);
			if (dynamic_cast<TH2D*>(o.object))
				preDrawTH2D(c, o);
		}

		virtual void postDraw( TCanvas *c, const VisDQMObject &o, const VisDQMImgInfo & ) override
		{
			c->cd();
			
			if (dynamic_cast<TH1F*>(o.object))
				postDrawTH1F(c, o);

			if (dynamic_cast<TH2F*>(o.object))
				postDrawTH2F(c, o);
		}

	private:
		void preDrawTH1F(TCanvas *, const VisDQMObject &o)
		{
			bool setColor = true;
			if (o.name.rfind(" U") == o.name.size() - 2)
				setColor = false;
			if (o.name.rfind(" V") == o.name.size() - 2)
				setColor = false;
			if (o.name.find("events per BX") != std::string::npos)
				setColor = false;

			TH1F* obj = dynamic_cast<TH1F*>(o.object);
			assert(obj);

			if (setColor)
				obj->SetLineColor(2);

			obj->SetLineWidth(2);
		}

		void preDrawTH2F(TCanvas *, const VisDQMObject &o)
		{
			TH2F* obj = dynamic_cast<TH2F*>(o.object);
			assert(obj);

			obj->SetOption("colz");
			gStyle->SetOptStat(0);
			obj->SetStats(kFALSE);
			gStyle->SetPalette(1);
		}

		void preDrawTH2D(TCanvas *, const VisDQMObject &o)
		{
			TH2D* obj = dynamic_cast<TH2D*>(o.object);
			assert(obj);

			obj->SetOption("colz");
			gStyle->SetOptStat(0);
			obj->SetStats(kFALSE);
			gStyle->SetPalette(1);
		}

		void postDrawTH1F(TCanvas *c, const VisDQMObject &)
		{
			c->SetGridx();
			c->SetGridy();
		}

		void postDrawTH2F(TCanvas *c, const VisDQMObject &)
		{
			c->SetGridx();
			c->SetGridy();
		}
};

//----------------------------------------------------------------------------------------------------

static CTPPSRenderPlugin instance;
