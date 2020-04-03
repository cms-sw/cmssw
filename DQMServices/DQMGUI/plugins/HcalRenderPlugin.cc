/*
 *	file:			HcalRenderPlugin.cc
 *	author:			Viktor Khristenko
 */

//	DQM includes
#include "DQM/DQMRenderPlugin.h"

//	User includes
#include "HcalObjectCustomizer.h"

//	ROOT includes
#include "TCanvas.h"
#include "TText.h"
#include "TColor.h"
//	std includes
#include <cstring>

// Render Plugin Class for Hcal Calib
class HcalRenderPlugin : public DQMRenderPlugin
{
	public:
		virtual void initialise(int, char**)
		{
			//	Initialize the Customizer
			_customizer.initialize_Type(hcaldqm::kHcal);
			_customizer.initialize_ColorSchemes();
			_customizer.initialize_Filters();
		}

		//	Check if we should draw this object
		virtual bool applies(VisDQMObject const& o, 
			VisDQMImgInfo const&)
		{
			//	return true only if that is Hcal/ or HcalCalib/
			return o.name.find("Hcal/")!=std::string::npos || 
				o.name.find("HcalCalib/")!=std::string::npos ||
				o.name.find("Hcal2/")!=std::string::npos;
		}

		//	if applies gives true - execute just before calling 
		virtual void preDraw(TCanvas *c, VisDQMObject const& object,
			VisDQMImgInfo const& iinfo, VisDQMRenderInfo &rinfo)
		{
			//	Check that the Object exists and inherits from TH1
			if (!object.object || !object.object->InheritsFrom(TH1::Class()))
				                    return;

			//	Apply Standard preDraw Customizations 
			_customizer.preDraw_Standard(c);

			//	Identify the Class of the Object and customize accordingly
			if (object.object->IsA()==TH1D::Class())
			{
				_customizer.pre_customize_1D(hcaldqm::kTH1D, c,
					object, iinfo, rinfo);
			}
			else if (object.object->IsA()==TH1F::Class())
			{
				_customizer.pre_customize_1D(hcaldqm::kTH1F, c,
					object, iinfo, rinfo);
			}
			else if (object.object->IsA()==TH2D::Class())
			{
				_customizer.pre_customize_2D(hcaldqm::kTH2D, c,
					object, iinfo, rinfo);
			}
			else if (object.object->IsA()==TH2F::Class())
			{
				_customizer.pre_customize_2D(hcaldqm::kTH2D, c,
					object, iinfo, rinfo);
			}
			else if (object.object->IsA()==TProfile::Class())
			{
				_customizer.pre_customize_1D(hcaldqm::kTProfile, c,
					object, iinfo, rinfo);
			}
			else if (object.object->IsA()==TProfile2D::Class())
			{
				_customizer.pre_customize_2D(hcaldqm::kTProfile2D, c,
					object, iinfo, rinfo);
			}
			else if (object.object->IsA()==TH3D::Class())
				return;
			else 
				return;

			return;
		}
		//	is called right after TObject::Draw()
		virtual void postDraw(TCanvas * c, VisDQMObject const& object,
			VisDQMImgInfo const& iinfo)
		{
			//	skip everything besides TH1 derived classes
			if (!object.object || !object.object->InheritsFrom(TH1::Class()))
				return;

			//	Apply Post Draw Standard Customizations
			_customizer.postDraw_Standard(c, object, iinfo);

			//	Identify the Class of the Object and customize accordingly
			if (object.object->IsA()==TH1D::Class())
			{
				_customizer.post_customize_1D(hcaldqm::kTH1D, c,
					object, iinfo);
			}
			else if (object.object->IsA()==TH1F::Class())
			{
				_customizer.post_customize_1D(hcaldqm::kTH1F, c,
					object, iinfo);
			}
			else if (object.object->IsA()==TH2D::Class())
			{
				_customizer.post_customize_2D(hcaldqm::kTH2D, c,
					object, iinfo);
			}
			else if (object.object->IsA()==TH2F::Class())
			{
				_customizer.post_customize_2D(hcaldqm::kTH2D, c,
					object, iinfo);
			}
			else if (object.object->IsA()==TProfile::Class())
			{
				_customizer.post_customize_1D(hcaldqm::kTProfile, c,
					object, iinfo);
			}
			else if (object.object->IsA()==TProfile2D::Class())
			{
				_customizer.post_customize_2D(hcaldqm::kTProfile2D, c,
					object, iinfo);
			}
			else if (object.object->IsA()==TH3D::Class())
				return;
			else 
				return;
			return;
		}
	protected:
		//	A wrapper around the Plugins...
		hcaldqm::HcalObjectCustomizer		_customizer;
};
static HcalRenderPlugin instance;
