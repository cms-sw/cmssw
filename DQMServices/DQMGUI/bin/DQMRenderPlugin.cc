#include "DQMRenderPlugin.h"
#include <algorithm>

std::vector<DQMRenderPlugin *> *DQMRenderPlugin::s_list = 0;

DQMRenderPlugin::DQMRenderPlugin(void)
{
  s_list->push_back(this);
}

DQMRenderPlugin::~DQMRenderPlugin (void)
{
  s_list->erase(std::find(s_list->begin(), s_list->end(), this));
}

void
DQMRenderPlugin::master(std::vector<DQMRenderPlugin *> *list)
{
  s_list = list;
}

void
DQMRenderPlugin::initialise (int, char **)
{}

void
DQMRenderPlugin::preDraw (TCanvas *,
			  const VisDQMObject &,
			  const VisDQMImgInfo &,
			  VisDQMRenderInfo &)
{}

void
DQMRenderPlugin::postDraw (TCanvas *,
			   const VisDQMObject &,
			   const VisDQMImgInfo &)
{}
