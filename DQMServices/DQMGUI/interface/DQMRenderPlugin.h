#ifndef DQM_DQMRENDERPLUGIN_H
#define DQM_DQMRENDERPLUGIN_H

#include "Objects.h"
#include <vector>
// everybody just expects this to be there.
#include <iostream>

class TCanvas;
class DQMStore;

class DQMRenderPlugin {
  static std::vector<DQMRenderPlugin *> *s_list;

public:
  DQMRenderPlugin(void);
  virtual ~DQMRenderPlugin(void);

  static void master(std::vector<DQMRenderPlugin *> *list);

  virtual void initialise(int argc, char **argv);
  virtual bool applies(const VisDQMObject &obj, const VisDQMImgInfo &img) = 0;
  virtual void preDraw(TCanvas *c, const VisDQMObject &obj, const VisDQMImgInfo &img, VisDQMRenderInfo &opts);
  virtual void postDraw(TCanvas *c, const VisDQMObject &obj, const VisDQMImgInfo &img);
};

#endif  // DQM_DQMRENDERPLUGIN_H
