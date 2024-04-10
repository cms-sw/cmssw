#ifndef _CommonTools_TrackerMap_TrackerMap_h_
#define _CommonTools_TrackerMap_TrackerMap_h_
#include <utility>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <map>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "TColor.h"
#include <cassert>
#include <vector>
#include "TPolyLine.h"

class TH1F;
class TLegend;
/*
#define  NUMFEDCH_INCOLUMN 12
#define  NUMFEDCH_INROW 8
#define  NUMFED_INCOLUMN 6
#define  NUMFED_INROW 4
#define  NUMFEDCRATE_INCOLUMN 3
#define  XFEDCSIZE  340
#define  YFEDCSIZE  400
*/

/*
#define  NUMFEDCH_INCOLUMN 12
#define  NUMFEDCH_INROW 8
#define  NUMFED_INCOLUMN 1
#define  NUMFED_INROW 21
//#define  NUMFEDCRATE_INCOLUMN 11
//#define  NUMFEDCRATE_INROW 3
#define  NUMFEDCRATE_INCOLUMN 16
#define  NUMFEDCRATE_INROW 2
//#define  XFEDCSIZE  160  // 14 boxes
//#define  YFEDCSIZE  2100 // 9*21 boxes
#define  XFEDCSIZE  140  // 14 boxes
#define  YFEDCSIZE  1890 // 9*21 boxes
*/

#define NUMPSUCH_INROW 18
#define NUMPSUCRATE_INCOLUMN 5
#define NUMPSURACK_INCOLUMN 6
#define NUMPSURACK_INROW 5
#define XPSURSIZE 150  // (5)*1.5 boxes
#define YPSURSIZE 360  // 18 boxes
#define XPSUOFFSET 50
#define YPSUOFFSET 100

#define NUMFEDCH_INCOLUMN 12
#define NUMFEDCH_INROW 8
#define NUMFED_INCOLUMN 21
#define NUMFED_INROW 1
//#define  NUMFEDCRATE_INCOLUMN 11
//#define  NUMFEDCRATE_INROW 3
#define NUMFEDCRATE_INCOLUMN 1
#define NUMFEDCRATE_INROW 31
//#define  XFEDCSIZE  160  // 14 boxes
//#define  YFEDCSIZE  2100 // 9*21 boxes
#define XFEDCSIZE 2940  // 14*21 boxes
#define YFEDCSIZE 90    // 9 boxes
#define XFEDOFFSET 150
#define YFEDOFFSET 100

class TmModule;
class TmApvPair;
class EventSetup;
class TmCcu;
class TmPsu;
class TrackerTopology;

class TrackerMap {
public:
  //TrackerMap(){TrackerMap(" ");};   //!< default constructor
  TrackerMap(std::string s = " ", int xsize1 = 340, int ysize1 = 200);
  TrackerMap(const edm::ParameterSet& iConfig);
  TrackerMap(const edm::ParameterSet& iConfig, const SiStripFedCabling* tkFed, const TrackerTopology* const topology);
  ~TrackerMap();  //!< default destructor

  void build();
  void init();
  void drawModule(TmModule* mod, int key, int layer, bool total, std::ofstream* file);
  void print(bool print_total = true, float minval = 0., float maxval = 0., std::string s = "svgmap");
  void printall(bool print_total = true,
                float minval = 0.,
                float maxval = 0.,
                std::string s = "svgmap",
                int width = 6000,
                int height = 3200);
  void printonline();
  void printlayers(bool print_total = true, float minval = 0., float maxval = 0., std::string s = "layer");
  void save(bool print_total = true,
            float minval = 0.,
            float maxval = 0.,
            std::string s = "svgmap.svg",
            int width = 1500,
            int height = 800);
  void save_as_fedtrackermap(bool print_total = true,
                             float minval = 0.,
                             float maxval = 0.,
                             std::string s = "fed_svgmap.svg",
                             int width = YFEDOFFSET + (YFEDCSIZE + YFEDOFFSET) * NUMFEDCRATE_INROW + 300,
                             int height = XFEDOFFSET + (XFEDCSIZE + XFEDOFFSET) * NUMFEDCRATE_INCOLUMN + 300);
  void save_as_fectrackermap(bool print_total = true,
                             float minval = 0.,
                             float maxval = 0.,
                             std::string s = "fec_svgmap.svg",
                             int width = 1500,
                             int height = 800);
  void save_as_psutrackermap(bool print_total = true,
                             float minval = 0.,
                             float maxval = 0.,
                             std::string s = "psu_svgmap.svg",
                             int width = YPSUOFFSET + (YPSURSIZE + YPSUOFFSET) * NUMPSURACK_INROW + 300,
                             int height = XPSUOFFSET + (XPSURSIZE + XPSUOFFSET) * NUMPSURACK_INCOLUMN + 300);
  void save_as_HVtrackermap(bool print_total = true,
                            float minval = 0.,
                            float maxval = 0.,
                            std::string s = "psu_svgmap.svg",
                            int width = 1500,
                            int height = 800);
  void drawApvPair(
      int crate, int numfed_incrate, bool total, TmApvPair* apvPair, std::ofstream* file, bool useApvPairValue);
  void drawCcu(int crate, int numfed_incrate, bool total, TmCcu* ccu, std::ofstream* file, bool useCcuValue);
  void drawPsu(int rack, int numcrate_inrack, bool print_total, TmPsu* psu, std::ofstream* svgfile, bool usePsuValue);
  void drawHV2(int rack, int numcrate_inrack, bool print_total, TmPsu* psu, std::ofstream* svgfile, bool usePsuValue);
  void drawHV3(int rack, int numcrate_inrack, bool print_total, TmPsu* psu, std::ofstream* svgfile, bool usePsuValue);
  void fill_current_val(int idmod, float current_val);
  void fill(int layer, int ring, int nmod, float x);
  void fill(int idmod, float qty);
  void fillc(int idmod, int RGBcode) { fillc(idmod, (RGBcode >> 16) & 0xFF, (RGBcode >> 8) & 0xFF, RGBcode & 0xFF); }
  void fillc(int idmod, int red, int green, int blue);
  void fillc(int layer, int ring, int nmod, int red, int green, int blue);
  void fillc_all_blank();
  void fill_all_blank();
  void fill_current_val_fed_channel(int fedId, int fedCh, float current_val);
  void fill_fed_channel(int fedId, int fedCh, float qty);
  void fill_fed_channel(int modId, float qty);
  void fillc_fed_channel(int fedId, int fedCh, int red, int green, int blue);
  void fillc_fec_channel(int crate, int slot, int ring, int addr, int red, int green, int blue);
  void fill_fec_channel(int crate, int slot, int ring, int addr, float qty);
  void fill_lv_channel(int rack, int crate, int board, float qty);
  void fillc_lv_channel(int rack, int crate, int board, int red, int green, int blue);
  void fill_hv_channel2(int rack, int crate, int board, float qty);
  void fillc_hv_channel2(int rack, int crate, int board, int red, int green, int blue);
  void fill_hv_channel3(int rack, int crate, int board, float qty);
  void fillc_hv_channel3(int rack, int crate, int board, int red, int green, int blue);
  int module(int fedId, int fedCh);
  void setText(int idmod, std::string s);
  void setText(int layer, int ring, int nmod, std::string s);
  void setPalette(int numpalette) { palette = numpalette; }
  void drawPalette(std::ofstream* file, int xoffset = 3660, int yoffset = 1540);
  void showPalette(bool printflag1) { printflag = printflag1; };
  void setTitle(std::string s) { title = s; };
  void setRange(float min, float max);
  std::pair<float, float> getAutomaticRange();
  void addPixel(bool addPixelfl) { addPixelFlag = addPixelfl; };
  void onlyPixel(bool onlyPixelfl) { onlyPixelFlag = onlyPixelfl; };
  void reset();
  void load(std::string s = "tmap.svg");
  int getxsize() { return xsize; };
  int getysize() { return ysize; };
  int getcolor(float value, int palette);
  std::ifstream* findfile(std::string filename);
  int getNumMod() { return number_modules; };
  std::vector<TColor*> vc;
  typedef std::map<const int, TmModule*> SmoduleMap;
  SmoduleMap smoduleMap;
  typedef std::map<const int, TmModule*> ImoduleMap;
  ImoduleMap imoduleMap;
  typedef std::map<const int, TmApvPair*> SvgApvPair;
  SvgApvPair apvMap;
  typedef std::multimap<const int, TmApvPair*> ModApvPair;
  ModApvPair apvModuleMap;
  typedef std::map<const int, int> SvgFed;
  SvgFed fedMap;
  SvgFed slotMap;
  typedef std::map<const int, TmCcu*> MapCcu;
  MapCcu ccuMap;
  typedef std::multimap<TmCcu*, TmModule*> FecModule;
  FecModule fecModuleMap;
  typedef std::map<const int, TmPsu*> MapPsu;
  MapPsu psuMap;
  typedef std::multimap<TmPsu*, TmModule*> PsuModule;
  PsuModule psuModuleMap;
  int palette;
  bool printflag;
  bool saveWebInterface;
  bool saveGeoTrackerMap;
  bool enableFedProcessing;
  bool enableFecProcessing;
  bool enableLVProcessing;
  bool enableHVProcessing;
  bool tkMapLog;
  std::string title;
  std::string jsfilename, infilename;
  std::string jsPath;
  bool psetAvailable;

  double phival(double x, double y) {
    double phi;
    if (fabs(x) >= 0.000001 && fabs(y) >= 0.000001) {
      phi = atan(y / x);
      if (y < 0. && x > 0)
        phi += 2. * M_PI;
      else if (x < 0.)
        phi += M_PI;
    } else if (fabs(y) < 0.000001)
      phi = x > 0 ? 0 : M_PI;
    else  // if (fabs(x) < 0.000001)
      phi = y > 0 ? M_PI / 2. : 3. * M_PI / 2.;
    return phi;
  }

  int find_layer(int ix, int iy) {
    int layer = 0;
    if (iy <= xsize) {  //endcap+z
      const int add = 15;
      layer = ix / ysize;
      layer = layer + add + 1;
    } else if (iy < 3 * xsize) {  //barrel
      const int add = 30;
      if (ix < 2 * ysize) {
        layer = 1;
      } else {
        layer = ix / (2 * ysize);
        if (iy < 2 * xsize)
          layer = layer * 2 + 1;
        else
          layer = layer * 2;
      }
      layer = layer + add;
    } else if (iy >= 3 * xsize) {  //endcap-z
      layer = ix / ysize;
      layer = 15 - layer;
    }
    return layer;
  }

  int getlayerCount(int subdet, int partdet) {
    int ncomponent = 0;
    if (subdet == 1) {                     //1=pixel
      if (partdet == 1 || partdet == 3) {  //1-3=encap
        ncomponent = 3;
      } else {
        ncomponent = 3;
      }  //barrel
    }
    if (subdet == 2) {                     //2=inner silicon
      if (partdet == 1 || partdet == 3) {  //1-3=encap
        ncomponent = 3;
      } else {
        ncomponent = 4;
      }  //barrel
    }
    if (subdet == 3) {                     //3=outer silicon
      if (partdet == 1 || partdet == 3) {  //1-3=encap
        ncomponent = 9;
      } else {
        ncomponent = 6;
      }  //barrel
    }
    return (ncomponent);
  }

  double xdpixel(double x) {
    double res;
    if (saveAsSingleLayer)
      res = ((x - xmin) / (xmax - xmin) * xsize);
    else
      res = ((x - xmin) / (xmax - xmin) * xsize) + ix;
    return res;
  }
  double ydpixel(double y) {
    double res = 0;
    double y1;
    y1 = (y - ymin) / (ymax - ymin);
    if (nlay > 30) {
      if (nlay < 34)
        res = 2 * ysize - (y1 * 2 * ysize);
      if (nlay == 34)
        res = 2.4 * ysize - (y1 * 2.4 * ysize);
      if (nlay > 34)
        res = 2.5 * ysize - (y1 * 2.5 * ysize);
    } else
      res = xsize - (y1 * xsize);
    if (!saveAsSingleLayer)
      res = res + iy;
    return res;
  }
  double xdpixelc(double x) {
    double res;
    if (saveAsSingleLayer)
      res = ((x - xmin) / (xmax - xmin) * XFEDCSIZE);
    else
      res = ((x - xmin) / (xmax - xmin) * XFEDCSIZE) + ix;
    return res;
  }
  double ydpixelc(double y) {
    double res;
    double y1;
    y1 = (y - ymin) / (ymax - ymin);
    if (saveAsSingleLayer)
      res = YFEDCSIZE - (y1 * YFEDCSIZE);
    else
      res = YFEDCSIZE - (y1 * YFEDCSIZE) + iy;
    return res;
  }
  double xdpixelfec(double x) {
    double res;
    if (saveAsSingleLayer)
      res = ((x - xmin) / (xmax - xmin) * xsize);
    else
      res = ((x - xmin) / (xmax - xmin) * xsize) + ix;
    return res;
  }
  double ydpixelfec(double y) {
    double res;
    double y1;
    y1 = (y - ymin) / (ymax - ymin);
    if (saveAsSingleLayer)
      res = 2 * ysize - (y1 * 2 * ysize);
    else
      res = 2 * ysize - (y1 * 2 * ysize) + iy;
    return res;
  }
  double xdpixelpsu(double x) {
    double res;
    if (saveAsSingleLayer)
      res = ((x - xmin) / (xmax - xmin) * XPSURSIZE);
    else
      res = ((x - xmin) / (xmax - xmin) * XPSURSIZE) + ix;
    return res;
  }
  double ydpixelpsu(double y) {
    double res;
    double y1;
    y1 = (y - ymin) / (ymax - ymin);
    if (saveAsSingleLayer)
      res = YPSURSIZE - (y1 * YPSURSIZE);
    else
      res = YPSURSIZE - (y1 * YPSURSIZE) + iy;
    return res;
  }

  void defcwindow(int num_crate) {
    //    ncrate = num_crate;
    int xoffset = XFEDOFFSET;
    int yoffset = YFEDOFFSET;
    xmin = 0.;
    xmax = (NUMFEDCH_INCOLUMN + 2) * NUMFED_INCOLUMN;
    ymin = 0.;
    ymax = (NUMFEDCH_INROW + 1) * NUMFED_INROW;

    ix = xoffset + ((NUMFEDCRATE_INCOLUMN - 1) - ((num_crate - 1) % NUMFEDCRATE_INCOLUMN)) * (XFEDCSIZE + XFEDOFFSET);
    iy = yoffset + ((num_crate - 1) / NUMFEDCRATE_INCOLUMN) * (YFEDCSIZE + YFEDOFFSET);
  }
  void deffecwindow(int num_crate) {
    //    ncrate = num_crate;
    int xoffset = xsize / 3;
    int yoffset = 2 * ysize;
    xmin = -1.;
    xmax = 37.;
    ymin = -10.;
    ymax = 40.;
    if (num_crate == 1 || num_crate == 3)
      ix = xoffset + xsize * 2;
    if (num_crate == 2 || num_crate == 4)
      ix = xoffset;
    iy = yoffset + ((num_crate - 1) / 2) * ysize * 4;
  }
  void defpsuwindow(int num_rack) {
    //    nrack = num_rack;
    int xoffset = XPSUOFFSET;
    int yoffset = YPSUOFFSET;
    xmin = 0;
    xmax = (NUMPSUCRATE_INCOLUMN)*1.5;
    ymin = 0;
    ymax = NUMPSUCH_INROW;

    ix = xoffset + ((NUMPSURACK_INCOLUMN - 1) - ((num_rack - 1) % NUMPSURACK_INCOLUMN)) * (XPSURSIZE + XPSUOFFSET);
    iy = yoffset + ((num_rack - 1) / NUMPSURACK_INCOLUMN) * (YPSURSIZE + YPSUOFFSET);
  }

  void defwindow(int num_lay) {
    // nlay = num_lay;
    if (posrel) {  // separated modules
      xmin = -2.;
      ymin = -2.;
      xmax = 2.;
      ymax = 2.;
      if (num_lay > 12 && num_lay < 19) {
        xmin = -.40;
        xmax = .40;
        ymin = -.40;
        ymax = .40;
      }
      if (num_lay > 30) {
        xmin = -0.1;
        xmax = 3.;
        ymin = -0.1;
        ymax = 8.5;
        if (num_lay < 34) {
          xmin = -0.3;
          xmax = 1.0;
        }
        if (num_lay > 33 && num_lay < 38) {
          xmax = 2.0;
        }
        if (num_lay > 37) {
          ymax = 8.;
        }  //inner
      }
    } else {  //overlayed modules
      xmin = -1.3;
      ymin = -1.3;
      xmax = 1.3;
      ymax = 1.3;
      if (num_lay > 12 && num_lay < 19) {
        xmin = -.20;
        xmax = .20;
        ymin = -.20;
        ymax = .20;
      }
      if (num_lay > 30) {
        xmin = -1.5;
        xmax = 1.5;
        ymin = -1.;
        ymax = 28.;
        if (num_lay < 34) {
          xmin = -0.5;
          xmax = 0.5;
        }
        if (num_lay > 33 && num_lay < 38) {
          xmin = -1.;
          xmax = 1.;
        }
      }
    }
    if (num_lay < 16) {
      ix = 0;
      if (num_lay == 15 || num_lay == 14)
        iy = (15 - num_lay) * 2 * ysize;
      else {
        if (num_lay > 9 && num_lay < 13)
          iy = 4 * ysize - (int)(ysize / 2.) + (12 - num_lay) * (int)(ysize / 1.50);
        else
          iy = 6 * ysize + (9 - num_lay) * (int)(ysize * 1.3);
      }
    }
    if (num_lay > 15 && num_lay < 31) {
      ix = 3 * xsize;
      if (num_lay == 16 || num_lay == 17)
        iy = (num_lay - 16) * 2 * ysize;
      else {
        if (num_lay > 18 && num_lay < 22)
          iy = 4 * ysize - (int)(ysize / 2.) + (num_lay - 19) * (int)(ysize / 1.50);
        else
          iy = 6 * ysize + (num_lay - 22) * (int)(ysize * 1.3);
      }
    }
    if (num_lay > 30) {
      if (num_lay == 31) {
        ix = (int)(1.5 * xsize);
        iy = 0;
      }
      if (num_lay == 32) {
        int il = (num_lay - 30) / 2;
        ix = xsize;
        iy = il * 2 * ysize;
      }
      if (num_lay == 33) {
        int il = (num_lay - 30) / 2;
        ix = 2 * xsize;
        iy = il * 2 * ysize;
      }
      if (num_lay == 34) {
        int il = (num_lay - 30) / 2;
        ix = xsize;
        iy = il * (int)(2.57 * ysize);
      }
      if (num_lay > 34 && num_lay % 2 == 0) {
        int il = (num_lay - 30) / 2;
        ix = xsize;
        iy = il * (int)(2.5 * ysize);
      }
      if (num_lay > 34 && num_lay % 2 != 0) {
        int il = (num_lay - 30) / 2;
        ix = 2 * xsize;
        iy = il * (int)(2.5 * ysize);
      }
    }
  }

  int getringCount(int subdet, int partdet, int layer) {
    int ncomponent = 0;
    if (subdet == 1) {                   //1=pixel
      if (partdet == 1 || partdet == 3)  //end-cap
        ncomponent = 7;
      else  //barrel
        ncomponent = 8;
    } else if (subdet == 2) {            //inner-silicon
      if (partdet == 1 || partdet == 3)  //end-cap
        ncomponent = 3;
      else  //barrel
        ncomponent = 12;
    } else if (subdet == 3) {  //outer-silicon
      if (partdet == 1) {      //end-cap-z
        if (layer == 1)
          ncomponent = 4;
        else if (layer == 2 || layer == 3)
          ncomponent = 5;
        else if (layer == 4 || layer == 5 || layer == 6)
          ncomponent = 6;
        else if (layer == 7 || layer == 8 || layer == 9)
          ncomponent = 7;
      } else if (partdet == 3) {  //endcap+z
        if (layer == 9)
          ncomponent = 4;
        else if (layer == 8 || layer == 7)
          ncomponent = 5;
        else if (layer == 6 || layer == 5 || layer == 4)
          ncomponent = 6;
        else if (layer == 3 || layer == 2 || layer == 1)
          ncomponent = 7;
      } else if (partdet == 2) {  //barrel
        ncomponent = 12;
      }
    }
    return (ncomponent);
  }

  int getmoduleCount(int subdet, int partdet, int layer, int ring) {
    int ncomponent = 0;
    int spicchif[] = {24, 24, 40, 56, 40, 56, 80};
    int spicchib[] = {20, 32, 44, 30, 38, 46, 56, 42, 48, 54, 60, 66, 74};
    int numero_layer = 0;

    if (partdet == 2) {  //barrel
      numero_layer = layer - 1;
      if (subdet == 2) {  //inner
        numero_layer = numero_layer + 3;
      }
      if (subdet == 3) {  //outer
        numero_layer = numero_layer + 7;
      }
      ncomponent = spicchib[numero_layer];
    }
    if (partdet != 2) {  //endcap
      if (subdet == 1)
        ncomponent = 24;  //pixel
      else
        ncomponent = spicchif[ring - 1];
    }
    return (ncomponent);
  }

  static int layerno(int subdet, int leftright, int layer) {
    if (subdet == 6 && leftright == 1)
      return (10 - layer);
    if (subdet == 6 && leftright == 2)
      return (layer + 21);
    if (subdet == 4 && leftright == 1)
      return (4 - layer + 9);
    if (subdet == 4 && leftright == 2)
      return (layer + 18);
    if (subdet == 2 && leftright == 1)
      return (4 - layer + 12);
    if (subdet == 2 && leftright == 2)
      return (layer + 15);
    if (subdet == 1)
      return (layer + 30);
    if (subdet == 3)
      return (layer + 33);
    if (subdet == 5)
      return (layer + 37);
    assert(false);
  }

  static bool isRingStereo(int key) {
    int layer = key / 100000;
    int ring = key - layer * 100000;
    ring = ring / 1000;
    if (layer == 34 || layer == 35 || layer == 38 || layer == 39)
      return true;
    if (layer < 13 || (layer > 18 && layer < 31))
      if (ring == 1 || ring == 2 || ring == 5)
        return true;
    return false;
  }

  int nlayer(int det, int part, int lay) {
    if (det == 3 && part == 1)
      return lay;
    if (det == 2 && part == 1)
      return lay + 9;
    if (det == 1 && part == 1)
      return lay + 12;
    if (det == 1 && part == 3)
      return lay + 15;
    if (det == 2 && part == 3)
      return lay + 18;
    if (det == 3 && part == 3)
      return lay + 21;
    if (det == 1 && part == 2)
      return lay + 30;
    if (det == 2 && part == 2)
      return lay + 33;
    if (det == 3 && part == 2)
      return lay + 37;
    std::cout
        << "Error in TrackerMap: this can never happen as det and part are comprised between 1 and 3, while they are "
        << det << " and " << part << " respectively" << std::endl;
    return lay;
  }

  std::string layername(int layer) {
    std::string s = " ";
    std::ostringstream ons;

    if (layer < 10)
      ons << "TEC -z Layer " << layer;
    if (layer < 13 && layer > 9)
      ons << "TID -z Layer " << layer - 9;
    if (layer < 16 && layer > 12)
      ons << "FPIX -z Layer " << layer - 12;
    if (layer < 19 && layer > 15)
      ons << "FPIX +z Layer " << layer - 15;
    if (layer < 22 && layer > 18)
      ons << "TID +z Layer " << layer - 18;
    if (layer < 31 && layer > 21)
      ons << "TEC +z Layer " << layer - 21;
    if (layer < 34 && layer > 30)
      ons << "TPB Layer " << layer - 30;
    if (layer < 38 && layer > 33)
      ons << "TIB Layer " << layer - 33;
    if (layer > 37)
      ons << "TOB Layer " << layer - 37;
    s = ons.str();
    return s;
  }

  int ntotRing[43];
  int firstRing[43];

protected:
  int nlay;
  //  int ncrate;
  //  int nrack;
  int ncrates;
  int firstcrate;
  int nfeccrates;
  int npsuracks;
  double xmin, xmax, ymin, ymax;
  int xsize, ysize, ix, iy;
  bool posrel;
  bool firstcall;
  std::ofstream* svgfile;
  std::ofstream* savefile;
  std::ifstream* jsfile;
  std::ifstream* inputfile;
  std::ifstream* ccufile;
  float gminvalue, gmaxvalue;
  float minvalue, maxvalue;
  int number_modules;
  bool temporary_file;

private:
  float oldz;
  bool saveAsSingleLayer;
  bool addPixelFlag;
  bool onlyPixelFlag;
  TLegend* buildLegend();
  std::vector<TPolyLine*> legInfos_;
  std::vector<std::string> legKeys_;
};
#endif
