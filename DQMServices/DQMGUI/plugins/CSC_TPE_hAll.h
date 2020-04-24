#ifndef CSC_TPE_hAll_H
#define CSC_TPE_hAll_H

/*
 * =====================================================================================
 *
 *       Filename:  CSC_TPE_hAll.h
 *
 *    Description:  CSC TPE render plugin
 *
 *        Version:  1.0
 *        Created:  02/02/2011
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Chad Jarvis, chad.jarvis@cern.ch
 *        Company:  CERN, CH
 *
 * =====================================================================================
 */

#include <TBox.h>
#include <TCanvas.h>
#include <TColor.h>
#include <TH1.h>
#include <TH2.h>
#include <TH3.h>
#include <TPRegexp.h>
#include <TPave.h>
#include <TPaveText.h>
#include <TROOT.h>
#include <TStyle.h>
#include <TText.h>
#include <bitset>
#include <cmath>
#include <iostream>
#include <string>

/**
 * @class CSC_TPE_hAll
 */
class CSC_TPE_hAll {
private:
  TBox* bBlank;

  TH2F* h_ratio;
  TPaveText* pave_title;
  TPaveText* pave_legend_label;
  TPaveText* pave_legend;
  TPaveText* tb_xaxis1[36];
  TPaveText* tb_xaxis2[18];
  TPaveText* tb[3][36][18];
  TPaveText* tb2[3][36][18];
  TPaveText* pave_total[11];
  TPaveText* pave_legend_a[6];

public:
  CSC_TPE_hAll();
  ~CSC_TPE_hAll();
  void draw(TH3*& me);
  bool is_bad_primitive(int bad1, int bad2, int mode);

private:
};

#endif
