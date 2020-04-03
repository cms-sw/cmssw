/*
 * =====================================================================================
 *
 *       Filename:  CSC_TPE_hAll.cc
 *
 *    Description:  Draws a hAll historgram of CSC L1 Trigger Primitives Emulator
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

#include "CSC_TPE_hAll.h"

CSC_TPE_hAll::CSC_TPE_hAll() {

  bBlank = new TBox(-1.0, -1.0, 1.0, 1.0);
  bBlank->SetFillColor(0);
  bBlank->SetLineColor(1);
  bBlank->SetLineStyle(1);

  h_ratio = new TH2F("h_ratio2","h_ratio2",36,0.5,36.5,18,0,18);
  h_ratio->GetXaxis()->SetTickLength(0);
  h_ratio->GetXaxis()->CenterTitle();
  h_ratio->SetTitle("");
  pave_title = new TPaveText(5,18.2,32,20);
  TText* text_title = pave_title->AddText("Data versus trigger emulator agreement");
  text_title->SetTextSize(0.04);
  pave_title->SetBorderSize(0);
  pave_title->SetFillColor(0);

  pave_legend_label = new TPaveText(-3.3,-2.0,0.5,-1.2);
  pave_legend_label->AddText("Problem Legend");
  //  TText *text_legend_label = pave_legend_label->AddText("Problem Legend");
  pave_legend_label->SetBorderSize(0);
  //  pave_legend_label->SetFillColor(0);
  pave_legend_label->SetFillColor(17);

  pave_legend = new TPaveText(-3.5,-2.2,36.6,-1.0);
  pave_legend->SetFillColor(17);
  pave_legend->SetBorderSize(1);

  for(int i=0; i<36; i++) {
    char stra[256]; sprintf(stra,"%i",i+1);
    tb_xaxis1[i] = new TPaveText(i+0.5, -0.5, i+1.5, 0);
    TText *text = tb_xaxis1[i]->AddText(stra);
    text->SetTextSize(0.015);
    tb_xaxis1[i]->SetFillColor(0);
    tb_xaxis1[i]->SetBorderSize(1);
  }
  for(int i=0; i<18; i++) {
    char stra[256]; sprintf(stra,"%i",i+1);
    tb_xaxis2[i] = new TPaveText(i*2+0.5, -1, 2*(i+1)+0.5, -0.5);
    TText *text = tb_xaxis2[i]->AddText(stra);
    text->SetTextSize(0.015);
    tb_xaxis2[i]->SetFillColor(0);
    tb_xaxis2[i]->SetBorderSize(1);
  }

  const char* label[18]={"ME-4/2","ME-4/1","ME-3/2","ME-3/1","ME-2/2","ME-2/1","ME-1/3","ME-1/2","ME-1/1",
			 "ME+1/1","ME+1/2","ME+1/3","ME+2/1","ME+2/2","ME+3/1","ME+3/2","ME+4/1","ME+4/2",};

  for(int i=0; i<18; i++) {
    h_ratio->GetYaxis()->SetBinLabel(i+1,label[i]);
  }

  for(int k=0; k<3; k++) {
    for(int i=0; i<36; i++) {
      for(int j=0; j<18; j++) {
	int ij=j;
	int ix=i;
	if(j<9) ij=9+j;
	else {
	  ij=17-j;
	}
	int m=1;
	if(ij==1 || ij==3 || ij==5 || ij==12 || ij==14 || ij==16)  {
	  m=2;
	  if(i>=18) continue;
	}
	tb[k][i][j] = new TPaveText(m*ix+0.5, ij+k*1.0/3, m*(ix+1)+0.5, ij+(k+1)*1.0/3);
	tb[k][i][j]->SetBorderSize(1);
	tb[k][i][j]->SetLineStyle(2);
	tb2[k][i][j] = new TPaveText(m*ix+0.5, ij, m*(ix+1)+0.5, ij+1);
	tb2[k][i][j]->SetFillColor(0);
	tb2[k][i][j]->SetBorderSize(1);
	tb2[k][i][j]->SetLineWidth(4);
	tb2[k][i][j]->SetFillStyle(0);
      }
    }
  }

  pave_total[0] = new TPaveText(36.6,15,40.9,17);
  pave_total[0]->SetBorderSize(1.0);
  pave_total[0]->AddText("L = ALCT*CLCT");
  pave_total[0]->AddText("C = CLCT            ");
  pave_total[0]->AddText("A = ALCT            ");

  pave_total[1] = new TPaveText(36.6,17,40.9,18);
  pave_total[1]->SetBorderSize(1.0);
  pave_total[1]->AddText("Trigger Primitives");

  pave_total[2] = new TPaveText(36.6,14,40.9,14.5);
  pave_total[2]->AddText("Number");
  pave_total[2]->SetTextSize(0.015);
  pave_total[2]->SetBorderSize(1.0);

  pave_total[3] = new TPaveText(36.6,12.5,40.9,14);
  pave_total[3]->SetBorderSize(1.0);

  pave_total[4] = new TPaveText(36.6,10.5,40.9,12);
  pave_total[4]->SetBorderSize(1.0);
  pave_total[4]->SetTextSize(0.014);
  pave_total[4]->AddText("Number agreement");
  pave_total[4]->AddText("with (without)");
  pave_total[4]->AddText("known problems");

  pave_total[5] = new TPaveText(36.6,8.5,40.9,10.5);
  pave_total[5]->SetTextSize(0.012);
  pave_total[5]->SetBorderSize(1.0);

  pave_total[6] = new TPaveText(36.6,6.5,40.9,8);
  pave_total[6]->SetBorderSize(1.0);
  pave_total[6]->SetTextSize(0.014);
  pave_total[6]->AddText("Match agreement");
  pave_total[6]->AddText("with (without)");
  pave_total[6]->AddText("known problems");

  pave_total[7] = new TPaveText(36.6,4.5,40.9,6.5);
  pave_total[7]->SetBorderSize(1.0);

  pave_total[8] = new TPaveText(36.6,2.5,40.9,3);
  pave_total[8]->SetBorderSize(1.0);
  pave_total[8]->SetTextSize(0.014);
  pave_total[8]->AddText("Unknown problems");

  pave_total[9] = new TPaveText(36.6,0.5,40.9,2.5);
  pave_total[9]->SetBorderSize(1.0);

  pave_total[10] = new TPaveText(36.6,-2,40.9,0);
  pave_total[10]->SetBorderSize(1.0);
  pave_total[10]->AddText("Blue box = ");
  pave_total[10]->AddText("known problem");
  pave_total[10]->SetLineColor(kBlue);
  pave_total[10]->SetLineWidth(3);

  char legend_str[6][256]={"OK","MINOR","MODERATE","BAD","Low stats","No data"};

  int myGray = 17;
  int myTeal = kTeal;
  int minor_color = kYellow+kGreen;
  for(int i=0; i<6; i++) {
    int x1 =36/6*i;
    int x2 =36/6*(i+1);
    pave_legend_a[i] = new TPaveText(0.5+x1,-2.0,0.5+x2,-1.2);
    pave_legend_a[i]->AddText(legend_str[i]);
    pave_legend_a[i]->SetBorderSize(1);
    pave_legend_a[i]->SetTextSize(0.017);
  }
  pave_legend_a[0]->SetFillColor(myTeal);
  pave_legend_a[1]->SetFillColor(minor_color);
  pave_legend_a[2]->SetFillColor(kYellow);
  pave_legend_a[3]->SetFillColor(kRed);
  pave_legend_a[4]->SetFillColor(myGray);
  pave_legend_a[5]->SetFillColor(kWhite);

}

CSC_TPE_hAll::~CSC_TPE_hAll() {
  delete bBlank;

  delete h_ratio;
  delete pave_title;
  delete pave_legend_label;
  delete pave_legend;
  for(int i=0; i<36; i++) {
    delete tb_xaxis1[i];
  }
  for(int i=0; i<18; i++) {
    delete tb_xaxis2[i];
  }
  for(int k=0; k<3; k++) {
    for(int i=0; i<36; i++) {
      for(int j=0; j<18; j++) {
	delete tb[k][i][j];
	delete tb2[k][i][j];
      }
    }
  }
  for(int i=0; i<11; i++) {
    delete pave_total[i];
  }
  for(int i=0; i<6; i++) {
    delete pave_legend_a[i];
  }
}

void CSC_TPE_hAll::draw(TH3*& me) {

  TColor *color = gROOT->GetColor(kTeal);
  color->SetRGB(1.0*51/256, 1.0, 1.0*102/256);
  gStyle->SetPalette(1, 0);
  gStyle->SetOptStat("");

  me->GetXaxis()->SetTicks("0");
  me->SetTitle("");
  int myGray = 17;
  int myTeal = kTeal;
  int minor_color = kYellow+kGreen;

  const char* moo_str[3]={"A","C","L"};
  me->Draw("");
  bBlank->Draw("l");

  int bad_chambers[4][18][36];
  for(int ifile=0; ifile<3; ifile++) {
    for(int i=0; i<18; i++) {
      for(int j=0; j<36; j++) {
	bad_chambers[ifile][i][j]=me->GetBinContent(13+ifile,i+1,j+1);
      }
    }
  }

  int num[6][36][18];
  int den[6][36][18];
  char tmp_str[256];

  for(int k=0; k<3; k++) {
    for(int i=0; i<36; i++) {
      for(int j=0; j<18; j++) {
	den[k+0][i][j]=me->GetBinContent(k*4+1,j+1,i+1);
	num[k+0][i][j]=me->GetBinContent(k*4+2,j+1,i+1);
	den[k+3][i][j]=me->GetBinContent(k*4+3,j+1,i+1);
	num[k+3][i][j]=me->GetBinContent(k*4+4,j+1,i+1);
      }
    }
  }

  h_ratio->Draw();
  pave_title->Draw();
  pave_legend_label->Draw();
  pave_legend->Draw();
  pave_legend_label->Draw();

  for(int i=0; i<36; i++) {
    tb_xaxis1[i]->Draw();
  }
  for(int i=0; i<18; i++) {
    tb_xaxis2[i]->Draw();
  }

  int problem_cnt[6]={0};

  int num_cnt1[3]={0};
  int den_cnt1[3]={0};
  int num_cnt2[3]={0};
  int den_cnt2[3]={0};

  int numb_cnt1[3]={0};
  int denb_cnt1[3]={0};
  int numb_cnt2[3]={0};
  int denb_cnt2[3]={0};

  for(int k=0; k<3; k++) {
    int mode=k;
    for(int i=0; i<36; i++) {
      for(int j=0; j<18; j++) {
	int ij=j;
	int ix=i;
	if(j<9) ij=9+j;
	else {
	  ij=17-j;
	}
	if(ij==1 || ij==3 || ij==5 || ij==12 || ij==14 || ij==16)  {
	  if(i>=18) continue;
	}

	int _num1=num[k][i][j];
	int _den1=den[k][i][j];
	int _num2=num[k+3][i][j];
	int _den2=den[k+3][i][j];

	num_cnt1[k]+=_num1;
	den_cnt1[k]+=_den1;
	num_cnt2[k]+=_num2;
	den_cnt2[k]+=_den2;

	if(bad_chambers[0][j][i]!=1 && mode==0) {
	  numb_cnt1[k]+=_num1;
	  denb_cnt1[k]+=_den1;
	  numb_cnt2[k]+=_num2;
	  denb_cnt2[k]+=_den2;
	}

	if(bad_chambers[1][j][i]!=1 && mode==1) {
	  numb_cnt1[k]+=_num1;
	  denb_cnt1[k]+=_den1;
	  numb_cnt2[k]+=_num2;
	  denb_cnt2[k]+=_den2;
	}
	if(!(bad_chambers[0][j][i]==1 || bad_chambers[1][j][i]==1) && mode==2) {
	  numb_cnt1[k]+=_num1;
	  denb_cnt1[k]+=_den1;
	  numb_cnt2[k]+=_num2;
	  denb_cnt2[k]+=_den2;
	}

	if(_den1!=0 || _den2!=0) {
	  if(!(_den1<10 && _den2<10)) {
	    tb[k][i][j]->Clear();
	    TText *text = tb[k][i][j]->AddText(moo_str[k]);
	    text->SetTextSize(0.015);
	    if(_num1==_den1 && _num2==_den2) {
	      tb[k][i][j]->SetFillColor(myTeal);
	      problem_cnt[0]++;
	    }
	    else {
	      float r = _den1!=0  ? 1.0*_num1/_den1 : 0;
	      float r2 = _den2!=0 ? 1.0*_num2/_den2 : 0;
	      if(r2<r) r=r2;
	      if(r>=0.99 ) {
		tb[k][i][j]->SetFillColor(minor_color);
		if(!(is_bad_primitive(bad_chambers[0][j][i], bad_chambers[1][j][i], mode)))
		  problem_cnt[1]++;
	      }
	      if(r<0.99 && r>=0.90) {
		tb[k][i][j]->SetFillColor(kYellow);
		if(!(is_bad_primitive(bad_chambers[0][j][i], bad_chambers[1][j][i], mode)))
		  problem_cnt[2]++;
	      }
	      if(r<0.90) {
		tb[k][i][j]->SetFillColor(kRed);
		if(!(is_bad_primitive(bad_chambers[0][j][i], bad_chambers[1][j][i], mode)))
		  problem_cnt[3]++;
	      }
	    }
	  }
	  else {tb[k][i][j]->SetFillColor(myGray);}
	  tb[k][i][j]->Draw();
	  tb2[k][i][j]->Draw();
	}
	else {
	  if(ij==0 || (ij==17 && (ix<9 || ix>12))) {
	  }
	  else {
	    problem_cnt[5]++;
	    tb[k][i][j]->SetFillColor(kWhite);
	    if(bad_chambers[k][j][i]==1) {
	      tb[k][i][j]->SetLineColor(kBlue);
	      tb[k][i][j]->SetLineWidth(2);
	      problem_cnt[4]++;
	    }
	    tb[k][i][j]->SetLineWidth(1);
	    tb[k][i][j]->SetLineStyle(2);
	    tb[k][i][j]->Draw();
	    tb2[k][i][j]->Draw();
	  }
	}
      }
    }
  }
  for(int k=0; k<3; k++) {
    int mode=k;
    for(int i=0; i<36; i++) {
      for(int j=0; j<18; j++) {
	int ij=j;
	//	int ix=i;
	if(j<9) ij=9+j;
	else {
	  ij=17-j;
	}
	if(ij==1 || ij==3 || ij==5 || ij==12 || ij==14 || ij==16)  {
	  if(i>=18) continue;
	}
	int _num1=num[k][i][j];
	int _den1=den[k][i][j];
	int _num2=num[k+3][i][j];
	int _den2=den[k+3][i][j];

	if(bad_chambers[2][j][i]==1) {
	  tb2[k][i][j]->SetLineColor(kBlue);
	  tb2[k][i][j]->Draw();
	}
	if(_num1!=_den1 || _num2!=_den2) {
	  if(bad_chambers[0][j][i]==1 && mode==0) {
	    problem_cnt[4]++;
	    tb[k][i][j]->SetLineColor(kBlue);
	    tb[k][i][j]->SetLineWidth(4);
	    tb[k][i][j]->SetLineStyle(1);
	    tb[k][i][j]->Draw();
	  }
	  if(bad_chambers[1][j][i]==1 && mode==1) {
	    problem_cnt[4]++;
	    tb[k][i][j]->SetLineColor(kBlue);
	    tb[k][i][j]->SetLineWidth(4);
	    tb[k][i][j]->SetLineStyle(1);
	    tb[k][i][j]->Draw();
	  }
	  if((bad_chambers[0][j][i]==1 || bad_chambers[1][j][i]==1) && mode==2) {
	    problem_cnt[4]++;
	    tb[k][i][j]->SetLineColor(kBlue);
	    tb[k][i][j]->SetLineWidth(4);
	    tb[k][i][j]->SetLineStyle(1);
	    tb[k][i][j]->Draw();
	  }
	}
      }
    }
  }

  pave_total[3]->Clear();
  sprintf(tmp_str,"L = %.1E",(float)den_cnt1[2]);
  pave_total[3]->AddText(tmp_str);
  sprintf(tmp_str,"C = %.1E",(float)den_cnt1[1]);
  pave_total[3]->AddText(tmp_str);
  sprintf(tmp_str,"A = %.1E",(float)den_cnt1[0]);
  pave_total[3]->AddText(tmp_str);

  pave_total[5]->Clear();
  sprintf(tmp_str,"L = %.2f%% (%.2f%%)",den_cnt1[2]!=0 ? 100.0*num_cnt1[2]/den_cnt1[2] : 0, denb_cnt1[2]!=0 ? 100.0*numb_cnt1[2]/denb_cnt1[2] : 0);
  pave_total[5]->AddText(tmp_str);
  sprintf(tmp_str,"C = %.2f%% (%.2f%%)",den_cnt1[1]!=0 ? 100.0*num_cnt1[1]/den_cnt1[1] : 0, denb_cnt1[1]!=0 ? 100.0*numb_cnt1[1]/denb_cnt1[1] : 0);
  pave_total[5]->AddText(tmp_str);
  sprintf(tmp_str,"A = %.2f%% (%.2f%%)",den_cnt1[0]!=0 ? 100.0*num_cnt1[0]/den_cnt1[0] : 0, denb_cnt1[0]!=0 ? 100.0*numb_cnt1[0]/denb_cnt1[0] : 0);
  pave_total[5]->AddText(tmp_str);

  pave_total[7]->Clear();
  sprintf(tmp_str,"L = %.2f%% (%.2f%%)",den_cnt2[2]!=0 ? 100.0*num_cnt2[2]/den_cnt2[2] : 0, denb_cnt2[2]!=0 ? 100.0*numb_cnt2[2]/denb_cnt2[2] : 0);
  pave_total[7]->AddText(tmp_str);
  sprintf(tmp_str,"C = %.2f%% (%.2f%%)",den_cnt2[1]!=0 ? 100.0*num_cnt2[1]/den_cnt2[1] : 0, denb_cnt2[1]!=0 ? 100.0*numb_cnt2[1]/denb_cnt2[1] : 0);
  pave_total[7]->AddText(tmp_str);
  sprintf(tmp_str,"A = %.2f%% (%.2f%%)",den_cnt2[0]!=0 ? 100.0*num_cnt2[0]/den_cnt2[0] : 0, denb_cnt2[0]!=0 ? 100.0*numb_cnt2[0]/denb_cnt2[0] : 0);
  pave_total[7]->AddText(tmp_str);

  pave_total[9]->Clear();
  sprintf(tmp_str,"MINOR = %i",problem_cnt[1]);
  pave_total[9]->AddText(tmp_str);
  sprintf(tmp_str,"MODERATE = %i",problem_cnt[2]);
  pave_total[9]->AddText(tmp_str);
  sprintf(tmp_str,"BAD = %i",problem_cnt[3]);
  pave_total[9]->AddText(tmp_str);

  for(int i=0; i<11; i++) {
  pave_total[i]->Draw();
  }
  for(int i=0; i<6; i++) {
    pave_legend_a[i]->Draw();
  }
  //  pave_legend_a[4]->Draw();

}

bool CSC_TPE_hAll::is_bad_primitive(int bad1, int bad2, int mode) {
  if(bad1==1 && mode==0) {
    return true;
  }

  if(bad2==1 && mode==1) {
    return true;
  }
  if((bad1==1 || bad2==1) && mode==2) {
    return true;
  }
  return false;
}
