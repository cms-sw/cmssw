#!/bin/bash

# to run this script, do
# ./checkResiduals.sh <filepath_new> <filepath_ref>
# where <filepath_new> and <filepath_ref> are the paths to the output root files from CSCValiation

# example:  ./checkResiduals.sh CMSSW_1_8_0_pre8/src/RecoLocalMuon/CSCValidation/test/validationHists_muongun.root CMSSW_1_7_4/src/RecoLocalMuon/CSCValidation/test/validationHists_muongun.root

ARG1=$1
ARG2=$2

MACRO=checkResid_temp.C
cat > ${MACRO}<<EOF

{

  std::string newReleaseFile = "${ARG1}";
  std::string refReleaseFile = "${ARG2}";

  ofstream out;
  out.open("residual_compare_results.txt");

  TFile *fn = new TFile(newReleaseFile.c_str(),"READ");
  TFile *fo = new TFile(refReleaseFile.c_str(),"READ");

  TF1 *f1 = new TF1("f1","gaus",-0.1,0.1);
  TF1 *f2 = new TF1("f2","gaus",-0.1,0.1);

  TH1F *ho;
  TH1F *hn;

  // ME 1/1a
  ho = (TH1F*)fo->Get("recHits/hSResid11a");
  hn = (TH1F*)fn->Get("recHits/hSResid11a");

  ho->Fit("f1","RNQ");
  hn->Fit("f2","RNQ");

  float sigmao11a = f1->GetParameter(2);
  float sigmaoerr11a = f1->GetParError(2);
  float sigman11a = f2->GetParameter(2);
  float sigmanerr11a = f2->GetParError(2);
  float diff11a = (sigman11a - sigmao11a);
  float differr11a = sqrt(sigmaoerr11a*sigmaoerr11a + sigmanerr11a*sigmanerr11a);

  out << "Results for ME11a:" << endl;
  out << endl;
  out << "Ref Fit: " << sigmao11a << " +/- " << sigmaoerr11a << endl;
  out << "New Fit: " << sigman11a << " +/- " << sigmanerr11a << endl;
  out << endl;
  out << "Diff: " << diff11a << " +/- " << differr11a << endl;
  out << endl;
  if (fabs(diff11a) > 0.01) out << "CAUTION!!! Residuals for this chamber type have changed!!!" << endl;
  out << endl;
  out << endl;

  // ME 1/1b
  ho = (TH1F*)fo->Get("recHits/hSResid11b");
  hn = (TH1F*)fn->Get("recHits/hSResid11b");

  ho->Fit("f1","RNQ");
  hn->Fit("f2","RNQ");

  float sigmao11b = f1->GetParameter(2);
  float sigmaoerr11b = f1->GetParError(2);
  float sigman11b = f2->GetParameter(2);
  float sigmanerr11b = f2->GetParError(2);
  float diff11b = (sigman11b - sigmao11b);
  float differr11b = sqrt(sigmaoerr11b*sigmaoerr11b + sigmanerr11b*sigmanerr11b);

  out << "Results for ME11b:" << endl;
  out << endl;
  out << "Ref Fit: " << sigmao11b << " +/- " << sigmaoerr11b << endl;
  out << "New Fit: " << sigman11b << " +/- " << sigmanerr11b << endl;
  out << endl;
  out << "Diff: " << diff11b << " +/- " << differr11b << endl;
  out << endl;
  if (fabs(diff11b) > 0.01) out << "CAUTION!!! Residuals for this chamber type have changed!!!" << endl;
  out << endl;
  out << endl;

  // ME 1/2
  ho = (TH1F*)fo->Get("recHits/hSResid12");
  hn = (TH1F*)fn->Get("recHits/hSResid12");

  ho->Fit("f1","RNQ");
  hn->Fit("f2","RNQ");

  float sigmao12 = f1->GetParameter(2);
  float sigmaoerr12 = f1->GetParError(2);
  float sigman12 = f2->GetParameter(2);
  float sigmanerr12 = f2->GetParError(2);
  float diff12 = (sigman12 - sigmao12);
  float differr12 = sqrt(sigmaoerr12*sigmaoerr12 + sigmanerr12*sigmanerr12);

  out << "Results for ME12:" << endl;
  out << endl;
  out << "Ref Fit: " << sigmao12 << " +/- " << sigmaoerr12 << endl;
  out << "New Fit: " << sigman12 << " +/- " << sigmanerr12 << endl;
  out << endl;
  out << "Diff: " << diff12 << " +/- " << differr12 << endl;
  out << endl;
  if (fabs(diff12) > 0.01) out << "CAUTION!!! Residuals for this chamber type have changed!!!" << endl;
  out << endl;
  out << endl;

  // ME 1/3
  ho = (TH1F*)fo->Get("recHits/hSResid13");
  hn = (TH1F*)fn->Get("recHits/hSResid13");

  ho->Fit("f1","RNQ");
  hn->Fit("f2","RNQ");

  float sigmao13 = f1->GetParameter(2);
  float sigmaoerr13 = f1->GetParError(2);
  float sigman13 = f2->GetParameter(2);
  float sigmanerr13 = f2->GetParError(2);
  float diff13 = (sigman13 - sigmao13);
  float differr13 = sqrt(sigmaoerr13*sigmaoerr13 + sigmanerr13*sigmanerr13);

  out << "Results for ME13:" << endl;
  out << endl;
  out << "Ref Fit: " << sigmao13 << " +/- " << sigmaoerr13 << endl;
  out << "New Fit: " << sigman13 << " +/- " << sigmanerr13 << endl;
  out << endl;
  out << "Diff: " << diff13 << " +/- " << differr13 << endl;
  out << endl;
  if (fabs(diff13) > 0.01) out << "CAUTION!!! Residuals for this chamber type have changed!!!" << endl;
  out << endl;
  out << endl;

  // ME 2/1
  ho = (TH1F*)fo->Get("recHits/hSResid21");
  hn = (TH1F*)fn->Get("recHits/hSResid21");

  ho->Fit("f1","RNQ");
  hn->Fit("f2","RNQ");

  float sigmao21 = f1->GetParameter(2);
  float sigmaoerr21 = f1->GetParError(2);
  float sigman21 = f2->GetParameter(2);
  float sigmanerr21 = f2->GetParError(2);
  float diff21 = (sigman21 - sigmao21);
  float differr21 = sqrt(sigmaoerr21*sigmaoerr21 + sigmanerr21*sigmanerr21);

  out << "Results for ME21:" << endl;
  out << endl;
  out << "Ref Fit: " << sigmao21 << " +/- " << sigmaoerr21 << endl;
  out << "New Fit: " << sigman21 << " +/- " << sigmanerr21 << endl;
  out << endl;
  out << "Diff: " << diff21 << " +/- " << differr21 << endl;
  out << endl;
  if (fabs(diff21) > 0.01) out << "CAUTION!!! Residuals for this chamber type have changed!!!" << endl;
  out << endl;
  out << endl;

  // ME 2/2
  ho = (TH1F*)fo->Get("recHits/hSResid22");
  hn = (TH1F*)fn->Get("recHits/hSResid22");

  ho->Fit("f1","RNQ");
  hn->Fit("f2","RNQ");

  float sigmao22 = f1->GetParameter(2);
  float sigmaoerr22 = f1->GetParError(2);
  float sigman22 = f2->GetParameter(2);
  float sigmanerr22 = f2->GetParError(2);
  float diff22 = (sigman22 - sigmao22);
  float differr22 = sqrt(sigmaoerr22*sigmaoerr22 + sigmanerr22*sigmanerr22);

  out << "Results for ME22:" << endl;
  out << endl;
  out << "Ref Fit: " << sigmao22 << " +/- " << sigmaoerr22 << endl;
  out << "New Fit: " << sigman22 << " +/- " << sigmanerr22 << endl;
  out << endl;
  out << "Diff: " << diff22 << " +/- " << differr22 << endl;
  out << endl;
  if (fabs(diff22) > 0.01) out << "CAUTION!!! Residuals for this chamber type have changed!!!" << endl;
  out << endl;
  out << endl;

  // ME 3/1
  ho = (TH1F*)fo->Get("recHits/hSResid31");
  hn = (TH1F*)fn->Get("recHits/hSResid31");

  ho->Fit("f1","RNQ");
  hn->Fit("f2","RNQ");

  float sigmao31 = f1->GetParameter(2);
  float sigmaoerr31 = f1->GetParError(2);
  float sigman31 = f2->GetParameter(2);
  float sigmanerr31 = f2->GetParError(2);
  float diff31 = (sigman31 - sigmao31);
  float differr31 = sqrt(sigmaoerr31*sigmaoerr31 + sigmanerr31*sigmanerr31);

  out << "Results for ME31:" << endl;
  out << endl;
  out << "Ref Fit: " << sigmao31 << " +/- " << sigmaoerr31 << endl;
  out << "New Fit: " << sigman31 << " +/- " << sigmanerr31 << endl;
  out << endl;
  out << "Diff: " << diff31 << " +/- " << differr31 << endl;
  out << endl;
  if (fabs(diff31) > 0.01) out << "CAUTION!!! Residuals for this chamber type have changed!!!" << endl;
  out << endl;
  out << endl;

  // ME 3/2
  ho = (TH1F*)fo->Get("recHits/hSResid32");
  hn = (TH1F*)fn->Get("recHits/hSResid32");

  ho->Fit("f1","RNQ");
  hn->Fit("f2","RNQ");

  float sigmao32 = f1->GetParameter(2);
  float sigmaoerr32 = f1->GetParError(2);
  float sigman32 = f2->GetParameter(2);
  float sigmanerr32 = f2->GetParError(2);
  float diff32 = (sigman32 - sigmao32);
  float differr32 = sqrt(sigmaoerr32*sigmaoerr32 + sigmanerr32*sigmanerr32);

  out << "Results for ME32:" << endl;
  out << endl;
  out << "Ref Fit: " << sigmao32 << " +/- " << sigmaoerr32 << endl;
  out << "New Fit: " << sigman32 << " +/- " << sigmanerr32 << endl;
  out << endl;
  out << "Diff: " << diff32 << " +/- " << differr32 << endl;
  out << endl;
  if (fabs(diff32) > 0.01) out << "CAUTION!!! Residuals for this chamber type have changed!!!" << endl;
  out << endl;
  out << endl;

  // ME 4/1
  ho = (TH1F*)fo->Get("recHits/hSResid41");
  hn = (TH1F*)fn->Get("recHits/hSResid41");

  ho->Fit("f1","RNQ");
  hn->Fit("f2","RNQ");

  float sigmao41 = f1->GetParameter(2);
  float sigmaoerr41 = f1->GetParError(2);
  float sigman41 = f2->GetParameter(2);
  float sigmanerr41 = f2->GetParError(2);
  float diff41 = (sigman41 - sigmao41);
  float differr41 = sqrt(sigmaoerr41*sigmaoerr41 + sigmanerr41*sigmanerr41);

  out << "Results for ME41:" << endl;
  out << endl;
  out << "Ref Fit: " << sigmao41 << " +/- " << sigmaoerr41 << endl;
  out << "New Fit: " << sigman41 << " +/- " << sigmanerr41 << endl;
  out << endl;
  out << "Diff: " << diff41 << " +/- " << differr41 << endl;
  out << endl;
  if (fabs(diff41) > 0.01) out << "CAUTION!!! Residuals for this chamber type have changed!!!" << endl;

  out.close();

}

EOF

root -l -q checkResid_temp.C

rm checkResid_temp.C 
