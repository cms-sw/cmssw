#!/bin/bash

# to run this script, do
# ./checkCalib.sh <filepath>
# where <filepath> is the path to the output root file from CSCValiation

# example:  ./checkCalib.sh ~/work/DEV/CMSSW_1_8_0_pre8/src/RecoLocalMuon/CSCValidation/test/validationHists_muongun.root

ARG1=$1

MACRO=checkCalib_temp.C
cat > ${MACRO}<<EOF

{
  std::string newReleaseFile = "${ARG1}";
  std::string refReleaseFile = "fakeCalibReference.root";

  ofstream out;
  out.open("calib_compare_results.txt");

  TFile *fn = new TFile(newReleaseFile.c_str(),"READ");
  TFile *fo = new TFile(refReleaseFile.c_str(),"READ");

  vector<TH1F*> hCo;
  vector<TH1F*> hCn;

  hCn.push_back((TH1F*)fn->Get("Calib/hCalibGainsS"));
  hCn.push_back((TH1F*)fn->Get("Calib/hCalibXtalkSL"));
  hCn.push_back((TH1F*)fn->Get("Calib/hCalibXtalkSR"));
  hCn.push_back((TH1F*)fn->Get("Calib/hCalibXtalkIL"));
  hCn.push_back((TH1F*)fn->Get("Calib/hCalibXtalkIR"));
  hCn.push_back((TH1F*)fn->Get("Calib/hCalibPedsP"));
  hCn.push_back((TH1F*)fn->Get("Calib/hCalibPedsR"));
  hCn.push_back((TH1F*)fn->Get("Calib/hCalibNoise33"));
  hCn.push_back((TH1F*)fn->Get("Calib/hCalibNoise34"));
  hCn.push_back((TH1F*)fn->Get("Calib/hCalibNoise35"));
  hCn.push_back((TH1F*)fn->Get("Calib/hCalibNoise44"));
  hCn.push_back((TH1F*)fn->Get("Calib/hCalibNoise45"));
  hCn.push_back((TH1F*)fn->Get("Calib/hCalibNoise46"));
  hCn.push_back((TH1F*)fn->Get("Calib/hCalibNoise55"));
  hCn.push_back((TH1F*)fn->Get("Calib/hCalibNoise56"));
  hCn.push_back((TH1F*)fn->Get("Calib/hCalibNoise57"));
  hCn.push_back((TH1F*)fn->Get("Calib/hCalibNoise66"));
  hCn.push_back((TH1F*)fn->Get("Calib/hCalibNoise67"));
  hCn.push_back((TH1F*)fn->Get("Calib/hCalibNoise77"));

  hCo.push_back((TH1F*)fo->Get("Calib/hCalibGainsS"));
  hCo.push_back((TH1F*)fo->Get("Calib/hCalibXtalkSL"));
  hCo.push_back((TH1F*)fo->Get("Calib/hCalibXtalkSR"));
  hCo.push_back((TH1F*)fo->Get("Calib/hCalibXtalkIL"));
  hCo.push_back((TH1F*)fo->Get("Calib/hCalibXtalkIR"));
  hCo.push_back((TH1F*)fo->Get("Calib/hCalibPedsP"));
  hCo.push_back((TH1F*)fo->Get("Calib/hCalibPedsR"));
  hCo.push_back((TH1F*)fo->Get("Calib/hCalibNoise33"));
  hCo.push_back((TH1F*)fo->Get("Calib/hCalibNoise34"));
  hCo.push_back((TH1F*)fo->Get("Calib/hCalibNoise35"));
  hCo.push_back((TH1F*)fo->Get("Calib/hCalibNoise44"));
  hCo.push_back((TH1F*)fo->Get("Calib/hCalibNoise45"));
  hCo.push_back((TH1F*)fo->Get("Calib/hCalibNoise46"));
  hCo.push_back((TH1F*)fo->Get("Calib/hCalibNoise55"));
  hCo.push_back((TH1F*)fo->Get("Calib/hCalibNoise56"));
  hCo.push_back((TH1F*)fo->Get("Calib/hCalibNoise57"));
  hCo.push_back((TH1F*)fo->Get("Calib/hCalibNoise66"));
  hCo.push_back((TH1F*)fo->Get("Calib/hCalibNoise67"));
  hCo.push_back((TH1F*)fo->Get("Calib/hCalibNoise77"));

  float diff = 0;
  vector<int> ndiff;
  vector<float> vdiff;

  for (int k = 0; k < 19; k++){
    ndiff.push_back(0);
    for (int i = 0; i < 400; i++){
      float vo = hCo[k]->GetBinContent(i+1);
      float vn = hCn[k]->GetBinContent(i+1);
      diff = (vn - vo)/vo;
      if (fabs(diff) > 0.01){ ndiff[k] = ndiff[k] + 1; vdiff.push_back(diff); }
    }
  }

  out << "Results: " << endl;
  out << "Number of channels with diff from reference > 1% for... " << endl;
  out << "Gain Slopes: " << ndiff[0] << endl;
  out << "Xtalk Slopes Left: " << ndiff[1] << endl;
  out << "Xtalk Slopes Right: " << ndiff[2] << endl;
  out << "Xtalk Intercepts Left: " << ndiff[3] << endl;
  out << "Xtalk Intercepts Right: " << ndiff[4] << endl;
  out << "Peds: " << ndiff[5] << endl;
  out << "Ped RMS: " << ndiff[6] << endl;
  out << "Noise Matrix 33: " << ndiff[7] << endl;
  out << "Noise Matrix 34: " << ndiff[8] << endl;
  out << "Noise Matrix 35: " << ndiff[9] << endl;
  out << "Noise Matrix 44: " << ndiff[10] << endl;
  out << "Noise Matrix 45: " << ndiff[11] << endl;
  out << "Noise Matrix 46: " << ndiff[12] << endl;
  out << "Noise Matrix 55: " << ndiff[13] << endl;
  out << "Noise Matrix 56: " << ndiff[14] << endl;
  out << "Noise Matrix 57: " << ndiff[15] << endl;
  out << "Noise Matrix 66: " << ndiff[16] << endl;
  out << "Noise Matrix 67: " << ndiff[17] << endl;
  out << "Noise Matrix 77: " << ndiff[18] << endl;

  out.close();
}

EOF

root -l -q ${MACRO}

rm checkCalib_temp.C

