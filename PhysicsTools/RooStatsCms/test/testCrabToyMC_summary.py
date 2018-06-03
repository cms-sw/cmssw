#!/usr/bin/env python

import os
import ROOT
import tarfile
import sys

if (len(sys.argv)<2):
    print "Reads the job outputs and collects the parameters values in an histograms."
    print "dirX represents the directory where the output tgz files are stored."
    print "Usage:",sys.argv[0],"dir1, dir2, dir3, ..."
    sys.exit(1)

mu_h=ROOT.TH1F("mu_h","#mu fitted values",50,-1,1)
sigma_h=ROOT.TH1F("sigma_h","#sigma fitted values",100,0,2)
lambda_h=ROOT.TH1F("lambda_h","#lambda fitted values",100,-.5,.5)
s_h=ROOT.TH1F("s_h","signal yield fitted values",200,700,1300)
b_h=ROOT.TH1F("b_h","background yield fitted values",200,700,1300)

mu_h_pull=ROOT.TH1F("mu_h_pull","Pull #mu fitted values",40,-5,5)
sigma_h_pull=ROOT.TH1F("sigma_h_pull","Pull #sigma fitted values",40,-5,5)
lambda_h_pull=ROOT.TH1F("lambda_h_pull","Pull #lambda fitted values",40,-5,5)
s_h_pull=ROOT.TH1F("s_h_pull","Pull signal yield fitted values",40,-5,5)
b_h_pull=ROOT.TH1F("b_h_pull","Pull background yield fitted values",40,-5,5)


h_dict={"RooRealVar::mu":(mu_h,mu_h_pull),
        "RooRealVar::sigma":(sigma_h,sigma_h_pull),
        "RooRealVar::lambda":(lambda_h,lambda_h_pull),
        "RooRealVar::s":(s_h,s_h_pull),
        "RooRealVar::b":(b_h,b_h_pull)}


real_val_dict={"RooRealVar::mu":0,
               "RooRealVar::sigma":1,
               "RooRealVar::lambda":-0.1,
               "RooRealVar::s":1000,
               "RooRealVar::b":1000}


targz_files=[]

# retrieve the targz files
for directory in sys.argv[1:]:
    for targzfile in [name for name in os.listdir(directory) if ".tgz" in name]:
        targz_files.append(directory+"/"+targzfile)

# inspect them
for targzfile in targz_files:
    f = tarfile.open(targzfile,"r:gz")
    for txtfilename in [name for name in f.getnames() if ".txt" in name]:
        print "Xtracting ",txtfilename
        txtfile = f.extractfile(txtfilename)
        for line in txtfile.readlines():
            for key,histos in h_dict.items():
                if key in line and "+/-" in line:
                    #disambiguation between sigma and s
                    if key == "RooRealVar::s" and "RooRealVar::sigma" in line: continue
                    value=float(line.split(" ")[2])
                    value_err=float(line.split(" ")[4])
                    histos[0].Fill(value)

                    real_val=real_val_dict[key]
                    pull=(value - real_val)/value_err
                    histos[1].Fill(pull)
                    #print "Value is %s real value is %s and sigma is %s\n" %(value,real_val,value_err)

ROOT.gROOT.SetStyle("Plain")
ROOT.gStyle.SetOptStat(2211)
# Write histos on a TFile
ofilename="testCrabToyMC_summary.root"
ofile = ROOT.TFile(ofilename,"RECREATE");ofile.cd()
for key,histos in h_dict.items():
    for histo in histos:
        histoname=histo.GetName()
        print "Saving ",histoname," on file ",ofilename
        histo.Write()
        c=ROOT.TCanvas()
        c.cd()
        histo.Draw()
        c.Print(histoname+".png")

ofile.Close()