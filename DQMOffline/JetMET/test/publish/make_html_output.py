#!/usr/bin/env python

#!/usr/bin/python

import ROOT
import sys,os,string,errno,shutil
import code
from ROOT import gROOT, gDirectory, gPad, gSystem, gRandom, gStyle
from ROOT import TH1, TH1F, TCanvas, TFile

ROOT.gROOT.SetBatch()
ROOT.gStyle.SetMarkerStyle(2)
ROOT.gStyle.SetMarkerSize(0.6)
ROOT.gStyle.SetMarkerColor(46)
ROOT.gStyle.SetOptStat(0)

inputdir=sys.argv[1]
outputdir=sys.argv[2]

print str(inputdir)
print str(outputdir)

files = os.listdir(inputdir)
print files

ROOT.gROOT.ProcessLine(".L ./plot_METDQM.C")
ROOT.gROOT.ProcessLine(".L ./plot_jets_data_vs_MC.C")
mettype = ['CaloMET','CaloMETNoHF''PfMET','TcMET','MuCorrMET']
        
rmscanvas   = TCanvas("rmscanvas","rmsplots",1280,768)
rmscanvas.Divide(2,2)
meancanvas   = TCanvas("meancanvas","meanplots",1280,768)
meancanvas.Divide(2,2)

h_SumEt_rms = TH1F("h_SumEt_rms","", len(files),0,len(files))
h_MET_rms   = TH1F("h_MET_rms",  "", len(files),0,len(files))
h_MEx_rms   = TH1F("h_MEx_rms",  "", len(files),0,len(files))
h_MEy_rms   = TH1F("h_MEy_rms",  "", len(files),0,len(files))

h_SumEt_mean = TH1F("h_SumEt_mean","", len(files),0,len(files))
h_MET_mean   = TH1F("h_MET_mean",  "", len(files),0,len(files))
h_MEx_mean   = TH1F("h_MEx_mean",  "", len(files),0,len(files))
h_MEy_mean   = TH1F("h_MEy_mean",  "", len(files),0,len(files))

h_SumEt_rms.GetXaxis().SetTitle("Run #")
h_SumEt_rms.GetYaxis().SetTitle("#sumE_{T} RMS")
h_SumEt_rms.GetXaxis().SetTitleOffset(1.6)
h_MET_rms.GetXaxis().SetTitle("Run #")
h_MET_rms.GetYaxis().SetTitle("#slashE_{T} RMS")
h_MET_rms.GetXaxis().SetTitleOffset(1.6)
h_MEx_rms.GetXaxis().SetTitle("Run #")
h_MEx_rms.GetYaxis().SetTitle("#slashE_{x} RMS")
h_MEx_rms.GetXaxis().SetTitleOffset(1.6)
h_MEy_rms.GetXaxis().SetTitle("Run #")
h_MEy_rms.GetYaxis().SetTitle("#slashE_{y} RMS")
h_MEy_rms.GetXaxis().SetTitleOffset(1.6)

h_SumEt_mean.GetXaxis().SetTitle("Run #")
h_SumEt_mean.GetYaxis().SetTitle("#sumE_{T} Mean")
h_SumEt_mean.GetXaxis().SetTitleOffset(1.6)
h_MET_mean.GetXaxis().SetTitle("Run #")
h_MET_mean.GetYaxis().SetTitle("#slashE_{T} Mean")
h_MET_mean.GetXaxis().SetTitleOffset(1.6)
h_MEx_mean.GetXaxis().SetTitle("Run #")
h_MEx_mean.GetYaxis().SetTitle("#slashE_{x} Mean")
h_MEx_mean.GetXaxis().SetTitleOffset(1.6)
h_MEy_mean.GetXaxis().SetTitle("Run #")
h_MEy_mean.GetYaxis().SetTitle("#slashE_{y} Mean")
h_MEy_mean.GetXaxis().SetTitleOffset(1.6)


for file in files:
    if (os.path.isfile(os.path.join(inputdir,file))):
        if ((file.find("R000000001")!=-1) or (file.find("MC")!=-1)):
            reference = os.path.join(inputdir,file)
            if (outputdir.find("afs")!=1):
                source = os.path.join(inputdir,file)
                destination = os.path.join(outputdir)
                try: shutil.copy2(source,destination)
                except IOError, err:
                    print "cannot copy:\n%s\n to\n%s"%(source,destination)
                    print "I/O error(%d): %s"%(err.errno,err.strerror)
            else:
                print "will not copy refernce root files to afs due to space restrictions"
            files.remove(file)

                
for index,file in enumerate(files):
    if (os.path.isfile(os.path.join(inputdir,file))):
        if ((file.find("R000000001")==-1) and (file.find("MC")==-1)):
            infilename = os.path.join(inputdir,file)
            run = file[10:20]
            run = run.lstrip("R")
            run = run.lstrip("0")

            
            h_SumEt_rms.GetXaxis().SetBinLabel(index+1,run)
            h_SumEt_rms.GetXaxis().LabelsOption("v")

            h_MET_rms.GetXaxis().SetBinLabel(index+1,run)
            h_MET_rms.GetXaxis().LabelsOption("v")

            h_MEx_rms.GetXaxis().SetBinLabel(index+1,run)
            h_MEx_rms.GetXaxis().LabelsOption("v")

            h_MEy_rms.GetXaxis().SetBinLabel(index+1,run)
            h_MEy_rms.GetXaxis().LabelsOption("v")
            
            h_SumEt_mean.GetXaxis().SetBinLabel(index+1,run)
            h_SumEt_mean.GetXaxis().LabelsOption("v")

            h_MET_mean.GetXaxis().SetBinLabel(index+1,run)
            h_MET_mean.GetXaxis().LabelsOption("v")

            h_MEx_mean.GetXaxis().SetBinLabel(index+1,run)
            h_MEx_mean.GetXaxis().LabelsOption("v")

            h_MEy_mean.GetXaxis().SetBinLabel(index+1,run)
            h_MEy_mean.GetXaxis().LabelsOption("v")
            
            # run the MET comparison plots
            try: os.makedirs(os.path.join(outputdir,run,"METDQM"))
            except OSError, err:
                if err.errno != errno.EEXIST: raise
               
            for test in mettype:
                try: os.mkdir(os.path.join(outputdir,run,"METDQM",test))
                except OSError, err:
                    if err.errno != errno.EEXIST: raise

                rootfile = TFile(infilename)

                if (test=="CaloMET"):
                    try: os.mkdir(os.path.join(outputdir,run,"CaloTowers"))
                    except OSError, err:
                        if err.errno != errno.EEXIST: raise


                    hist = "DQMData/Run %s/JetMET/Run summary/MET/%s/BasicCleanup/METTask_CaloSumET"%(run,test)
                    rms  = rootfile.Get(hist).GetRMS()
                    mean = rootfile.Get(hist).GetMean()
                    h_SumEt_rms.SetBinContent(index+1,rms)
                    h_SumEt_mean.SetBinContent(index+1,mean)
                    
                    hist = "DQMData/Run %s/JetMET/Run summary/MET/%s/BasicCleanup/METTask_CaloMET"%(run,test)
                    rms  = rootfile.Get(hist).GetRMS()
                    mean = rootfile.Get(hist).GetMean()
                    h_MET_rms.SetBinContent(index+1,rms)
                    h_MET_mean.SetBinContent(index+1,mean)
                    
                    hist = "DQMData/Run %s/JetMET/Run summary/MET/%s/BasicCleanup/METTask_CaloMEx"%(run,test)
                    rms  = rootfile.Get(hist).GetRMS()
                    mean = rootfile.Get(hist).GetMean()
                    h_MEx_rms.SetBinContent(index+1,rms)
                    h_MEx_mean.SetBinContent(index+1,mean)
                    
                    hist = "DQMData/Run %s/JetMET/Run summary/MET/%s/BasicCleanup/METTask_CaloMEy"%(run,test)
                    rms  = rootfile.Get(hist).GetRMS()
                    mean = rootfile.Get(hist).GetMean()
                    h_MEy_rms.SetBinContent(index+1,rms)
                    h_MEy_mean.SetBinContent(index+1,mean)

                plotdirectory = "DQMData/Run %s/JetMET/Run summary/MET/%s"%(run,test)
                if (rootfile.GetDirectory(plotdirectory)):
                    metcmd = "plot_METDQM(\"%s\",\"%s\",%d,\"%s\",\"%s\")"%(infilename,reference,int(float(run)),outputdir,test)
                    print metcmd
                    ROOT.gROOT.ProcessLine(metcmd)
                else :
                    print "Directory "+plotdirectory+" does not exist, not running creating plots."
                    
            # run the jet comparison plots
            try: os.makedirs(os.path.join(outputdir,run,"JetDQM","CaloJetAntiKt"))
            except OSError, err:
                if err.errno != errno.EEXIST: raise

            jetcmd = "plot_jets_data_vs_MC(\"%s\",\"%s\",%d,\"%s\")"%(infilename,reference,int(float(run)),outputdir)
            print jetcmd
            ROOT.gROOT.ProcessLine(jetcmd)
            source = os.path.join(os.getcwd(),"result.root")
            destination = os.path.join(outputdir,run,"JetDQM")
            try: shutil.copy2(source,destination)
            except IOError, err:
                print "cannot copy:\n%s\n to\n%s"%(source,destination)
                print "I/O error(%d): %s"%(err.errno,err.strerror)
            

            if (outputdir.find("afs")!=1):
                source = os.path.join(inputdir,file)
                destination = os.path.join(outputdir,run)
                try: shutil.copy2(source,destination)
                except IOError, err:
                    print "cannot copy:\n%s\n to\n%s"%(source,destination)
                    print "I/O error(%d): %s"%(err.errno,err.strerror)
            else:
                print "will not copy source root files to afs due to space restrictions"
            
            source = os.path.join(os.getcwd(),"UFAV.html")
            destination = os.path.join(outputdir,run)
            try: shutil.copy2(source,destination)
            except IOError, err:
                print "cannot copy:\n%s\n to\n%s"%(source,destination)
                print "I/O error(%d): %s"%(err.errno,err.strerror)
            
rmscanvas.cd(1)
h_SumEt_rms.Draw("p")
rmscanvas.cd(2)
h_MET_rms.Draw("p")
rmscanvas.cd(3)
h_MEx_rms.Draw("p")
rmscanvas.cd(4)
h_MEy_rms.Draw("p")

meancanvas.cd(1)
h_SumEt_mean.Draw("p")
meancanvas.cd(2)
h_MET_mean.Draw("p")
meancanvas.cd(3)
h_MEx_mean.Draw("p")
meancanvas.cd(4)
h_MEy_mean.Draw("p")

outfilename = "%s/rms.gif"%(outputdir)
rmscanvas.SaveAs(outfilename)
outfilename = "%s/mean.gif"%(outputdir)
meancanvas.SaveAs(outfilename)
