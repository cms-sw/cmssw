#! /usr/bin/env python3
import ROOT
import optparse
import re
import os

def main():
    parser = optparse.OptionParser()
    (options, args) = parser.parse_args()
    
    ROOT.gROOT.SetBatch(True)
    for filename in args:
       
        if "vDrift_" in filename:
            name="vdrift"
            m=re.search("vDrift_[A-Za-z]+(\d*)",filename)
            
            run=m.group(1)
            
            f = open('dtVDriftAnalyzer_cfg.py','w')
            print("from CalibMuon.DTCalibration.Workflow.addPoolDBESSource import addPoolDBESSource", file=f)
            print("from CalibMuon.DTCalibration.dtVDriftAnalyzer_cfg import process", file=f)
            print("addPoolDBESSource(process = process, moduleName = 'vDriftDB',record = 'DTMtimeRcd',tag = 'vDrift', connect = 'sqlite_file:"+filename+"')", file=f)
            print("process.dtVDriftAnalyzer.rootFileName = 'dtVDriftAnalyzer_dtVDriftCalibration"+run+".root'", file=f)
            f.close()
            os.system("cmsRun dtVDriftAnalyzer_cfg.py")
            runvdrift(name, run, "dtVDriftAnalyzer_dtVDriftCalibration"+run+".root")

        elif "DQM" in filename:
            name="DQM"
            m=re.search("R000(\d*)__",filename)
            run=m.group(1)
            path="DQMData/Run "+run+"/DT/Run summary/DTCalibValidation"
            runttrig(name, filename, path, run)
        elif "TestPulses" in filename:
            name="T0"
            runt0(name,filename,123456)
        elif "residuals" in filename:
            name="DTRV"
            m=re.search("Run(\d*)\_residuals.root",filename)
            run=m.group(1)
            path="DTResiduals"
            runttrig(name, filename, path, run)
        else:
            print("filename ?= ", filename)
            print("The file name pattern is not recognized! So we do nothing.")            

def runttrig(name, filename, path, run):
    from CalibMuon.DTCalibration.PlottingTools.plotResiduals import plot
    for SL in [1,2,3]:
        print(filename, SL, path, run)
        mean,sigma = plot(filename, SL, dir=path, run=run)
        mean[0].Print(name+run+"-SL"+str(SL)+"-mean.png")
        #mean[0].SaveAs(name+run+"-SL"+str(SL)+"-mean.root")
        sigma[0].Print(name+run+"-SL"+str(SL)+"-sigma.png")
        #sigma[0].SaveAs(name+run+"-SL"+str(SL)+"-sigma.root")

def runvdrift(name, run, filename):
    from CalibMuon.DTCalibration.PlottingTools.plotVDriftFromHistos import plot
    for SL in [1,2,3]:
        mean = plot(filename, SL, run=run)
        mean[0].Print(name+run+"-SL"+str(SL)+"-mean.png")
        #mean[0].Print(name+run+"-SL"+str(SL)+"-mean.pdf")
        #mean[0].SaveAs(name+run+"-SL"+str(SL)+"-mean.root")
        #sigma[0].Print(name+run+"-SL"+str(SL)+"-sigma.pdf")
        
def runt0(name,filename,run):
    from CalibMuon.DTCalibration.PlottingTools.plotT0FromHistos import plot
    for SL in [1,2,3]:
        mean = plot(filename, SL ,run)
        mean[0].Print(name+run+"-SL"+str(SL)+"-mean.pdf")
        mean[0].SaveAs(name+run+"-SL"+str(SL)+"-mean.root")

if __name__=="__main__":
    main()

    #plot("DTResidualValidation_210614.root", 1,dir="DTResiduals")
