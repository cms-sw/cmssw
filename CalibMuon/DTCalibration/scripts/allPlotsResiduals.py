#! /usr/bin/env python
from __future__ import print_function
import ROOT
import optparse
import re
import os

def main():
    parser = optparse.OptionParser()
    (options, args) = parser.parse_args()
    ROOT.gROOT.SetBatch(True)
    for filename in args:
        if "vDrift_segment_" in filename:
            #name="DQM"
            m=re.search("vDrift_segment_(\d*)",filename)
            run=m.group(1)
            
            #path="DQMData/Run "+run+"/DT/Run summary/DTCalibValidation"

            f = open('dtVDriftAnalyzer_cfg.py','w')
            print("from CalibMuon.DTCalibration.Workflow.addPoolDBESSource import addPoolDBESSource", file=f)
            print("from CalibMuon.DTCalibration.dtVDriftAnalyzer_cfg import process", file=f)
            print("addPoolDBESSource(process = process, moduleName = 'vDriftDB',record = 'DTMtimeRcd',tag = 'vDrift', connect = 'sqlite_file:"+filename+"')", file=f)
            print("process.dtVDriftAnalyzer.rootFileName = 'dtVDriftAnalyzer_dtVDriftCalibration"+run+".root'", file=f)
            f.close()
            os.system("cmsRun dtVDriftAnalyzer_cfg.py")
            name="vdrift"
            runvdrift(name, run, "dtVDriftAnalyzer_dtVDriftCalibration"+run+".root")

        if "vDrift_meantimer_" in filename:
            #name="DQM"
            m=re.search("vDrift_meantimer_(\d*)",filename)
            run=m.group(1)
            
            #path="DQMData/Run "+run+"/DT/Run summary/DTCalibValidation"

            f = open('dtVDriftAnalyzer_cfg.py','w')
            print("from CalibMuon.DTCalibration.Workflow.addPoolDBESSource import addPoolDBESSource", file=f)
            print("from CalibMuon.DTCalibration.dtVDriftAnalyzer_cfg import process", file=f)
            print("addPoolDBESSource(process = process, moduleName = 'vDriftDB',record = 'DTMtimeRcd',tag = 'vDrift', connect = 'sqlite_file:"+filename+"')", file=f)
            print("process.dtVDriftAnalyzer.rootFileName = 'dtVDriftAnalyzer_dtVDriftCalibration"+run+".root'", file=f)
            f.close()
            os.system("cmsRun dtVDriftAnalyzer_cfg.py")
            name="vdrift"
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
        else:
            name="DTRV"
            m=re.search("DTResidualValidation_(\d*)\.root",filename)
            run=m.group(1)
            path="DTResiduals"
            runttrig(name, filename, path, run)

def runttrig(name, filename, path, run):
    from CalibMuon.DTCalibration.PlottingTools.plotResiduals import *
    for SL in [1,2,3]:
        mean,sigma = plot(filename, SL,dir=path)
        mean[0].Print(name+run+"-SL"+str(SL)+"-mean.pdf")
        mean[0].SaveAs(name+run+"-SL"+str(SL)+"-mean.root")
        sigma[0].Print(name+run+"-SL"+str(SL)+"-sigma.pdf")
        sigma[0].SaveAs(name+run+"-SL"+str(SL)+"-sigma.root")
def runvdrift(name, run, filename):
    from CalibMuon.DTCalibration.PlottingTools.plotVDriftFromHistos import *
    for SL in [1,2,3]:
        mean = plot(filename, SL)
        mean[0].Print(name+run+"-SL"+str(SL)+"-mean.pdf")
        mean[0].SaveAs(name+run+"-SL"+str(SL)+"-mean.root")
        #sigma[0].Print(name+run+"-SL"+str(SL)+"-sigma.pdf")
        
    #plot("DTResidualValidation_210614.root", 1,dir="DTResiduals")
    #plot("DQM_V0001_R000210634__StreamExpress__HIRun2013-DtCalib-Express-v1-dtTtrigCalibrationFromResiduals-NEW-usedb-rev1__ALCARECO.root", 1,dir="DQMData/Run 210634/DT/Run summary/DTCalibValidation")
def runt0(name,filename,run):
    from CalibMuon.DTCalibration.PlottingTools.plotT0FromHistos import *
    for SL in [1,2,3]:
        mean = plot(filename, SL ,run)
        mean[0].Print(name+run+"-SL"+str(SL)+"-mean.pdf")
        mean[0].SaveAs(name+run+"-SL"+str(SL)+"-mean.root")

if __name__=="__main__":
    main()
