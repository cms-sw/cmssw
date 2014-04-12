import os

class ProducePlots:
    def makePlots(self):
        # print "cd "+self.CMSSW_Version+"/src; eval `scramv1 r -sh`; cd "+self.Dir+"; mkdir -pv HistoricDQMPlots/"+self.TagName+"; cd HistoricDQMPlots/"+self.TagName+"; "+self.DetName+"HDQMInspector "+self.Database+" "+self.TagName+" "+self.Password+" empty 40"
        # Plots without any selection
        plotsDir = self.Dir+"/HistoricDQMPlots/Plots_"+self.DetName+"HistoricInfoClient"
        self.plotAllAndLast40Runs(plotsDir, "NoSelection")
        # Plots with good runs selection
        plotsDir = self.Dir+"/HistoricDQMPlots/Plots_"+self.DetName+"HistoricInfoClient"
        import SelectRuns
        selectRuns = SelectRuns.SelectRuns()
        selectRuns.BaseDir = self.Dir
        selectRuns.Group = self.Group
        selectRuns.FirstRun = self.FirstRun
        selectRuns.FileName = self.Dir+"/SelectedGoodRuns_"+self.DetName+"_"+self.RunType+".txt"
        selectRuns.HLTNameFilter = ""
        selectRuns.QualityFlag = self.QualityFlag
        selectRuns.makeList()
        self.plotAllAndLast40Runs(plotsDir, "GoodRuns", selectRuns.FileName)

    def plotAllAndLast40Runs(self, plotsDir, selectionType, selectionFileName = "empty"):
        # Plot all runs
        os.system("mkdir -pv "+plotsDir)
        os.system("cd "+self.CMSSW_Version+"/src; eval `scramv1 r -sh`; cd "+plotsDir+"; "+self.DetName+"HDQMInspector "+self.Database+" "+self.TagName+" "+self.Password+" "+selectionFileName+" 1 2000000")
        self.convertAndMove(plotsDir, selectionType+"/AllRuns")
        # Plot the last 40 runs
        os.system("mkdir -pv "+plotsDir)
        os.system("cd "+self.CMSSW_Version+"/src; eval `scramv1 r -sh`; cd "+plotsDir+"; "+self.DetName+"HDQMInspector "+self.Database+" "+self.TagName+" "+self.Password+" "+selectionFileName+" 40")
        self.convertAndMove(plotsDir, selectionType+"/Last40Runs")

    def convertAndMove(self, plotsDir, type):
        # print "cd "+plotsDir+"; cp "+self.Dir+"/DeanConvert.pl .; ./DeanConvert.pl; rm -f DeanConvert.pl"
        # print "cp "+self.Dir+"/html/"+self.DetName+"HDQMInspector.html "+plotsDir+"/index.html"
        # print "rm -rf "+self.StorageDir+"backup/"+self.RunType+"/"+type
        # print "mkdir -pv "+self.StorageDir+"backup/"+self.RunType+"/"+type
        # print "mv "+destinationDir+"/Plots_"+self.DetName+"HistoricInfoClient "+self.StorageDir+"backup/"+self.RunType+"/"+type
        # print "mkdir -pv "+destinationDir
        # print "mv "+plotsDir+" "+destinationDir
        # print "cp "+self.Dir+"/index.html "+destinationDir
        destinationDir = self.StorageDir+"/"+self.RunType+"/"+type
        # Create the small images and copy the html for the expert plots
        os.system("cd "+plotsDir+"; cp "+self.Dir+"/DeanConvert.pl .; ./DeanConvert.pl; rm -f DeanConvert.pl")
        os.system("cp "+self.Dir+"/html/"+self.DetName+"HDQMInspector.html "+plotsDir+"/index.html")
        # Backup the old files
        backupDir = self.StorageDir+"backup/"+self.RunType+"/"+type
        os.system("rm -rf "+backupDir)
        os.system("mkdir -pv "+backupDir)
        os.system("mv "+destinationDir+"/Plots_"+self.DetName+"HistoricInfoClient "+backupDir)
        # Move the plots to the web area and copy the html file for the main web page
        os.system("mkdir -pv "+destinationDir)
        os.system("mv "+plotsDir+" "+destinationDir)
        os.system("cp "+self.Dir+"/index.html "+destinationDir)
