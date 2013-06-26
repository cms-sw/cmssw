import os

class SelectRuns:

    BaseDir = "/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/HDQM/Cron/Scripts/RunSelection/Test"
    Group = "Collisions10"
    # Group = "Cosmics10"
    FirstRun = "1"
    FileName = BaseDir+"/SelectedGoodRuns.txt"
    HLTNameFilter = ""
    QualityFlag = "Strip:GOOD"

    def makeList(self):
        # Save the old run list file
        os.system("mv "+self.FileName+" "+self.FileName+".old")

        # Create the cfg for the run registry script
        inputFile = open(self.BaseDir+"/runreg_template.cfg", "r")
        # print "OutputFileName = ", self.FileName
        outputFileContent = inputFile.read().replace("GROUP", self.Group).replace("FIRSTRUN", self.FirstRun).replace("OUTPUTFILENAME", self.FileName).replace("HLTNAMEFILTER", self.HLTNameFilter).replace("QUALITYFLAG", self.QualityFlag)
        outputFile = open(self.BaseDir+"/runreg.cfg", "w")
        outputFile.write(outputFileContent)
        outputFile.close()

        # Produce the new run list file
        os.system("source /afs/cern.ch/cms/ccs/wm/scripts/Crab/crab.sh; python "+self.BaseDir+"/runregparse.py")

        # Check if the file changed
        import filecmp
        if os.path.isfile(self.FileName+".old"):
            if filecmp.cmp(self.FileName, self.FileName+".old"):
                # They are equal
                return 1
            return 0
