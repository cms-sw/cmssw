import os

class PopulateDB:
    def run(self):
        os.system("cat "+self.TemplatesDir+"/template_"+self.DetName+"HistoryDQMService_cfg.py | sed -e \"s@RUNNUMBER@"+self.RunNumber+"@g\" -e \"s@FILENAME@"+self.FileName+"@\" -e \"s@TAGNAME@"+self.TagName+"@g\" -e \"s@DATABASE@"+self.Database+"@\" -e \"s@AUTHENTICATIONPATH@"+self.AuthenticationPath+"@\" > "+self.Dir+"Run_"+self.DetName+"_"+self.RunNumber+".py")
        print "cd "+self.CMSSW_Version+"; eval `scramv1 r -sh`; cd "+self.Dir+"; cmsRun "+self.Dir+"Run_"+self.DetName+"_"+self.RunNumber+".py > "+self.Dir+"Run_"+self.DetName+"_"+self.RunNumber+".log"
        os.system("cd "+self.CMSSW_Version+"; eval `scramv1 r -sh`; cd "+self.Dir+"; cmsRun "+self.Dir+"Run_"+self.DetName+"_"+self.RunNumber+".py > "+self.Dir+"Run_"+self.DetName+"_"+self.RunNumber+".log")
