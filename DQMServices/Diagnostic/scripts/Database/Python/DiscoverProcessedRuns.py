import os

class DiscoverProcessedRuns:
    def runsList(self):
        print "cd "+self.CMSSW_Version+"; eval `scramv1 r -sh`; cd -; cmscond_list_iov -c "+self.Database+" -P "+self.AuthenticationPath+" -t "+self.TagName
        fullList = os.popen("source /afs/cern.ch/cms/sw/cmsset_default.sh; cd "+self.CMSSW_Version+"/src; eval `scramv1 r -sh`; cd -; cmscond_list_iov -c "+self.Database+" -P "+self.AuthenticationPath+" -t "+self.TagName).read()
        runsListInTag = list()
        # print fullList
        for line in fullList.split("\n"):
            # print line
            if( line.find("=HDQMSummary") != -1 and not line.startswith("1 ")):
                # print line.split()[0]
                runsListInTag.append(line.split()[0])
        return runsListInTag
