import os,commands,re,urllib2
class checkforupdate:
    def __init__(self,statusfilename='tagstatus.txt'):
        self.lumiurl='http://cms-service-lumi.web.cern.ch/cms-service-lumi/'+statusfilename
    def fetchTagsHTTP(self):
        taglist=[]
        openurl=urllib2.build_opener()
        tagfile=openurl.open(self.lumiurl)
        tagfileStr=tagfile.read()
        for tag in tagfileStr.lstrip('\n').strip('\n').split('\n'):
            fields=tag.split(',')
            taglist.append([fields[0],fields[1],fields[2]])
        return taglist
    def runningVersion(self,cmsswWorkingBase,scriptname,isverbose=True):
        currentdir=os.getcwd()
        os.chdir(os.path.join(cmsswWorkingBase,'src','RecoLuminosity','LumiDB','scripts'))
        cvscmmd='cvs status '+scriptname
        (cmmdstatus,result)=commands.getstatusoutput(cvscmmd+'| grep "Sticky Tag:"')
        os.chdir(currentdir)
        cleanresult=result.lstrip().strip()
        cleanresult=re.sub(r'\s+|\t+',' ',cleanresult)
        allfields=cleanresult.split(' ')
        workingversion = "n/a"
        for line in filter(lambda line: "Sticky Tag" in line, result.split('\n')):
            workingversion = line.split()[2]
        if workingversion=='(none)':
            workingversion='HEAD'
        if isverbose:
            print 'checking current version......'
            print '  project base : '+cmsswWorkingBase
            print '  script : '+scriptname
            print '  version : '+workingversion
        return workingversion
    def checkforupdate(self,workingtag,isverbose=True):
        newtags=self.fetchTagsHTTP()
        if workingtag=='(none)':#means HEAD
            if isverbose:
                print 'checking update for HEAD'
                print '  no update'
            return []
        w=workingtag.lstrip('V').split('-')
        if len(w)!=3:
            #print workingtag+' is not a release tag, can not compare'
            return []
        w=[int(r) for r in w]
        updatetags=[]
        for [tagstr,ismajor,desc] in newtags:
            digits=[int(r) for r in tagstr.lstrip('V').split('-')]
            if digits[0]==w[0] and  digits[1]==w[1] and digits[2]==w[2]:
                continue
            if digits[0]<w[0]: 
                continue            
            elif digits[0]==w[0]:
                if digits[1]<w[1]:
                    continue
                elif digits[1]==w[1]:
                    if digits[2]<=w[2]:
                        continue
            updatetags.append([tagstr,ismajor,desc])
        if isverbose:
            print 'checking update for '+workingtag
            if not updatetags:
                print '  no update'
                return []
            for [tag,ismajor,description] in updatetags:
                if ismajor=='1':
                    print '  major update, tag ',tag+' , '+description
                else:
                    print '  minor update, tag ',tag+' , '+description
        return updatetags
        
if __name__=='__main__':
    scriptname='lumiCalc2.py'
    cmsswWorkingBase=os.environ['CMSSW_BASE']    
    c=checkforupdate()
    workingversion=c.runningVersion(cmsswWorkingBase,scriptname)
    #c.checkforupdate('V03-03-07')
    c.checkforupdate(workingversion)
   
    
