import commands
import re
castorlist=[]
runlist=[136035,136066,136088,136119,137028,139020,139375,139411,139459,139790,140126,140331,140362,140401,141961,142040,142076,142137,142191,142265,142422,142514,142537,142558,142664,142936,143193,143665,143731,143835,143962,144011,144089,144114,146514,146589,146644,146807,146944,147043,147048,147116,147222,147284,147390, 147454, 147757, 147929, 148002, 148032,148058,148864,148953]
for month in['05','06','07','08','09','10']:
    dirname='/castor/cern.ch/cms/store/lumi/2010'+month
    castorresult=commands.getoutput('nsls '+dirname)
    filenames=castorresult.split('\n')
    filenames=[dirname+'/'+i for i in filenames]
    castorlist.extend(filenames)
#print castorlist
matchlist=[]
p=r'/castor/cern.ch/cms/store/lumi/2010\d{2}/CMS_LUMI_RAW_2010\d{4}_000(\d{6})_0001_\d{1}.root'
for run in runlist:
    for filename in castorlist:
        m=re.search(p,filename)
        if m.group(1) ==str(run):
            matchlist.append(filename)
            break
print matchlist

