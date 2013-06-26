import csv,os,sys,coral
from RecoLuminosity.LumiDB import dbUtil
filename='fillsummary.dat'
conn='oracle://cms_orcoff_prep/cms_lumi_dev_offline'

def parseFillFile(filename):
    result=[]
    f=open(filename)
    csvReader=csv.reader(f,delimiter=',')
    for row in csvReader:
        if len(row)<3: continue
        result.append([int(row[0]),row[1],int(row[2])])
    return result
def updateLumiSummarydata(dbsession,fillinfo):
    '''
    input: fillinfo [[fillnum,fillscheme,ncollidingbunches],[fillnum,fillscheme,ncollidingbunches]]
    update lumisummary set FILLSCHEME=:fillscheme, NCOLLIDINGBUNCHES=:ncollidingbunches where
    FILLNUM=:fillnum
    '''
    dbsession.transaction().start(False)
    db=dbUtil.dbUtil(dbsession.nominalSchema())
    updateAction='FILLSCHEME=:fillscheme,NCOLLIDINGBUNCHES=:ncollidingbunches'
    updateCondition='FILLNUM=:fillnum'
    bindvarDef=[('fillscheme','string'),('ncollidingbunches','unsigned int'),('fillnum','unsigned int')]
    bulkinput=[]
    for fillline in fillinfo:
        bulkinput.append([('fillscheme',fillline[1]),('ncollidingbunches',fillline[2]),('fillnum',fillline[0])])
    db.updateRows('CMSRUNSUMMARY',updateAction,updateCondition,bindvarDef,bulkinput)
    dbsession.transaction().commit()
def main():
    fillinfo=parseFillFile(filename)
    #print fillinfo
    msg=coral.MessageStream('')
    msg.setMsgVerbosity(coral.message_Level_Debug)
    os.environ['CORAL_AUTH_PATH']='/afs/cern.ch/user/x/xiezhen'
    svc = coral.ConnectionService()
    dbsession=svc.connect(conn,accessMode=coral.access_Update)
    updateLumiSummarydata(dbsession,fillinfo)
    del dbsession
    del svc
if __name__=='__main__':
    sys.exit(main())
