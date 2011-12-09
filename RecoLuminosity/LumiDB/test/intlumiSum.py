#!/usr/bin/env python
import os,sys,csv
from RecoLuminosity.LumiDB import argparse,idDealer,nameDealer,CommonUtil,dbUtil,sessionManager
def insertIntglumiData(dbsession,intlumitorun,bulksize=1000):
    '''
    input intlumitorun [(runnumber,intglumiub)]
    insert into INTGLUMI(intglumi_id,runnum,startrun,intglumi) values()
    '''
    if not intlumitorun or len(intlumitorun)==0:
        return
    dataDef=[]
    dataDef.append(('RUNNUM','unsigned int'))
    dataDef.append(('STARTRUN','unsigned int'))
    dataDef.append(('INTGLUMI','float'))
    nrows=0
    committedrows=0
    perrunData=[]
    startrun=min([x[0] for x in intlumitorun])
    try:
        for (runnum,intglumi) in intlumitorun:
            nrows+=1
            committedrows+=1
            perrunData.append([('RUNNUM',runnum),('STARTRUN',startrun),('INTGLUMI',intglumi)])
            if nrows==bulksize:
                print 'committing ',nrows,' rows'
                db=dbUtil.dbUtil(dbsession.nominalSchema())
                dbsession.transaction().start(False)
                db.bulkInsert('INTGLUMI',dataDef,perrunData)
                dbsession.transaction().commit()
                nrows=0
                perrunData=[]
            elif committedrows==len(intlumitorun):
                print 'committing trg at the end '
                dbsession.transaction().start(False)
                db=dbUtil.dbUtil(dbsession.nominalSchema())
                db.bulkInsert('INTGLUMI',dataDef,perrunData)
                dbsession.transaction().commit()            
    except Exception, e:
        dbsession.transaction().rollback()
        del dbsession
        raise Exception, 'insertIntglumiData: '+str(e)

def parselumifile(ifilename):
    '''
    input:filename
    output: [(runnumber,delivered)...]
    '''
    result=[]
    try:
        csvfile=open(ifilename,'rb')
        reader=csv.reader(csvfile,delimiter=',',skipinitialspace=True)
        for row in reader:
            runnumber=row[0]
            try:
                runnumber=int(runnumber)
            except ValueError:
                continue
            delivered=float(row[2])
            result.append((runnumber,delivered))
    except Exception,e:
        raise RuntimeError(str(e))
    return result

def lumiuptorun(irunlumi):
    '''
    input: [(runnumber,delivered),...]
    output:[(runnumber,lumisofar),...]
    '''
    intlumiuptorun=[]
    for i,(runnumber,lumival) in enumerate(irunlumi):
        lumivals=[x[1] for x in irunlumi]
        intlumisofar=sum(lumivals[0:i+1])
        intlumiuptorun.append((runnumber,intlumisofar))
    return intlumiuptorun

if __name__=='__main__':
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),description = "intlumiSum",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parse arguments
    parser.add_argument('-c',dest='connect',action='store',required=False,help='connect string to lumiDB',default='oracle://cms_orcon_prod/cms_lumi_prod')
    parser.add_argument('-P',dest='authpath',action='store',required=False,help='authentication.xml dir',default='/afs/cern.ch/cms/lumi')
    parser.add_argument('-i',dest='inputfile',action='store',required=False,help='input file full name',default='/afs/cern.ch/cms/lumi/pp7TeVstable-2011delivered-zerocorrection.csv')
    parser.add_argument('--dryrun',dest='dryrun',action='store_true',
                        help='only print pasing result')
    parser.add_argument('--debug',dest='debug',action='store_true',
                        help='debug mode')
    
    options=parser.parse_args()
    ifilename=options.inputfile
    irunlumimap= parselumifile(ifilename)
    intlumitorun=lumiuptorun(irunlumimap)
    if options.dryrun:
        print intlumitorun
        exit(0)
    if options.authpath:
        os.environ['CORAL_AUTH_PATH'] = options.authpath
    
    svc=sessionManager.sessionManager(options.connect,authpath=options.authpath,debugON=options.debug)
    session=svc.openSession(isReadOnly=False,cpp2sqltype=[('unsigned int','NUMBER(10)'),('unsigned long long','NUMBER(20)')])
    insertIntglumiData(session,intlumitorun,bulksize=1000)        
    del session
    del svc 

    
