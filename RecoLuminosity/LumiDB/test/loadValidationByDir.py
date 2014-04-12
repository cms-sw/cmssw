#! /usr/bin/python
import os,os.path,sys,fnmatch,commands
dbname='oracle://cms_orcon_prod/cms_lumi_prod'
authpath='/home/lumidb/auth/writer'
def main(*args):
    listoffiles=[]
    try:
        dirname=args[1]
	dirList=os.listdir(dirname)
        for fname in dirList:
            if fnmatch.fnmatch(fname,'Status*.csv'):
               listoffiles.append(fname)
	for fname in listoffiles:
            command=' '.join(['lumiValidate.py','-c',dbname,'-P',authpath,'-i',os.path.join(dirname,fname),'batchupdate'])
            print command	
	    statusAndOutput=commands.getstatusoutput(command)
            if not statusAndOutput[0]==0:
               print 'Error when uploading ',fname
               print statusAndOutput[1]
            else:
               print 'Done'
    except IndexError:
        print 'dir name should be provided'
        return 1
    except Exception,er:
        print str(er)
        return 2
    else:
        return 0
if __name__=='__main__':
    sys.exit(main(*sys.argv))    
