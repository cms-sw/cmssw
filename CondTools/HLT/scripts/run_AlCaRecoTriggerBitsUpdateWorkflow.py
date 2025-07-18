#!/usr/bin/env python3

"""
Example script to test reading from local sqlite db.
"""
import os
import sys
import ast
import optparse
import multiprocessing
import CondCore.Utilities.conddblib as conddb

officialdbs = {
    # frontier connections
    'frontier://PromptProd/CMS_CONDITIONS'   :'pro',
    'frontier://FrontierProd/CMS_CONDITIONS' :'pro',
    'frontier://FrontierArc/CMS_CONDITIONS'  :'arc',
    'frontier://FrontierInt/CMS_CONDITIONS'  :'int',
    'frontier://FrontierPrep/CMS_CONDITIONS' :'dev'
}

##############################################
def getCommandOutput(command):
##############################################
    """This function executes `command` and returns it output.
    Arguments:
    - `command`: Shell command to be invoked by this function.
    """
    print(command)
    child = os.popen(command)
    data = child.read()
    err = child.close()
    if err:
        raise Exception('%s failed w/ exit code %d' % (command, err))
    return data

##############################################
def get_iovs(db, tag):
##############################################
    """Retrieve the list of IOVs from `db` for `tag`.

    Arguments:
    - `db`: database connection string
    - `tag`: tag of database record
    """

    ### unfortunately some gymnastics is needed here
    ### to make use of the conddb library
    if db in officialdbs.keys():
        db = officialdbs[db]

    ## allow to use input sqlite files as well
    db = db.replace("sqlite_file:", "").replace("sqlite:", "")

    con = conddb.connect(url = conddb.make_url(db))
    session = con.session()
    IOV = session.get_dbtype(conddb.IOV)

    iovs = set(session.query(IOV.since).filter(IOV.tag_name == tag).all())
    if len(iovs) == 0:
        print("No IOVs found for tag '"+tag+"' in database '"+db+"'.")
        sys.exit(1)

    session.close()

    return sorted([int(item[0]) for item in iovs])

##############################################
def updateBits(blob):
##############################################
    if(blob[0]<blob[1]):
        command = 'conddb --yes --db %s copy %s --destdb %s --from %s --to %s' % (blob[4],
                                                                                  blob[5],
                                                                                  blob[5]+"_IOV_"+str(blob[2])+".db",
                                                                                  str(blob[2]) ,
                                                                                  str(blob[3]))
        getCommandOutput(command)
    else:
        # last IOV needs special command
        command = 'conddb --yes --db %s copy %s --destdb %s --from %s' % (blob[4],
                                                                          blob[5],
                                                                          blob[5]+"_IOV_"+str(blob[2])+".db",
                                                                          str(blob[2]))
        getCommandOutput(command)

    # update the trigger bits
    cmsRunCommand='cmsRun $CMSSW_BASE/src/CondTools/HLT/test/AlCaRecoTriggerBitsRcdUpdate_TEMPL_cfg.py \
    inputDB=%s inputTag=%s outputDB=%s outputTag=%s firstRun=%s' % ("sqlite_file:"+blob[5]+"_IOV_"+str(blob[2])+".db",
                                                                    blob[5],
                                                                    "sqlite_file:"+blob[5]+"_IOV_"+str(blob[2])+"_updated.db",
                                                                    blob[6],
                                                                    str(blob[2]))
    getCommandOutput(cmsRunCommand)

##############################################
def main():
##############################################
     
    defaultDB     = 'sqlite_file:mySqlite.db'
    defaultInTag  = 'myInTag'
    defaultOutTag = 'myOutTag'
    defaultProc   = 20

    parser = optparse.OptionParser(usage = 'Usage: %prog [options] <file> [<file> ...]\n')
    parser.add_option('-f', '--inputDB',
                      dest = 'inputDB',
                      default = defaultDB,
                      help = 'file to inspect')
    parser.add_option('-i', '--inputTag',
                      dest = 'InputTag',
                      default = defaultInTag,
                      help = 'tag to be inspected')
    parser.add_option('-d', '--destTag',
                      dest = 'destTag',
                      default = defaultOutTag,
                      help = 'tag to be written')
    parser.add_option('-p', '--processes',
                      dest = 'nproc',
                      default = defaultProc,
                      help = 'multiprocesses to run')
    parser.add_option("-C", '--clean',
                      dest="doTheCleaning",
                      action="store_true",
                      default = True,
                      help = 'if true remove the transient files')

    (options, arguments) = parser.parse_args()

    db_url = options.inputDB
    if db_url in officialdbs.keys():
        db_url = officialdbs[db_url]

    ## allow to use input sqlite files as well
    db_url = db_url.replace("sqlite_file:", "").replace("sqlite:", "")

    sinces = get_iovs(options.inputDB,options.InputTag)

    print("List of sinces: %s" % sinces)

    myInputTuple=[]

    for i,since in enumerate(sinces):
        if(i<len(sinces)-1):
            #                   0             1        2               3      4                5               6
            myInputTuple.append((i,len(sinces)-1,sinces[i],sinces[i+1]-1,db_url,options.InputTag,options.destTag))
        else:
            myInputTuple.append((i,len(sinces)-1,sinces[i],-1,db_url,options.InputTag,options.destTag))

    pool = multiprocessing.Pool(processes=options.nproc)  # start nproc worker processes
    count = pool.map(updateBits,myInputTuple)

    # merge the output
    for i,since in enumerate(sinces):
        mergeCommand='conddb --yes --db %s copy %s --destdb %s --from %s' % (options.InputTag+"_IOV_"+str(sinces[i])+"_updated.db",
                                                                             options.destTag,
                                                                             options.destTag+".db",
                                                                             str(sinces[i]))
        getCommandOutput(mergeCommand)

    # clean the house (after all is done)
    if(options.doTheCleaning):
        cleanCommand = 'rm -fr *updated*.db *IOV_*.db'
        getCommandOutput(cleanCommand)
    else:
        print("======> keeping the transient files")
          
if __name__ == "__main__":        
     main()
