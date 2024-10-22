import os
import sys
import calendar
import optparse
import importlib
import sqlalchemy
import subprocess
import CondCore.Utilities.conddblib as conddb

##############################################
def execme(command, dryrun=False):
##############################################
    """This function executes `command` and returns it output.
    Arguments:
    - `command`: Shell command to be invoked by this function.
    """
    if dryrun:
        print(command)
        return None
    else:
        child = os.popen(command)
        data = child.read()
        err = child.close()
        if err:
            raise Exception('%s failed w/ exit code %d' % (command, err))
        return data

##############################################
def main():
##############################################

    defaultGT='auto:run3_data_prompt'
    #defaultGT='123X_dataRun3_Prompt_v5'
    defaultRun=346512
    
    parser = optparse.OptionParser(usage = 'Usage: %prog [options] <file> [<file> ...]\n')
    
    parser.add_option('-G', '--inputGT',
                      dest = 'inputGT',
                      default = defaultGT,
                      help = 'Global Tag to get conditions',
                      )

    parser.add_option('-r', '--inputRun',
                      dest = 'inputRun',
                      default = defaultRun,
                      help = 'run to be used',
                      )

    (options, arguments) = parser.parse_args()

    print("Input configuration")
    print("globalTag: ",options.inputGT)
    print("runNumber: ",options.inputRun)
    
    con = conddb.connect(url = conddb.make_url())
    session = con.session()
    RunInfo = session.get_dbtype(conddb.RunInfo)
    
    bestRun = session.query(RunInfo.run_number,RunInfo.start_time, RunInfo.end_time).filter(RunInfo.run_number >= options.inputRun).first()
    if bestRun is None:
        raise Exception("Run %s can't be matched with an existing run in the database." %options.runNumber)
    
    start= bestRun[1]
    stop = bestRun[2]
    
    bestRunStartTime = calendar.timegm( bestRun[1].utctimetuple() ) << 32
    bestRunStopTime  = calendar.timegm( bestRun[2].utctimetuple() ) << 32
    
    print("run start time:",start,"(",bestRunStartTime,")")
    print("run stop time: ",stop,"(",bestRunStopTime,")")
    
    gtstring=str(options.inputGT)
    if('auto' in gtstring):
        from Configuration.AlCa import autoCond
        key=gtstring.replace('auto:','')
        print("An autoCond Global Tag was used, will use key %s" % key)
        gtstring=autoCond.autoCond[key]
        print("Will use the resolved key %s" % gtstring)

    command='cmsRun $CMSSW_BASE/src/CondTools/SiStrip/test/db_tree_dump.py outputRootFile=sistrip_db_tree_'+gtstring+'_'+str(options.inputRun)+'.root GlobalTag='+options.inputGT+' runNumber='+str(options.inputRun)+' runStartTime='+str(bestRunStartTime)
    
    data = execme(command)
    print("\n output of execution: \n\n",data)

if __name__ == "__main__":        
    main()
