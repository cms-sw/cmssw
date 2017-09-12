__author__ = 'Giacomo Govi'

import subprocess
import json
import os
import re



def check_cmssw_version( cmssw_version ):
    if cmssw_version.find('CMSSW_')==-1:
        raise Exception("CMSSW version '%s' has not been found in the expected format" %cmssw_version)
    ver = cmssw_version
    p = re.compile('(CMSSW_.*_X)(_.*)?$')
    m = p.match(ver)
    if not m is None:
        gs = m.groups()
        if gs[1] != None:
            raise Exception("IB version can't be processed.")
    return ver

def is_release_cycle( cmssw_version ):
    ind = cmssw_version.find('_X')
    return not ind == -1

def strip_cmssw_version( cmssw_version ):
    ver = cmssw_version
    if not cmssw_version.find('_X') == -1: 
        ver = ver.replace('X','0')
    ind = ver.find('_pre')
    if not ind == -1:
        ver = ver[0:ind]
    return ver    

def cmssw_version_to_int( cmssw_version ):
    ver = cmssw_version.split('CMSSW_')[1]
    ip = 0
    f = ver.find('_patch')
    if not f == -1:
        ip = int(ver[f+6:])
        ver = ver[0:f]
    dgs = ver.split('_')
    return int('%d%02d%02d%02d' %(int(dgs[0]),int(dgs[1]),int(dgs[2]),ip))

def cmp_cmssw_version( ver1, ver2 ):
    intVer1 = cmssw_version_to_int(ver1)
    intVer2 = cmssw_version_to_int(ver2)
    if intVer1<intVer2:
        return -1
    elif intVer1>intVer2:
        return 1
    else:
        return 0

def strip_boost_version( boost_version ):
    bver = boost_version
    f = bver.find('_')
    if not f==-1:
        bver = boost_version.replace('_','.')
    f = bver.find('-')
    if not f==-1:
        bver = bver[0:f]
    return bver    

def boost_version_to_int( boost_version ):
    bver = strip_boost_version( boost_version )
    dgs = bver.split('.')    
    tmpl = '%d%02d%02d'
    pars = [0,0,0]
    ind = 0
    for d in dgs:
        if ind <=2:
            pars[ind] = int(d)
        ind += 1
    return int(tmpl %(pars[0],pars[1],pars[2]))

def cmp_boost_version( ver1, ver2 ):
    intVer1 = boost_version_to_int(ver1)
    intVer2 = boost_version_to_int(ver2)
    if intVer1<intVer2:
        return -1
    elif intVer1>intVer2:
        return 1
    else:
        return 0

archs = { 3070000 : ['slc5_amd64_gcc434'], 5010100 : ['slc5_amd64_gcc462'], 6000000 : ['slc6_amd64_gcc472'], 
          7000000 : ['slc6_amd64_gcc481'], 7020000 : ['slc6_amd64_gcc491'], 7060000 : ['slc6_amd64_gcc493'],
          8000000 : ['slc6_amd64_gcc493','slc6_amd64_gcc530'], 9030000 : ['slc6_amd64_gcc630']}

cmsPath = '/cvmfs/cms.cern.ch'

def get_production_arch( version ):
    nv = cmssw_version_to_int( version )
    rbound = None
    for r in sorted(archs.keys()):
        if r <= nv:
            rbound = r
        else:
            break
    if not rbound is None:
        return archs[rbound]
    return None

def get_release_root( cmssw_version, arch ):
    cmssw_folder = 'cmssw'
    if cmssw_version.find('_patch') != -1:
        cmssw_folder= 'cmssw-patch'
    ret = '%s/%s/cms/%s' %(cmsPath,arch,cmssw_folder)
    return ret 

def get_cmssw_boost( arch, releaseDir ):
    cmd =  'source %s/cmsset_default.sh; export SCRAM_ARCH=%s; cd %s/src ; eval `scram runtime -sh`; scram tool info boost; ' %(cmsPath,arch,releaseDir)
    pipe = subprocess.Popen( cmd, shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    out = pipe.communicate()[0]
    cfgLines = out.split('\n')
    boost_version = None
    for line in cfgLines:
        if line.startswith('Version'):
            boost_version = line.split('Version :')[1].strip()
            break
    if not boost_version is None:
        boost_version = strip_boost_version(boost_version)
    return boost_version

def lookup_boost_for_run( iov, timeType, boost_run_map ):
    if timeType == 'Lumi':
        iov = iov >> 32
        timeType = 'Run'
    if iov == 0:
        iov=1
    entry = None
    try:
        if timeType == 'Time':
            entry = max([x for x in boost_run_map if x[1]<=iov])
        elif timeType == 'Run':
            entry = max([x for x in boost_run_map if x[0]<=iov])
        else:
            raise Exception('TimeType %s cannot be processed' %timeType)
    except Exception as e:
        raise e
    return entry[2]

def get_boost_version_from_streamer_info( streamer_info ):
    streamer_info = streamer_info.replace('\x00','')
    iovBoostVersion = None
    if streamer_info == '0':
        iovBoostVersion = '1.51.0'
    elif streamer_info[0:2]==' {':
        try:
            iovBoostVersion = str(json.loads(streamer_info)['tech_version'])
            iovBoostVersion = strip_boost_version(iovBoostVersion)
        except ValueError as e:
            raise Exception("Could not parse streamer info [%s]: %s" %(streamer_info,str(e)))
    else:
        raise Exception("Streamer info found in unexpected format.")
    return iovBoostVersion

def do_update_tag_boost_version( tagBoostVersion, iovBoostVersion, iov, timetype, boost_run_map ):
    # for hash timetype we need to take the greatest version of the set                                                                                                                 
    if timetype == 'Hash':
        if tagBoostVersion is None or cmp_boost_version(tagBoostVersion,iovBoostVersion)<0:
            tagBoostVersion = iovBoostVersion
    # for run, lumi and time we lookup in the boost_run_map to find the reference
    else:
        if tagBoostVersion is None:
            tagBoostVersion = iovBoostVersion
        else:
            referenceBoost = lookup_boost_for_run( iov, timetype, boost_run_map )
            if cmp_boost_version( referenceBoost, iovBoostVersion )<0:
                if cmp_boost_version(  tagBoostVersion, iovBoostVersion )<0:
                    tagBoostVersion = iovBoostVersion
            else:
                if cmp_boost_version(  tagBoostVersion, iovBoostVersion )>0:
                    tagBoostVersion = iovBoostVersion
    return tagBoostVersion

def update_tag_boost_version( tagBoostVersion, streamer_info, iov, timetype, boost_run_map ):
    iovBoostVersion = get_boost_version_from_streamer_info( streamer_info )
    return iovBoostVersion, do_update_tag_boost_version( tagBoostVersion, iovBoostVersion, iov, timetype, boost_run_map )

            
