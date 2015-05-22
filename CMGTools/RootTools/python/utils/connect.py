import sys
import pprint
import os
import re
import operator
import subprocess
import re

from CMGTools.Production.cmgdbApi import CmgdbApi
from CMGTools.RootTools.utils.getFiles import getFiles

db = CmgdbApi()
db.connect()

def findFirstAncestor(dataset_id, info):
    cols, rows = db.sql("select parent_dataset_id, path_name, primary_dataset_entries, number_total_jobs, task_id, dataset_entries FROM dataset_details where dataset_id={dataset_id}".format(dataset_id=dataset_id))
    if len(rows)==0:
        print 'cannot find dataset with id', dataset_id
    elif len(rows)>1:
        assert(False)
    else:
        parent_id = rows[0][0]
        groups = ['tauMu_fullsel_tree_CMG', 'tauMu_fullsel_tree', 'tauEle_fullsel_tree_CMG',
                  'tauEle_fullsel_tree', 'diTau_fullsel_tree_CMG', 'diTau_fullsel_tree','cmgTuple',
                  'htt_fullsel_tree_CMG', 'htt_fullsel_tree', 'PFAOD']
        igroup = 0
        while 1:
            #import pdb ; pdb.set_trace()
            ginfo = groupInfo(dataset_id, groups[igroup])
            if ginfo != None:
                break
            igroup+=1
        file_group_name, number_files_good, number_files_bad, number_files_missing, dataset_fraction = ginfo
        dinfo=dict(
            dataset_id = dataset_id,
            parent_dataset_id = rows[0][0],
            path_name = rows[0][1],
            primary_dataset_entries = rows[0][2],
            number_total_jobs = rows[0][3],
            file_group_name = file_group_name,
            number_files_good = number_files_good,
            number_files_bad = number_files_bad,
            number_files_missing = number_files_missing,
            task_id = rows[0][4],
            dataset_entries = rows[0][5],
            dataset_fraction = dataset_fraction
            )


        # pprint.pprint(dinfo)
        info.append(dinfo)

        if parent_id is None:
            # print 'last in the DB'
            return
        findFirstAncestor( parent_id, info )


def groupInfo(dataset_id, group_name):
    cols, rows = db.sql("select file_group_name, number_files_good, number_files_bad, number_files_missing, dataset_fraction from file_group_details where dataset_id={dataset_id} and file_group_name='{group_name}'".format(
        dataset_id=dataset_id,
        group_name=group_name
        ))
    if len(rows)==0:
        return None
    elif len(rows)>1:
        raise ValueError('several dataset_id / group_name pairs found.')
    else:
        file_group_name, number_files_good, number_files_bad, number_files_missing, dataset_fraction = rows[0]
        return file_group_name, number_files_good, number_files_bad, number_files_missing, dataset_fraction



class DatasetInfo(list):
    def get(self, stepName):
        matches = [ stepi for stepi in self if stepi['step']==stepName]
        return matches

    def __str__(self):
        theStrs = [
            'primary_dataset_entries = {nentries}'.format(nentries=self.primary_dataset_entries)
            ]
        theStrs.extend( map(str, self) )
        return '\n'.join(theStrs)

reTAU = re.compile('TAU\S+')
rePatMerge = re.compile('Merge\S*')
rePatPFAOD = re.compile('V\d+')
rePatPATCMG = re.compile('PAT_CMG\S+')


def processInfo(info):
    dsInfo = DatasetInfo()
    dsInfo.primary_dataset_entries = None
    dsInfo.dataset_entries = None
    for ds in info:
        job_eff = None
        fraction = None
        skim = False
        # print ds
        pid = ds['parent_dataset_id']
        path_name = ds['path_name']
        pde = ds['primary_dataset_entries']
        njobs = ds['number_total_jobs']
        nmiss = ds['number_files_missing']
        nbad = ds['number_files_bad']
        dataset_fraction = ds['dataset_fraction']
        task_id = ds['task_id']
        # pid, path_name, pde, njobs, nmiss, nbad, dataset_fraction, task_id = ds
        # try to find the total number of entries in the CMS dataset
        if pde>0:
            if dsInfo.primary_dataset_entries is None:
                dsInfo.primary_dataset_entries=pde
            elif dsInfo.primary_dataset_entries != pde:
                print 'WARNING! there can only be one value for primary_dataset_entries in the history of a dataset, see task',task_id
        else:
            print 'WARNING! primary_dataset_entries==-1 for',path_name
        # which step is that?
        base = os.path.basename(path_name)
        fraction = dataset_fraction
        if dsInfo.dataset_entries == None:
            dsInfo.dataset_entries = ds['dataset_entries']
        if base.lower().find('tauele')!=-1 or base.lower().find('taumu')!=-1 :
            step = 'TAUTAU'
        elif rePatPFAOD.match(base):
            step = 'PFAOD'
        elif rePatPATCMG.match(base):
            step = 'PATCMG'
            # if fraction:
            #     fraction /=2.
        elif rePatMerge.match(base):
            step = 'MERGE'
        else:
            step = 'Unknown'

        try    :
          nmiss + nbad
        except :
          njobs, nbad, nmiss = retrieveInfosFromBadPublished(ds)

        if nmiss+nbad == 0:
            job_eff = 1
        else:
            job_eff = 1 - (nmiss + nbad)/float(njobs)
        # print 'job efficiency', job_eff
        if njobs and fraction :
            if job_eff - fraction > 0.1:
                # high job efficiency, but low dataset_fraction.
                # print 'skimmin'
                skim = True
        else:
            pass
            # print 'WARNING, number_total_jobs not set for', path_name, 'see savannah task', task_id
        # storing info
        dsInfo.append( dict( path_name = path_name,
                             step      = step,
                             jobeff    = job_eff,
                             fraction  = fraction,
                             skim      = skim,
                             task_id   = task_id,
                             pde       = pde
                             )
                       )
        # pprint.pprint( dsInfo[-1] )
    return dsInfo



rePatMass = re.compile('M-(\d+)_')

def findAlias(path_name, aliases):
    name = None
    for dsname, alias in aliases.iteritems():
        pat = re.compile(dsname)
        if pat.match(path_name):
            name = alias
    if name is None:
        return None
    match = rePatMass.search(path_name)
    if match and not path_name.startswith('/DY'):
        mass = match.group(1)
        return name + mass
    else:
        return name



def retrieveInfosFromBadPublished(ds) :
    print '\n'*2
    print 'WARNING!: has this dataset been published with -f option and some infos got lost? Trying to retrieve njobs informations manually'
    print ds

    num_of_files        = 0
    num_of_bad_jobs     = 0
    num_of_missing_jobs = 0 ### dummy as long as I don't figure out where tot num of jobs is stored

    ic = ''

    ### retrieve the owner of the dataset
    command = 'getInfo.py -s "select file_owner, path_name from dataset_details where path_name like \'{SAMPLE}\' order by path_name"'.format(SAMPLE = ds['path_name'])
    cmd = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    for line in cmd.stdout :
      if ds['path_name'] in line :
        fulluser = line.split('||')[0]
        fulluser = fulluser.strip(' ')

    if fulluser == 'cmgtools' :
      user  = 'cmgtools'
      group = 'user'
    elif fulluser == 'cmgtools_group' :
      user  = 'cmgtools'
      group = 'group'
    else :
      user  = fulluser
      group = 'user'

    ### list all files in the eos directory
    eos_command = '/afs/cern.ch/project/eos/installation/0.2.31/bin/eos.select'
    command = "{EOS} ls /store/cmst3/{GROUP}/{USER}/CMG{SAMPLE}".format(EOS   = eos_command,\
                                                                        GROUP = group      ,\
                                                                        USER  = user       ,\
                                                                        SAMPLE=ds['path_name'])
    cmd = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    ic_version = 0
    ### count the number of files
    for line in cmd.stdout :
      if '.root' in line :
        num_of_files += 1
      ### retrieve the IntegrityCheck file
      if 'IntegrityCheck' in line :
        ### pick the most recent IntegrityCheck (ARE TAG NUMBER INCREMENTAL?!)
        new_ic_version = re.findall(r"(\d{1,1000})", line)
        if new_ic_version > ic_version :
          ic = line.rstrip('\n')
          ic_version = new_ic_version

    ### stage the IntegrityCheck file in the user $HOME (THERE'S A WAY TO READ TEXT FILES FROM EOS?)
    cmsStage_command = '/afs/cern.ch/cms/caf/scripts/cmsStage'
    os.system("{EOS} /store/cmst3/{GROUP}/{USER}/CMG{SAMPLE}/{INT} {HOME}".format(EOS    = cmsStage_command,\
                                                                                  GROUP  = group           ,\
                                                                                  USER   = user            ,\
                                                                                  SAMPLE = ds['path_name'] ,\
                                                                                  INT    = ic              ,\
                                                                                  HOME   = os.environ['HOME']))
    icfile = file('{HOME}/{INT}'.format(HOME = os.environ['HOME'], INT  = ic))

    ### read the IntegrityCheck file and retrieve the number of bad jobs
    if icfile != '' :
      for line in icfile:
        if 'BadJobs' in line:
          num_of_bad_jobs_list = re.findall(r"(\d{1,100})", line)
          break

    if   len(num_of_bad_jobs_list)==1 :
      num_of_bad_jobs = int(num_of_bad_jobs_list[0])
    elif len(num_of_bad_jobs_list)==0 :
      pass
    else :
      print 'WARNING!: number of bad jobs in {INT} badly formatted \nimposing it to 0'.format(INT=ic)

    ### clean up the user $HOME
    os.system('rm {HOME}/{INT}'.format(HOME = os.environ['HOME'], INT  = ic) )

    ### assign sensate numbers
    njobs = num_of_files
    nbad  = num_of_bad_jobs
    nmiss = num_of_missing_jobs
    print 'got these numbers: \n njobs = %d \n nbad  = %d \n nmiss = %d' %(njobs, nbad, nmiss)
    print '\n'*2

    return njobs, nbad, nmiss




def connectSample(components, row, filePattern, aliases, cache, verbose):
    id = row[0]
    path_name = row[1]
    file_owner = row[2]
    info = []
    compName = findAlias(path_name, aliases)
    #import pdb ; pdb.set_trace()
    if compName is None:
        print 'WARNING: cannot find alias for', path_name
        return False
    findFirstAncestor(id, info)
    dsInfo = processInfo(info)
    if verbose:
        pprint.pprint( dsInfo )
    path_name = dsInfo[0]['path_name']
    globalEff = 1.
    nEvents = dsInfo.primary_dataset_entries
    taskurl = 'https://savannah.cern.ch/task/?{task_id}'.format(task_id=dsInfo[0]['task_id'])
    for step in dsInfo:
        eff = 0.
        if step['step']=='TAUTAU':
            eff = step['jobeff']
        elif step['step']=='MERGE':
            eff = step['jobeff']
        elif step['step']=='PATCMG':
            eff = step['fraction']
            if eff is None:
                eff = step['jobeff']
        elif step['step']=='PFAOD':
            eff = 1.0 # not to double count with PATCMG
        else:
            eff = step['jobeff']
        if eff is None:
            print 'WARNING: efficiency not determined for',compName
            eff = 0.0
        try:
            globalEff *= eff
        except TypeError:
            pprint.pprint(dsInfo)
            raise
    comps = [comp for comp in components if comp.name == compName]
    if len(comps)>1:
        #import pdb ; pdb.set_trace()
        print 'WARNING find several components for compName', compName
        print map(str, comps)
        return False
    elif len(comps)==0:
        print 'WARNING no component found for compName', compName
        #import pdb; pdb.set_trace()
        return False
    comp = comps[0]
    comp.dataset_entries = dsInfo.dataset_entries
    if not ( comp.name.startswith('data_') or \
             comp.name.startswith('embed_') ):
        comp.nGenEvents = nEvents
        if comp.nGenEvents is None:
            print 'WARNING: nGenEvents is None, setting it to 1.'
            comp.nGenEvents = 1.
        if comp.nGenEvents != 1.:
            comp.nGenEvents *= globalEff
        else:
            globalEff = -1.
            comp.nGenEvents = 0
    print 'LOADING:', comp.name, path_name, nEvents, globalEff, taskurl
    # print dsInfo
    comp.files = getFiles(path_name, file_owner, filePattern, cache)
    if comp.name.startswith('data_'):
        if globalEff<0.99:
            print 'ARGH! data sample is not complete.', taskurl
            print dsInfo
    else:
        if globalEff<0.9:
            print 'WEIRD! Efficiency is way too low ({globalEff})! you might have to edit your cfg manually.'.format(globalEff=globalEff)
            print dsInfo


def connect(components, samplePattern, filePattern, aliases, cache, verbose=False):
    """
    Find some information about a list of components.

    The CMGDB is searched for all datasets matching the SQL samplePattern
    (SQL patterns use % as a wildcard).

    The datasets must also match one of the patterns provided as a key in the aliases
    dictionary, which looks like this:
    aliases = {
    '/DYJetsToLL_M-50_TuneZ2Star_8TeV-madgraph-tarball.*':'DYJets',
    '/TTJets_MassiveBinDECAY_TuneZ2star_8TeV.*START53.*':'TTJets',
    '/WW_TuneZ2star_8TeV.*':'WW',
    '/WZ_TuneZ2star_8TeV.*':'WZ',
    }
    This dictionary allows to match a given CMG dataset to a component.

    * If no match is found in the aliases directory, a harmless warning is printed.
      - If you don't need the dataset, no action is needed.
      - If you need the dataset, just add an entry to your aliases dictionary,
      and call the function again.

    * If several datasets match a given pattern in the aliases directory,
    the last one will be associated to the component.
    You probably want to make sure that your patterns are accurate enough to be matched
    by a single dataset, the one you need. In the example above, we were careful
    enough to include the string 8TeV in the pattern, to be sure not to match both the
    2011 and 2012 datasets to the same component.

    For each dataset, the CMG database is used to look at the whole dataset history
    (PFAOD->PAT+CMG->Anything->Anything else), and to estimate a global computing efficiency E.
    The CMG database is also used to get the number of generated events nGen in the original
    CMS primary dataset. For the corresponding component, the attribute nGenEvents is then set
    to nGen / E.

    If the computing efficiency is too low (not equal to 1.0 for the data, or below 95% for the MC),
    a warning is issued. Take these warnings seriously, as:
       - they could be the sign of a problem in the automatic calculation of the computing efficiency
       - you probably want to use all events in a given dataset.

    Each component also gets a new attribute, dataset_entries, which is equal to the number of events in
    the dataset, as read from the CMG DB. Knowing this number of entries will allow us to guess how to
    split the component in chunks, see splitFactor in this directory.

    Finally, the dataset is used to get the list of good files in the dataset. This list is set as the
    files attribute in the corresponding component.

    Need help? contact Colin, this module is a bit tricky.
    """

    pattern = samplePattern
    cols, rows = db.sql("select dataset_id, path_name, file_owner from dataset_details where path_name like '{pattern}' order by path_name".format(
        pattern = samplePattern
        ))

#     import pdb ; pdb.set_trace()
    for row in rows:
        connectSample(components, row, filePattern, aliases, cache, verbose)



if __name__ == '__main__':
    pass
    info = []
    #findFirstAncestor(4470, info)
    # processInfo( info )
    print groupInfo(3829,'cmgTuple')
    # groupInfo(3829,'patTuple')
