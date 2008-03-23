#!/usr/bin/env python

import sys
import os
import getopt
import time

import FWCore.ParameterSet.Config as cms
import pickle

from xml.dom.minidom import Document, Element, Text

#######################################################################
def red(string):
    return '%s%s%s' %('\033[1;31m',string,'\033[1;0m')    
def green(string):
    return '%s%s%s' %('\033[1;32m',string,'\033[1;0m') 
def yellow(string):
    return '%s%s%s' %('\033[1;33m',string,'\033[1;0m')     
#######################################################################


def pickleParameterSet(pklname,command):
    """

    pickle parameter set

    """
    # Add the option to dump a pickle on disk
    command += ' --dump_pickle %s' %pklname
    print yellow('\nExecuting %s ..\n' %command)
    os.system(command)
    
    parameter_set=command
        
    try:
        file=open(pklname,"r")
        process=pickle.load(file)
        file.close()        
    except Exception,ex:
        print ''
        print 'ParameterSet: ',parameter_set,'could not be converted to python dictionary, error msg:',str(ex)
        print ''
        sys.exit(1)

    try:
        relValPSet = getattr(process,'ReleaseValidation')
    except:
        print 'Parameter-Set:',parameter_set,'is missing the ReleaseValidation PSet'
        print ''
        print 'Please add a PSet named ReleaseValidation defining the following parameters:'
        print ''
        print 'untracked PSet ReleaseValidation = {'
        print '  untracked uint32 totalNumberOfEvents = 1000'
        print '  untracked uint32 eventsPerJob        = 25'
        print '  untracked string primaryDatasetName  = \'RelValExample\''
        print '}'
        sys.exit(1)

    try:
        relValTotEvents = getattr(relValPSet,'totalNumberOfEvents')
    except:
        print 'Parameter-Set:',parameter_set,'is missing the ReleaseValidation PSet'
        print ''
        print 'Please add a PSet named ReleaseValidation defining the following parameters:'
        print ''
        print 'untracked PSet ReleaseValidation = {'
        print '  untracked uint32 totalNumberOfEvents = 1000'
        print '  untracked uint32 eventsPerJob        = 25'
        print '  untracked string primaryDatasetName  = \'RelValExample\''
        print '}'
        sys.exit(1)

    try:
        relValTotEvents = getattr(relValPSet,'totalNumberOfEvents')
    except:
        print 'Parameter-Set:',parameter_set,'is missing the ReleaseValidation PSet'
        print ''
        print 'Please add a PSet named ReleaseValidation defining the following parameters:'
        print ''
        print 'untracked PSet ReleaseValidation = {'
        print '  untracked uint32 totalNumberOfEvents = 1000'
        print '  untracked uint32 eventsPerJob        = 25'
        print '  untracked string primaryDatasetName  = \'RelValExample\''
        print '}'
        sys.exit(1)

    try:
        relValTotEvents = getattr(relValPSet,'eventsPerJob')
    except:
        print 'Parameter-Set:',parameter_set,'is missing the ReleaseValidation PSet'
        print ''
        print 'Please add a PSet named ReleaseValidation defining the following parameters:'
        print ''
        print 'untracked PSet ReleaseValidation = {'
        print '  untracked uint32 totalNumberOfEvents = 1000'
        print '  untracked uint32 eventsPerJob        = 25'
        print '  untracked string primaryDatasetName  = \'RelValExample\''
        print '}'
        sys.exit(1)

    try:
        relValTotEvents = getattr(relValPSet,'primaryDatasetName')
    except:
        print 'Parameter-Set:',parameter_set,'is missing the ReleaseValidation PSet'
        print ''
        print 'Please add a PSet named ReleaseValidation defining the following parameters:'
        print ''
        print 'untracked PSet ReleaseValidation = {'
        print '  untracked uint32 totalNumberOfEvents = 1000'
        print '  untracked uint32 eventsPerJob        = 25'
        print '  untracked string primaryDatasetName  = \'RelValExample\''
        print '}'
        sys.exit(1)

    # pickle parameter-set by replacing cfg extension with pkl extension
    file = open(pklname,'w')
    print yellow('\nRe-pickling on %s after modifying..\n'%pklname)
    pickle.dump(process,file)
    file.close()

def main(argv) :
    """
    
    prepareRelVal
    
    - read in RelVal sample parameter-set list from text file, each parameter-set contains RelVal PSet
    - prepare pickle files from RelVal samples
    - requires:
      - setup'ed CMSSW user project area
      - checkout of Configuration/ReleaseValidation
    - has to be executed in Configuration/ReleaseValidation/data
    - output
      - workflow to be injected into RelValInjector

    required parameters
    --samples      <textfile>   : list of RelVal sample parameter-sets in plain text file, one sample per line, # marks comment
    --cfg          <cfg>        : pickle given parameter-set, options are either --samples or --cfg
    --cms-path     <path>       : path to CMS software area
    
    optional parameters         :
    --help (-h)                 : help
    --debug (-d)                : debug statements
    
    
    """
    
    begin=time.time()
    
    # default
    try:
        version = os.environ.get("CMSSW_VERSION")
    except:
        print ''
        print 'CMSSW version cannot be determined from $CMSSW_VERSION'
        sys.exit(2)

    try:
        architecture = os.environ.get("SCRAM_ARCH")
    except:
        print ''
        print 'CMSSW architecture cannot be determined from $SCRAM_ARCH'
        sys.exit(2)

    samples      = ''
    single_cfg          = ''
    debug        = 0
    cms_path     = ''

    try:
        opts, args = getopt.getopt(argv, "", ["help", "debug", "samples=", "cms-path=", "cfg="])
    except getopt.GetoptError:
        print main.__doc__
        sys.exit(2)

    # check command line parameter
    for opt, arg in opts :
        if opt == "--help" :
            print main.__doc__
            sys.exit()
        elif opt == "--debug" :
            debug = 1
        elif opt == "--samples" :
            samples = arg
        elif opt == "--cfg" :
            single_cfg = arg
        elif opt == "--cms-path" :
            cms_path = arg

    if samples != '' and single_cfg != '' :
        print ''
        print 'Please use either --samples or --cfg !'
        print ''
        print main.__doc__
        sys.exit(2)

    if samples != '' and cms_path == '':
        print main.__doc__
        sys.exit(2)


    if samples != '' :
        # read in samples
        cfgs = []
        try:
            file = open(samples)
        except IOError:
            print 'file with list of parameter-sets cannot be opened!'
            sys.exit(1)
        for line in file.readlines():
            line = line.strip()
            if len(line) > 0 and line[0] != '#':
                basename,command=map(str.strip,line.split('@@@'))
                if basename!='No_Pickle':
                    pklname='%s.pkl'%basename
                    cfgs.append((pklname,command))

        # pickle parameter-sets
        for pklname,command in cfgs:
            print 'Store python representation of parameter-set created by:',command,'in pickle file',pklname
            pickleParameterSet(pklname,command)

        # write workflow xml
        try:
            topNode = Element("RelValSpec")

            for pklname,command in cfgs:
                element = Element("RelValTest")
                element.setAttribute("Name", pklname.replace('.pkl',''))
            
                PickleFileElement = Element("PickleFile")
                PickleFileElement.setAttribute("Value", os.path.join(os.getcwd(), pklname))
                element.appendChild(PickleFileElement)
            
                PickleFileElement = Element("CMSPath")
                PickleFileElement.setAttribute("Value", cms_path)
                element.appendChild(PickleFileElement)
            
                PickleFileElement = Element("CMSSWArchitecture")
                PickleFileElement.setAttribute("Value", architecture)
                element.appendChild(PickleFileElement)
            
                PickleFileElement = Element("CMSSWVersion")
                PickleFileElement.setAttribute("Value", version)
                element.appendChild(PickleFileElement)
            
                topNode.appendChild(element)
    
            doc = Document()
            doc.appendChild(topNode)
            handle = open("relval_workflows.xml", "w")
            handle.write(doc.toprettyxml())
            handle.close()
        except Exception,ex:
            print 'Preparation of ProdAgent workflow: relval_workflows.xml failed with message:',ex
            sys.exit(1)
        
        print ''
        print 'Prepared ProdAgent workflow: relval_workflows.xml to be injected into the RelValInjector by:'
        print 'python2.4 publish.py RelValInjector:Inject',os.path.join(os.getcwd(), 'relval_workflows.xml')

#     if single_cfg != '' :
#         pickleParameterSet(single_cfg)
#         print 'Store python representation of parameter-set:',single_cfg,'in pickle file',single_cfg.replace('.cfg','.pkl')
    
    elapsed=time.time()-begin
    print green('\nTotal Time elapsed: %s m %s s' %(int(elapsed/60.), int(elapsed)%60))    

if __name__ == '__main__' :
    main(sys.argv[1:])
