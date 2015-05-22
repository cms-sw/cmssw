#!/usr/bin/env python
# Colin
# creates new source file for a dataset on castor
# compiles the python module
# prints the line to be added to the cfg. 

import os, sys, re

def doImportNewSource( sampleName, sourceFileListCmd, fileName = 'source_cff.py'):
    # making local source directory ---------
    
    tmp = './python/sources' + sampleName
    ldir = re.sub( '-', '_', tmp)
    mkdir = 'mkdir -p ' + ldir
    print mkdir
    os.system( mkdir )
    
    # creating source file ------------------
    
    sourceFile = ldir + '/' + fileName
    
    if os.path.isfile( sourceFile ):
        print sourceFile
        print 'already exists. define another output file name'
        sys.exit(1)

    sourceFileList = sourceFileListCmd + ' > ' + sourceFile
    print sourceFileList
    os.system(sourceFileList)

    # compile new source file
    os.system( 'scram b python')

    # printing module load command ----------
    base = os.environ['CMSSW_BASE']
    cwd = os.getcwd()
    cwd = re.sub('%s/src/' % base, '', cwd)
    # cwd now equal to package name

    # replace ./python by package name 
    module = re.sub( './python', cwd, sourceFile)
    # replace / by .
    module = re.sub( '/', '.', module)
    # remove trailing .py
    module = re.sub( '\.py$', '', module)

    os.system( 'cat %s' % sourceFile )

    print 'new source file ready to be used:'
    print sourceFile
    print 'process.load("%s")' % module
  
