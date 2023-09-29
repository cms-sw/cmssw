from __future__ import print_function

import os
import shutil
import sys
import time
import glob
import importlib
import logging
import subprocess

# as we need to load the shared lib from here, make sure it's in our path:
if os.path.join( os.environ['CMSSW_BASE'], 'src') not in sys.path:
    sys.path.append( os.path.join( os.environ['CMSSW_BASE'], 'src') )

# -------------------------------------------------------------------------------------------------------

payload2xmlCodeTemplate = """

#include "CondCore/Utilities/interface/Payload2XMLModule.h"
#include "CondCore/Utilities/src/CondFormats.h"

PAYLOAD_2XML_MODULE( %s ){
  PAYLOAD_2XML_CLASS( %s );
}

""" 

buildFileTemplate = """
<flags CXXFLAGS="-Wno-sign-compare -Wno-unused-variable -Os"/>
<library   file="%s" name="%s">
  <use   name="CondCore/Utilities"/>
  <use name="py3-pybind11"/>
  <use name="python3"/>
</library>
<export>
  <lib   name="1"/>
</export>
"""

# helper function
def sanitize(typeName):
    return typeName.replace(' ','').replace('<','_').replace('>','')

def localLibName( payloadType ):
    # required to avoid ( unlikely ) clashes between lib names from templates and lib names from classes
    prefix = ''
    if '<' in payloadType and '>' in payloadType:
        prefix = 't'
    ptype = payloadType
    if '::' in payloadType:
        ptype = payloadType.replace('::','_')
    return "%s_%spayload2xml" %(sanitize(ptype),prefix)

def boost_version_for_this_release():
    import pluginUtilities_payload2xml as mod2XML
    return mod2XML.boost_version_label()

class CondXmlProcessor(object):

    def __init__(self, condDBIn):
        self.conddb = condDBIn

        if not os.path.exists( os.path.join( os.environ['CMSSW_BASE'], 'src') ):
            raise Exception("Looks like you are not running in a CMSSW developer area, $CMSSW_BASE/src/ does not exist")

        self.fakePkgName = "fakeSubSys4pl/fakePkg4pl"
        self._pl2xml_tmpDir = os.path.join( os.environ['CMSSW_BASE'], 'src', self.fakePkgName )

        self.doCleanup = False

    def __del__(self):

        if self.doCleanup: 
            shutil.rmtree( '/'.join( self._pl2xml_tmpDir.split('/')[:-1] ) )
        return 

    def discover(self, payloadType):
        libName = 'pluginUtilities_payload2xml.so'

        isReadOnlyRel = (os.environ['CMSSW_RELEASE_BASE'] == '')
        if isReadOnlyRel:
            logging.debug('Looks like the current working environment is a read-only release')

        # first search CMSSW_BASE (developer area), then CMSSW_RELEASE_BASE (release base),
        # and finally CMSSW_FULL_RELEASE_BASE (full release base, defined only for patch releases)
        foundLib = False
        for cmsswBase in ['CMSSW_BASE', 'CMSSW_RELEASE_BASE', 'CMSSW_FULL_RELEASE_BASE']:
            # Skip to next in case one is not defined or is empty
            if not (cmsswBase in os.environ and os.environ[cmsswBase] != ''):
                continue
            libDir = os.path.join( os.environ[cmsswBase], 'lib', os.environ['SCRAM_ARCH'] )
            libPath = os.path.join( libDir, libName )
            if cmsswBase == 'CMSSW_BASE':
                devLibDir = libDir
            foundLib = os.path.isfile( libPath )
            if foundLib:
                logging.debug('Found built-in library with XML converters: %s' %libPath)
                break
        if not foundLib:
            # this should never happen !!
            raise Exception('No built-in library found with XML converters (library name: %s).' %libName)

        logging.debug('Importing built-in library %s' %libPath)
        module = importlib.import_module( libName.replace('.so', '') )
        functors = dir(module)
        funcName = payloadType+'2xml'
        if funcName in functors:
            logging.info('XML converter for payload class %s found in the built-in library.' %payloadType)
            return getattr( module, funcName)
        if isReadOnlyRel:
            # give up if it is a read-only release...
            raise Exception('No XML converter suitable for payload class %s has been found in the built-in library.' %payloadType)
        libName = 'plugin%s.so' %localLibName( payloadType )
        libPath = os.path.join( devLibDir, libName )
        if os.path.exists( libPath ):
            logging.info('Found local library with XML converter for class %s' %payloadType)
            module = importlib.import_module( libName.replace('.so', '') )
            return getattr( module, funcName)
        logging.warning('No XML converter for payload class %s found in the built-in library.' %payloadType)
        return None

    def prepPayload2xml(self, payloadType):

        converter = self.discover(payloadType)
        if converter: return converter

        #otherwise, go for the code generation in the local checkout area.
        startTime = time.time()

        libName = localLibName( payloadType )
        pluginName = 'plugin%s' % libName
        tmpLibName = "Tmp_payload2xml"
        tmpPluginName = 'plugin%s' %tmpLibName

        libDir = os.path.join( os.environ["CMSSW_BASE"], 'lib', os.environ["SCRAM_ARCH"] )
        tmpLibFile = os.path.join( libDir,tmpPluginName+'.so' )
        code = payload2xmlCodeTemplate %(pluginName,payloadType) 

        tmpSrcFileName = 'Local_2XML.cpp' 
        tmpDir = self._pl2xml_tmpDir
        if ( os.path.exists( tmpDir ) ) :
            msg = '\nERROR: %s already exists, please remove if you did not create that manually !!' % tmpDir
            raise Exception(msg)

        logging.debug('Creating temporary package %s' %self._pl2xml_tmpDir)
        os.makedirs( tmpDir+'/plugins' )

        buildFileName = "%s/plugins/BuildFile.xml" % (tmpDir,)
        with open(buildFileName, 'w') as buildFile:
            buildFile.write( buildFileTemplate %(tmpSrcFileName,tmpLibName) )
            buildFile.close()

        tmpSrcFilePath = "%s/plugins/%s" % (tmpDir, tmpSrcFileName,)
        with open(tmpSrcFilePath, 'w') as codeFile:
            codeFile.write(code)
            codeFile.close()

        cmd = "source $CMS_PATH/cmsset_default.sh;"
        cmd += "(cd %s ; scram b 2>&1 >build.log)" %tmpDir
        pipe = subprocess.Popen( cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT )
        out, err = pipe.communicate()
        ret = pipe.returncode

        buildTime = time.time()-startTime
        logging.info("Building done in %s sec., return code from build: %s" %(buildTime,ret) )

        if (ret != 0):
            logging.error("Local build for xml dump failed.")
            return None

        libFile = os.path.join(libDir,pluginName + '.so')
        shutil.copyfile(tmpLibFile,libFile)

        module =  importlib.import_module( pluginName )
        funcName = payloadType+'2xml'
        functor = getattr( module, funcName ) 
        self.doCleanup = True
        return functor

    def payload2xml(self, session, payloadHash, destFile):

        Payload = session.get_dbtype(self.conddb.Payload)
        # get payload from DB:
        result = session.query(Payload.data, Payload.object_type).filter(Payload.hash == payloadHash).one()
        data, plType = result
        logging.info('Found payload of type %s' %plType)

        convFuncName = sanitize(plType)+'2xml'
        xmlConverter = self.prepPayload2xml(plType)

        if xmlConverter is not None:
            obj = xmlConverter()
            resultXML = obj.write( data )
            if destFile is None:
                print(resultXML)   
            else:
                with open(destFile, 'a') as outFile:
                    outFile.write(resultXML)
                    
