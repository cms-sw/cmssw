
import os
import shutil
import sys
import time
import glob
import importlib

# as we need to load the shared lib from here, make sure it's in our path:
if os.path.join( os.environ['CMSSW_BASE'], 'src') not in sys.path:
   sys.path.append( os.path.join( os.environ['CMSSW_BASE'], 'src') )

# -------------------------------------------------------------------------------------------------------

payload2xmlCodeTemplate = """

#include <iostream>
#include <string>
#include <memory>

#include <boost/python/class.hpp>
#include <boost/python/module.hpp>
#include <boost/python/init.hpp>
#include <boost/python/def.hpp>
#include <iostream>
#include <string>
#include <sstream>

#include "boost/archive/xml_oarchive.hpp"
#include "CondFormats/Serialization/interface/Serializable.h"
#include "CondFormats/Serialization/interface/Archive.h"

#include "CondCore/Utilities/src/CondFormats.h"

namespace { // Avoid cluttering the global namespace.

  std::string %(plTypeSan)s2xml( const std::string &payloadData, const std::string &payloadType ) { 

      // now to convert
      std::unique_ptr< %(plType)s > payload;

      std::stringbuf sdataBuf;
      sdataBuf.pubsetbuf( const_cast<char *> ( payloadData.c_str() ), payloadData.size() );

      std::istream inBuffer( &sdataBuf );
      eos::portable_iarchive ia( inBuffer );
      payload.reset( new %(plType)s );
      ia >> (*payload);

      // now we have the object in memory, convert it to xml in a string and return it
     
      std::ostringstream outBuffer;
      boost::archive::xml_oarchive xmlResult( outBuffer );
      xmlResult << boost::serialization::make_nvp( "cmsCondPayload", *payload );

      return outBuffer.str();
  }

} // end namespace


BOOST_PYTHON_MODULE(%(mdName)s)
{
    using namespace boost::python;
    def ("%(plTypeSan)s2xml", %(plTypeSan)s2xml);
}

""" 

buildFileTemplate = """
<flags CXXFLAGS="-Wno-sign-compare -Wno-unused-variable -Os"/>
<use   name="boost"/>
<use   name="boost_python"/>
<use   name="boost_iostreams"/>
<use   name="boost_serialization"/>
<use   name="boost_program_options"/>
<use   name="CondCore/DBCommon"/>
<use   name="CondCore/IOVService"/>
<use   name="CondCore/MetaDataService"/>
<use   name="CondCore/TagCollection"/>
<use   name="CondCore/CondDB"/>
<use   name="CondFormats/HLTObjects"/>
<use   name="CondFormats/Alignment"/>
<use   name="CondFormats/BeamSpotObjects"/>
<use   name="CondFormats/CastorObjects"/>
<use   name="CondFormats/HIObjects"/>
<use   name="CondFormats/CSCObjects"/>
<use   name="CondFormats/DTObjects"/>
<use   name="CondFormats/ESObjects"/>
<use   name="CondFormats/EcalObjects"/>
<use   name="CondFormats/EgammaObjects"/>
<use   name="CondFormats/Luminosity"/>
<use   name="CondFormats/HcalObjects"/>
<use   name="CondFormats/JetMETObjects"/>
<use   name="CondFormats/L1TObjects"/>
<use   name="CondFormats/PhysicsToolsObjects"/>
<use   name="CondFormats/GeometryObjects"/>
<use   name="CondFormats/RecoMuonObjects"/>
<use   name="CondFormats/RPCObjects"/>
<use   name="CondFormats/RunInfo"/>
<use   name="CondFormats/SiPixelObjects"/>
<use   name="CondFormats/SiStripObjects"/>
<use   name="CondFormats/Common"/>
<use   name="CondFormats/BTauObjects"/>
<use   name="CondFormats/MFObjects"/>
<export>
  <lib   name="1"/>
</export>
"""

# helper function
def sanitize(typeName):
    return typeName.replace(' ','').replace('<','_').replace('>','')

class CondXmlProcessor(object):

    def __init__(self, condDBIn):
    	self.conddb = condDBIn
    	self._pl2xml_isPrepared = False

	if not os.path.exists( os.path.join( os.environ['CMSSW_BASE'], 'src') ):
	   raise Exception("Looks like you are not running in a CMSSW developer area, $CMSSW_BASE/src/ does not exist")

	self.fakePkgName = "fakeSubSys4pl/fakePkg4pl"
	self._pl2xml_tmpDir = os.path.join( os.environ['CMSSW_BASE'], 'src', self.fakePkgName )

	self.doCleanup = True

    def __del__(self):

    	if self.doCleanup: 
           shutil.rmtree( '/'.join( self._pl2xml_tmpDir.split('/')[:-1] ) )
           os.unlink( os.path.join( os.environ['CMSSW_BASE'], 'src', './pl2xmlComp.so') )
        return 

    def discover(self, payloadType):

    	# print "discover> checking for plugin of type %s" % payloadType

        # first search in developer area:
	libDir = os.path.join( os.environ["CMSSW_BASE"], 'lib', os.environ["SCRAM_ARCH"] )
	pluginList = glob.glob( libDir + '/plugin%s_toXML.so' % sanitize(payloadType) )

        # if nothing found there, check release:
        if not pluginList:
	   libDir = os.path.join( os.environ["CMSSW_RELEASE_BASE"], 'lib', os.environ["SCRAM_ARCH"] )
	   pluginList = glob.glob( libDir + '/plugin%s_toXML.so' % sanitize(payloadType) )

	# if pluginList: 
	#    print "found plugin for %s (in %s) : %s " % (payloadType, libDir, pluginList)
	# else:
	#    print "no plugin found for type %s" % payloadType

	xmlConverter = None
	if len(pluginList) > 0:
           dirPath, libName = os.path.split( pluginList[0] )
	   sys.path.append(dirPath)
	   # print "going to import %s from %s" % (libName, dirPath)
	   xmlConverter = importlib.import_module( libName.replace('.so', '') )
	   # print "found methods: ", dir(xmlConverter)
	   self.doCleanup = False

	return xmlConverter

    def prepPayload2xml(self, session, payload):

    	startTime = time.time()
    
        # get payload from DB:
        result = session.query(self.conddb.Payload.data, self.conddb.Payload.object_type).filter(self.conddb.Payload.hash == payload).one()
        data, plType = result
    
        info = { "mdName" : "pl2xmlComp",
        	 'plType' : plType,
        	 'plTypeSan' : sanitize(plType),
    	    }
    
        converter = self.discover(plType)
	if converter: return converter

        code = payload2xmlCodeTemplate % info
    
        tmpDir = self._pl2xml_tmpDir
        if ( os.path.exists( tmpDir ) ) :
           msg = '\nERROR: %s already exists, please remove if you did not create that manually !!' % tmpDir
           self.doCleanup = False
	   raise Exception(msg)

        os.makedirs( tmpDir+'/src' )
    
        buildFileName = "%s/BuildFile.xml" % (tmpDir,)
        with open(buildFileName, 'w') as buildFile:
        	 buildFile.write( buildFileTemplate )
    	 	 buildFile.close()
    
        tmpFileName = "%s/src/%s" % (tmpDir, info['mdName'],)
        with open(tmpFileName+'.cpp', 'w') as codeFile:
        	 codeFile.write(code)
    	 	 codeFile.close()
    
	libDir = os.path.join( os.environ["CMSSW_BASE"], 'tmp', os.environ["SCRAM_ARCH"], 'src', self.fakePkgName, 'src', self.fakePkgName.replace('/',''))
	libName = libDir + '/lib%s.so' % self.fakePkgName.replace('/','') 
    	cmd = "source /afs/cern.ch/cms/cmsset_default.sh;"
    	cmd += "(cd %s ; scram b 2>&1 >build.log && cp %s $CMSSW_BASE/src/pl2xmlComp.so )" % (tmpDir, libName)
    	ret = os.system(cmd)
	if ret != 0 : self.doCleanup = False

	buildTime = time.time()-startTime
	print >> sys.stderr, "buillding done in ", buildTime, 'sec., return code from build: ', ret

	if (ret != 0):
           return None

        return importlib.import_module( 'pl2xmlComp' )
    
    def payload2xml(self, session, payload):
    
        if not self._pl2xml_isPrepared:
	   xmlConverter = self.prepPayload2xml(session, payload)
           if not xmlConverter:
              msg = "Error preparing code for "+payload
              raise Exception(msg)
           self._pl2xml_isPrepared = True

    
        # get payload from DB:
        result = session.query(self.conddb.Payload.data, self.conddb.Payload.object_type).filter(self.conddb.Payload.hash == payload).one()
        data, plType = result
    
        convFuncName = sanitize(plType)+'2xml'
        sys.path.append('.')
	func = getattr(xmlConverter, convFuncName)
    	resultXML = func( str(data), str(plType) )

        print resultXML    
    
