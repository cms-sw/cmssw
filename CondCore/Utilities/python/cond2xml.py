
import os
import shutil
import sys
import time

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

  std::string payload2xml( const std::string &payloadData, const std::string &payloadType ) { 

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
    def ("payload2xml", payload2xml);
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

class CondXmlProcessor(object):

    def __init__(self, condDBIn):
    	self.conddb = condDBIn
    	self._pl2xml_isPrepared = False
	self._pl2xml_tmpDir = "fakeSubSys4pl/fakePkg4pl"
	self.doCleanup = True

    def __del__(self):

    	if self.doCleanup: 
 	   shutil.rmtree( self._pl2xml_tmpDir.split('/')[0] )
           os.unlink('./pl2xmlComp.so')
        return 

    def prepPayload2xml(self, session, payload):

    	startTime = time.time()
    
        # get payload from DB:
        result = session.query(self.conddb.Payload.data, self.conddb.Payload.object_type).filter(self.conddb.Payload.hash == payload).one()
        data, plType = result
    
        info = { "mdName" : "pl2xmlComp",
        	     'plType' : plType,
    	    }
    
        code = payload2xmlCodeTemplate % info
    
        tmpDir = self._pl2xml_tmpDir
        if ( os.path.exists( tmpDir.split('/')[0] ) or
 	     os.path.exists( tmpDir ) ) :
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
    
    	cmd = "source /afs/cern.ch/cms/cmsset_default.sh;"
    	cmd += "(cd %s ; scram b 2>&1 >build.log && cp %s/tmp/%s/src/%s/src/%s/lib%s.so ../../pl2xmlComp.so )" % (tmpDir, os.environ["CMSSW_BASE"], os.environ["SCRAM_ARCH"], tmpDir, tmpDir.replace('/',''), tmpDir.replace('/','') ) 
    	ret = os.system(cmd)
	if ret != 0 : self.doCleanup = False

	buildTime = time.time()-startTime
	print >> sys.stderr, "buillding done in ", buildTime, 'sec., return code from build: ', ret

        return (ret == 0)
    
    def payload2xml(self, session, payload):
    
        if not self._pl2xml_isPrepared:
           if not self.prepPayload2xml(session, payload):
              msg = "Error preparing code for "+payload
              raise Exception(msg)
           self._pl2xml_isPrepared = True
    
        # get payload from DB:
        result = session.query(self.conddb.Payload.data, self.conddb.Payload.object_type).filter(self.conddb.Payload.hash == payload).one()
        data, plType = result
    
        sys.path.append('.')
        import pl2xmlComp
        resultXML = pl2xmlComp.payload2xml( data, plType )
        print resultXML    
    
