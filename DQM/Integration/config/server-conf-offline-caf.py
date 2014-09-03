import os.path
global CONFIGDIR
BASEDIR   = os.path.dirname(os.path.dirname(__file__))
CONFIGDIR = os.path.normcase(os.path.abspath(__file__)).rsplit('/', 1)[0]

LAYOUTS = ["%s/%s_caf_layouts.py" % (CONFIGDIR, x) for x in
	   ("tkal",)]

modules = ("Monitoring.DQM.GUI",)
envsetup = "export QUIET_ASSERT=a"

server.port        = 8040
server.serverDir   = '/dqmdata/caf/gui'
server.baseUrl     = '/dqm/caf'
server.title       = 'CMS data quality'
server.serviceName = 'CERN CAF'

# Plugins.
server.plugin('render', BASEDIR + "/style/*.cc")

# Extensions.
server.extend('DQMRenderLink', server.pathOfPlugin('render'))
server.extend('DQMToJSON')
server.extend('DQMFileAccess', '/dev/null', '/dqmdata/caf/uploads',
              { 'CAF': '/dqmdata/caf/repository/data'})
	        
# Sources.
server.source('DQMUnknown')
server.source('DQMOverlay')
server.source('DQMStripChart')
server.source('DQMArchive', '/dqmdata/caf/ix', '^/Global/')
server.source('DQMLayout', *LAYOUTS)

# Services and workspaces
execfile(CONFIGDIR + "/dqm-services.py")
execfile(CONFIGDIR + "/workspaces-caf.py")
