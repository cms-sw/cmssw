import os.path
global CONFIGDIR
BASEDIR   = os.path.dirname(os.path.dirname(__file__))
CONFIGDIR = os.path.normcase(os.path.abspath(__file__)).rsplit('/', 1)[0]

LAYOUTS = ["%s/shift_%s_T0_layout.py" % (CONFIGDIR, x) for x in
           ("castor","eb", "ee", "csc", "rpc", "hcal", "hcalcalib", "l1t", "l1temulator", "hlt", "pixel", "sistrip", "dt", "muons", "jetmet", "egamma")]
LAYOUTS += ["%s/%s_T0_layouts.py" % (CONFIGDIR, x) for x in
           ("castor","pixel","sistrip","hcal", "hcalcalib", "eb", "ee", "hltmuon")]
LAYOUTS += [CONFIGDIR + "/shift_hlt_relval_layout.py"]
LAYOUTS += [CONFIGDIR + "/hlt_relval-layouts.py"]

modules = ("Monitoring.DQM.GUI",)
envsetup = "export QUIET_ASSERT=a"

server.port        = 8060
server.serverDir   = '/dqmdata/dev/gui'
server.baseUrl     = '/dqm/dev'
server.title       = 'CMS data quality'
server.serviceName = 'CERN Development'

# Plugins.
server.plugin('render', BASEDIR + "/style/*.cc")

# Extensions.
server.extend('DQMRenderLink', server.pathOfPlugin('render'))
server.extend('DQMToJSON')
server.extend('DQMFileAccess', '/dev/null', '/dqmdata/dev/uploads',
              { 'Development': '/dqmdata/dev/repository/data'})
	        
# Sources.
server.source('DQMUnknown')
server.source('DQMOverlay')
server.source('DQMStripChart')
server.source('DQMLive', 'localhost:8061')
server.source('DQMArchive', '/dqmdata/dev/ix', '^/Global/')
server.source('DQMLayout', *LAYOUTS)

# Services and workspaces
execfile(CONFIGDIR + "/dqm-services.py")
execfile(CONFIGDIR + "/workspaces-dev.py")
