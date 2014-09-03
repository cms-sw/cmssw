import os.path
global CONFIGDIR
BASEDIR   = os.path.dirname(os.path.dirname(__file__))
CONFIGDIR = os.path.normcase(os.path.abspath(__file__)).rsplit('/', 1)[0]

LAYOUTS = ["%s/shift_%s_T0_layout.py" % (CONFIGDIR, x) for x in
           ("castor","eb", "ee", "es", "csc", "rpc", "hcal", "hcalcalib", "l1t", "l1temulator", "hlt", "pixel", "sistrip", "tracking", "dt", "muons", "jetmet", "egamma")]
LAYOUTS += ["%s/%s_overview_layouts.py" % (CONFIGDIR, x) for x in
            ("sistrip","ecal","hcal","beammonitor","l1t","hlt")]
LAYOUTS += ["%s/%s_T0_layouts.py" % (CONFIGDIR, x) for x in
           ("btag","castor","csc","pixel","sistrip","tracking","hcal", "hcalcalib", "eb", "ee", "es", "hltmuon", "rpc")]
LAYOUTS += [CONFIGDIR + "/shift_hlt_relval_layout.py"]
LAYOUTS += [CONFIGDIR + "/hlt_relval-layouts.py"]

modules = ("Monitoring.DQM.GUI",)
envsetup = "export QUIET_ASSERT=a"

server.port        = 8080
server.serverDir   = '/dqmdata/offline/gui'
server.baseUrl     = '/dqm/offline'
server.title       = 'CMS data quality'
server.serviceName = 'Offline'

# Plugins.
server.plugin('render', BASEDIR + "/style/*.cc")

# Extensions.
server.extend('DQMRenderLink', server.pathOfPlugin('render'))
server.extend('DQMToJSON')
server.extend('DQMFileAccess', '/dev/null', '/dqmdata/offline/uploads',
	      { 'ROOT': '/dqmdata/offline/repository/data',
	        'ZIP': '/dqmdata/offline/repository/zipped'})
	        
# Sources.
server.source('DQMUnknown')
server.source('DQMOverlay')
server.source('DQMStripChart')
server.source('DQMArchive', '/dqmdata/offline/ix', '^/Global/')
server.source('DQMLayout', *LAYOUTS)

# Services and workspaces.
execfile(CONFIGDIR + "/dqm-services.py")
execfile(CONFIGDIR + "/workspaces-tier-0.py")
