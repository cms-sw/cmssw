import os.path
global CONFIGDIR
BASEDIR   = os.path.dirname(os.path.dirname(__file__))
CONFIGDIR = os.path.normcase(os.path.abspath(__file__)).rsplit('/', 1)[0]

#LAYOUTS = ["%s/shift_%s-layout.py" % (CONFIGDIR, x) for x in
LAYOUTS = ["%s/shift_%s_T0_layout.py" % (CONFIGDIR, x) for x in
           ("castor","eb", "ee","es", "csc", "rpc", "hcal", "hcalcalib", "hlt","l1t", "l1temulator", "pixel", "sistrip","tracking", "dt", "muons", "jetmet", "egamma")]
LAYOUTS += ["%s/%s_overview_layouts.py" % (CONFIGDIR, x) for x in
            ("sistrip","ecal","hcal","beammonitor","l1t","hlt")]
#LAYOUTS += ["%s/%s_-layouts.py" % (CONFIGDIR, x) for x in
LAYOUTS += ["%s/%s_T0_layouts.py" % (CONFIGDIR, x) for x in
           ("btag","castor","csc","pixel","sistrip","tracking","hcal", "hcalcalib", "eb", "ee","es", "rpc")]
LAYOUTS += ["%s/%s_caf_layouts.py" % (CONFIGDIR, x) for x in
           ("tkal",)]
LAYOUTS += [CONFIGDIR + "/shift_hlt_relval_layout.py"]
LAYOUTS += [CONFIGDIR + "/hlt_relval-layouts.py"]

modules = ("Monitoring.DQM.GUI",)
envsetup = "export QUIET_ASSERT=a"

#server.instrument  = 'valgrind --num-callers=999 `cmsvgsupp` --error-limit=no'
#server.instrument  = 'valgrind --tool=helgrind --num-callers=999 --error-limit=no'
#server.instrument  = 'igprof -d -t python -pp'
#server.instrument  = 'igprof -d -t python -mp'
server.port        = 8070
server.serverDir   = '/dqmdata/dqmtest/gui'
server.baseUrl     = '/dqm/offline-test'
server.title       = 'CMS data quality'
server.serviceName = 'Offline Test'

# Plugins.
server.plugin('render', "/dqmdata/dqmtest/style/*.cc")

# Extensions.
server.extend('DQMRenderLink', server.pathOfPlugin('render'))
server.extend('DQMToJSON')
server.extend('DQMFileAccess', '/dev/null', '/dqmdata/dqmtest/uploads',
	      { 'ROOT': '/dqmdata/dqmtest/repository/data',
	        'ZIP': '/dqmdata/dqmtest/repository/zipped' })

# Sources.
server.source('DQMUnknown')
server.source('DQMOverlay')
server.source('DQMStripChart')

# Read from the production index to test that the layouts work.
server.source('DQMArchive', '/dqmdata/offline/ix', '^/Global/Online')
server.source('DQMLayout', *LAYOUTS)

# Services and workspaces.
execfile(CONFIGDIR + "/dqm-services.py")
execfile(CONFIGDIR + "/workspaces-tier-0.py")
