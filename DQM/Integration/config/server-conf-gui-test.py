import os.path
global CONFIGDIR
BASEDIR   = os.path.dirname(os.path.dirname(__file__))
CONFIGDIR = os.path.normcase(os.path.abspath(__file__)).rsplit('/', 1)[0]

LAYOUTS = ["%s/%s-layouts.py" % (CONFIGDIR, x) for x in
	   ("castor","csc", "dt", "eb", "ee", "es","hcal", "hcalcalib", "hlt", "hlx", "l1t", "l1temulator", "rpc", "pixel", "sistrip")]
LAYOUTS += ["%s/%s_overview_layouts.py" % (CONFIGDIR, x) for x in
            ("sistrip","ecal","hcal","beammonitor","l1t","hlt")]
LAYOUTS += ["%s/shift_%s_layout.py" % (CONFIGDIR, x) for x in
            ("castor","csc", "dt", "eb", "ee", "es","hcal", "hcalcalib", "hlt", "hlx", "l1t", "l1temulator", "rpc", "pixel", "sistrip" , "fed" )]

modules = ("Monitoring.DQM.GUI",)
envsetup = "export QUIET_ASSERT=a"

#server.instrument  = 'valgrind --num-callers=999 `cmsvgsupp` --error-limit=no'
#server.instrument  = 'valgrind --tool=helgrind --num-callers=999 --error-limit=no'
#server.instrument  = 'igprof -d -t python -pp'
server.serverDir   = '/home/dqm/gui'
server.baseUrl     = '/dqm/gui-test'
server.title       = 'CMS data quality'
server.serviceName = 'GUI test'

server.plugin('render', BASEDIR + "/style/*.cc")
server.extend('DQMRenderLink', server.pathOfPlugin('render'))
server.extend('DQMFileAccess', None, None,
              { "Original": "/dqmdata/dqm/repository/original",
                "iSpy": "/dqmdata/EventDisplay/done" })
server.source('DQMUnknown')
server.source('DQMOverlay')
server.source('DQMStripChart')
server.source('DQMLive', 'dqm-integration:9090')
server.source('DQMArchive', '/home/dqm/ix', '^/Global/')
server.source('DQMLayout', *LAYOUTS)

execfile(CONFIGDIR + "/dqm-services.py")
execfile(CONFIGDIR + "/workspaces-online.py")
