import os.path, socket; global CONFIGDIR
CONFIGDIR = os.path.normcase(os.path.abspath(__file__)).rsplit('/', 1)[0]
BASEDIR   = CONFIGDIR.replace("/current/config/dqmgui", "")
STATEDIR  = "%s/state/dqmgui/relval" % BASEDIR
LOGDIR    = "%s/logs/dqmgui/relval" % BASEDIR

LAYOUTS = ["%s/layouts/shift_%s_relval_layout.py" % (CONFIGDIR, x) for x in
           ("hlt", "ecal" )]
LAYOUTS += ["%s/layouts/%s_overview_layouts.py" % (CONFIGDIR, x) for x in
            ("sistrip","ecal","hcal","beammonitor","l1t","hlt")]
LAYOUTS += ["%s/layouts/%s_relval-layouts.py" % (CONFIGDIR, x) for x in
           ("hlt", "ecal" )]

modules = ("Monitoring.DQM.GUI",)

#server.instrument  = 'valgrind --num-callers=999 `cmsvgsupp` --error-limit=no'
#server.instrument  = 'valgrind --tool=helgrind --num-callers=999 --error-limit=no'
#server.instrument  = 'igprof -d -t python -pp'
#server.instrument  = 'igprof -d -t python -mp'
server.port        = 8081
server.serverDir   = STATEDIR
server.logFile     = '%s/weblog-%%Y%%m%%d.log' % LOGDIR
server.baseUrl     = '/dqm/relval'
server.title       = 'CMS data quality'
server.serviceName = 'RelVal'

server.plugin('render', "%s/style/*.cc" % CONFIGDIR)
server.extend('DQMRenderLink', server.pathOfPlugin('render'))
server.extend('DQMToJSON')
server.extend('DQMFileAccess', None, "%s/uploads" % STATEDIR,
              { "ROOT": "%s/data" % STATEDIR,
                "ZIP": "%s/zipped" % STATEDIR })
server.extend('DQMLayoutAccess', None, STATEDIR,
              ['/DC=ch/DC=cern/OU=Organic Units/OU=Users/CN=lilopera/CN=692665/CN=Luis Ignacio Lopera Gonzalez',
               '/DC=ch/DC=cern/OU=Organic Units/OU=Users/CN=rovere/CN=653292/CN=Marco Rovere' ])
server.source('DQMUnknown')
server.source('DQMOverlay')
server.source('DQMStripChart')
server.source('DQMArchive', "%s/ix" % STATEDIR, '^/Global/')
server.source('DQMLayout')

execfile(CONFIGDIR + "/dqm-services.py")
execfile(CONFIGDIR + "/workspaces-relval.py")
