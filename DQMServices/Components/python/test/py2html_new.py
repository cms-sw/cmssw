#!/usr/bin/env python
# Original Author:  Marco Rovere
#         Created:  Tue Feb 9 10:06:02 CEST 2010
# $Id: py2html_new.py,v 1.1 2013/04/26 12:17:10 deguio Exp $

import re
import sys, os
import sqlite3
import FWCore.ParameterSet.Config as cms
import locale

locale.setlocale(locale.LC_ALL, 'en_US')

class visitor:
    def __init__(self, out):
        self.out = out
        self.env = os.getenv('CMSSW_RELEASE_BASE')
        self.t = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        self.level_ = 0
        self.level = {}
        self.level[self.level_] = 0
        
    def reset(self):
        self.t = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        
    def enter(self, value):
        if type(value) == cms.Sequence:
            self.out.write('<ol><div class=sequence>%s</div>\n' % value.label())
            self.level_ +=1
            self.level[self.level_] = 0
        else:
            if type(value) == cms.EDAnalyzer:
                self.out.write( '<li>EDAnalyzer ' )
            elif type(value) == cms.EDProducer:
                self.out.write( '<li>EDProducer ' )
            elif type(value) == cms.EDFilter:
                self.out.write( '<li>EDFilter ' )
            mem = self.dumpProducerOrFilter(value)
            for i in range(len(mem)):
                self.t[i] += int(mem[i])
                self.level[self.level_] += int(mem[i])
    def leave(self, value):
        if type(value) == cms.Sequence:
            self.out.write('(%s, %s, %s, %s, %s) %f [%f] - %s' % (prettyInt(self.t[0]),\
                                                                  prettyInt(self.t[1]),\
                                                                  prettyInt(self.t[2]),\
                                                                  prettyInt(self.t[3]),\
                                                                  prettyInt(self.t[4]),\
                                                                  (self.t[0]+self.t[1]+self.t[2]+self.t[3]+self.t[4])/1024./1024.,\
                                                                  self.level[self.level_]/1024./1024.,\
                                                                  value.label()))
#            print self.level
            self.level_ -= 1
            if self.level_ > 0:
                self.level[self.level_] += self.level[self.level_+1]
                self.level[self.level_+1] = 0
            self.out.write( '</ol>\n')
        else:
            self.out.write( '</li>\n')
    def Fake(self):
        return 'Not_Available'
    
    def dumpProducerOrFilter(self, value):
        type_ = getattr(value, 'type_', self.Fake)
        filename_ = getattr(value, '_filename', self.Fake())
        lbl_ = getattr(value, 'label_', self.Fake)
        dumpConfig_ = getattr(value, 'dumpConfig', self.Fake)
        link = ''
        counter = 0
        t = {}
        for step in steps:
          toCheck = step
          if toCheck == 'ctor':
            toCheck = type_()
          cur = conn.execute("select mainrows.cumulative_count, name from symbols inner join mainrows on mainrows.symbol_id = symbols.id where name like '%s::%s%%';" % (type_(),toCheck))
          for r in cur:
            t[counter] = int(r[0])
          if len(t) != counter+1:
            t[counter] = 0
          counter += 1
        stats = '(%s, %s, %s, %s, %s) %s' % (prettyInt(t[0]), \
                                             prettyInt(t[1]), \
                                             prettyInt(t[2]), \
                                             prettyInt(t[3]), \
                                             prettyInt(t[4]), \
                                             prettyFloat((t[0]+t[1]+t[2]+t[3]+t[4])/1024./1024.))
        link = '<a href=http://cmslxr.fnal.gov/lxr/ident?i=' + type_() + '>' + type_() + '</a> ' + stats + '\n'
        out.write(link + ', label <a href=' + lbl_() + '.html>' + lbl_() +'</a>, defined in ' + filename_ + '</li>\n')
        tmpout = open('./html/'+lbl_() + '.html','w')
        tmpout.write(preamble())
        tmpout.write( '<pre>\n')
        cutAtColumn = 978
        gg = dumpConfig_()
        for line in gg.split('\n'):
            blocks = len(line)/cutAtColumn + 1
            for i in range(0,blocks):
                tmpout.write('%s' % line[i*cutAtColumn:(i+1)*cutAtColumn])
                if blocks > 1 and not (i == blocks):
                    tmpout.write('\\ \n')
                else:
                    tmpout.write('\n')
        tmpout.write( '<pre>\n')
        tmpout.write(endDocument())
        tmpout.close()
        return t

def prettyInt(val):
    return locale.format("%d", int(val), grouping=True).replace(',', "'")

def prettyFloat(val):
    return locale.format("%.2f", float(val), grouping=True).replace(',', "'")

def preamble():
    return """
<html>
 <head>
  <title>Config Browser</title>
  <style type="text/css">
  ol {
   font-family: Arial;
   font-size: 10pt;
   padding-left: 25px;
  }
  .sequence {
   font-weight: bold;
  }
</style>
 </head>

 <body>
"""

def endDocument():
    return "</body>\n</html>\n"

def checkRel():
    env = os.getenv('CMSSW_RELEASE_BASE')
    if env is not None:
        print 'Working with release ', os.getenv('CMSSW_VERSION'), ' in area ', env
        print os.getcwd()
    else:
        print 'You must set a proper CMSSW environment first. Quitting.'
        sys.exit(1)

# Check argument list
if len(sys.argv) < 3:
    print 'Error.\nUsage py2tex python_config_file igprof_report'
    sys.exit(1)
else:
    toload = sys.argv[1]
    toload = re.sub('\.py', '', toload)
    towrite = 'index.html'

conn = sqlite3.connect(sys.argv[2])
steps = ('ctor', 'beginJob', 'beginRun', 'beginLuminosityBlock', 'analyze')
pwd = os.getenv('PWD') +'/'
sys.path.append(pwd)
checkRel()

print "Trying to load", toload
try:
    a = __import__(str(toload))
except:
    print 'Import Failed, quitting...'
    sys.exit(1)

if not os.path.exists(pwd+'html'):
    os.mkdir(pwd+'html')
out = open('./html/'+towrite,'w')

# Main starts here.

# Check if the supplied configuration file has a process defined in
# it. In case there is no process defined (e.g. when the file only
# contains paths and/or sequences), create a small fake configuraion
# file, with a DUMMY process and load the file under investigation on
# top of it.

try:
    print a.process.process
except:
    print 'No process defined in the supplied configuration file: creating a DUMMY one'
    cmsswBase = os.getenv('CMSSW_BASE')
    tmpFile = open('dummy.py','w')
    tmpFile.write("import FWCore.ParameterSet.Config as cms\n")
    tmpFile.write("process = cms.Process('FAKE')\n\n")
    pythonLibToLoad = (pwd + sys.argv[1]).replace(cmsswBase,'').replace('/src/','').replace('/', '.').replace('.python','').replace('.py','')
    tmpFile.write("process.load('" + pythonLibToLoad + "')")
    tmpFile.close()
    a = __import__('dummy')
    os.unlink('dummy.py')
    os.unlink('dummy.pyc')

out.write(preamble())
out.write( '<h1>Paths</h1>\n')
v = visitor(out)
for k in a.process.paths.keys():
    out.write('<H2>%s</H2>\n' % k)
    a.process.paths[k].visit(v)
    v.reset()

out.write( '<h1>End Paths</h1>\n')
v.reset()

for k in a.process.endpaths.keys():
    out.write('<H2>%s</H2>\n' % k)
    a.process.endpaths[k].visit(v)
    v.reset()

out.write(endDocument())
