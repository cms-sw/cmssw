#!/usr/bin/env python

import CMGTools.Production.eostools as castortools

import os, sys
from optparse import OptionParser

import CMGTools.Production.eostools as castortools

class Migrate(object):
    
    def __init__(self, src, dest):
        
        #this seems to matter to xrd
        if not dest.endswith(os.sep):
            dest = dest+ os.sep
        
        self.src = src
        self.dest = dest
        
        #self.src_files = [f for f in castortools.listFiles(src, rec = False) if f and castortools.isFile(f)]
        #self.dest_files = [f for f in castortools.listFiles(dest, rec = False) if f and castortools.isFile(f)]
        self.src_files = [(f[4],f[1]) for f in castortools.listFiles(src, rec = False, full_info = True) if f]
        self.dest_files = [(f[4],f[1]) for f in castortools.listFiles(dest, rec = False, full_info = True) if f]
        
        self.problem_files = []
        
        #clean the files if they are already present
        src_dist = {}
        for s in self.src_files:
            src_dist[os.path.basename(s[0])] = s
        
        for d in self.dest_files:
            base = os.path.basename(d[0])
            if src_dist.has_key(base):
                s = src_dist[base]
                if d[1] != s[1]:
                    self.problem(s,'File exists in destination, but with different file size.')
                print "Removing '%s' from copy list" % s[0]
                self.src_files.remove(s)
                
    
    def problem(self, name, reason):
        print 'File problem',name,reason
        self.problem_files.append((name,reason))
    
    def copy(self, path, dest, fraction = 1.0):
        print 'copying',path
        out, err, ret = castortools.runXRDCommand(path,'cp',dest)
        if "cp returned 0" in out:
            print "[%f] The file '%s' was copied successfully" % (fraction,path)
        else:
            self.problem(path,'The copy failed. Someoutput is here "%s" and here "%s"' % (out, err))
    
    def migrate(self):
        
        count = 0
        for s in self.src_files:
            fraction = count/(len(self.src_files)*1.)
            self.copy(s[0],self.dest,fraction)
            count += 1
        
        if self.problem_files:
            print >> sys.stderr, 'The following files had problems, and must be handled manually'
            for p in self.problem_files:
                print >> sys.stderr, p[0],p[1]
        

if __name__ == '__main__':

    parser = OptionParser()
    parser.usage = """
%s --src source_dir --dest dest_dir

Both the source and destination must exist
"""

    parser.add_option("-s", "--src", dest="source",
                  help="The source directory", default=None)
    parser.add_option("-d", "--dest", dest="dest",
                  help="The destination directory", default=None)

    (options,args) = parser.parse_args()

    if options.source is None or options.dest is None:
        print >> sys.stderr, 'Both the source and destination must be set'
        sys.exit(-1)
        
    if not castortools.isDirectory(options.source) or not castortools.isDirectory(options.dest):
        print >> sys.stderr, 'Both the source and destination directories must exist'
        sys.exit(-1)
        
    m = Migrate(options.source,options.dest)
    m.migrate()
