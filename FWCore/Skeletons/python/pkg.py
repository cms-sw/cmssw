#!/usr/bin/env python
#-*- coding: utf-8 -*-
#pylint: disable-msg=W0122,R0914,R0912

"""
File       : pkg.py
Author     : Valentin Kuznetsov <vkuznet@gmail.com>
Description: AbstractGenerator class provides basic functionality
to generate CMSSW class from given template
"""

# system modules
import os
import sys
import time
import pprint

# package modules
from FWCore.Skeletons.utils import parse_word, functor, user_info, tree

class AbstractPkg(object):
    """
    AbstractPkg takes care how to generate code from template/PKG
    package area. The PKG can be any directory which may include
    any types of files, e.g. C++ (.cc), python (.py), etc.
    This class relies on specific logic which we outline here:

        - each template may use tags defined with double underscores
          enclosure, e.g. __class__, __record__, etc.
        - each template may have example tags, such tags should
          start with @example_. While processing template user may
          choose to strip them off or keep the code behind those tags
        - in addition user may specify pure python code which can
          operate with user defined tags. This code snipped should
          be enclosed with #python_begin and #python_end lines
          which declares start and end of python block
    """
    def __init__(self, config=None):
        super(AbstractPkg, self).__init__()
        if  not config:
            self.config = {}
        else:
            self.config = config
        self.pname  = self.config.get('pname', None)
        self.tmpl   = self.config.get('tmpl', None)
        self.debug  = self.config.get('debug', 0)
        self.tdir   = self.config.get('tmpl_dir')
        self.author = user_info(self.config.get('author', None))
        self.date   = time.strftime("%a, %d %b %Y %H:%M:%S GMT", time.gmtime())
        self.rcsid  = '$%s$' % 'Id' # CVS commit is too smart
        self.not_in_dir = self.config.get('not_in_dir', [])
        
    def tmpl_etags(self):
        "Scan template files and return example tags"
        keys = []
        sdir = '%s/%s' % (self.tdir, self.tmpl)
        for name in os.listdir(sdir):
            if  name[-1] == '~':
                continue
            if  name == 'CVS':
                continue
            fname = os.path.join(sdir, name)
            with open(fname, 'r') as stream:
                for line in stream.readlines():
                    if  line.find('@example_') != -1: # possible tag
                        keys += [k for k in line.split() if \
                                    k.find('@example_') != -1]
        return set(keys)

    def print_etags(self):
        "Print out template example tags"
        for key in self.tmpl_etags():
            print key

    def tmpl_tags(self):
        "Scan template files and return template tags"
        keys = []
        sdir = '%s/%s' % (self.tdir, self.tmpl)
        for name in os.listdir(sdir):
            if  name[-1] == '~':
                continue
            if  name == 'CVS':
                continue
            fname = os.path.join(sdir, name)
            with open(fname, 'r') as stream:
                for line in stream.readlines():
                    if  line.find('__') != -1: # possible key
                        keys += [k for k in parse_word(line)]
        return set(keys)

    def print_tags(self):
        "Print out template keys"
        for key in self.tmpl_tags():
            print key

    def parse_etags(self, line):
        """
        Determine either skip or keep given line based on class tags 
        meta-strings
        """
        tmpl_etags = self.tmpl_etags()
        keep_etags = self.config.get('tmpl_etags', [])
        for tag in tmpl_etags:
            if  keep_etags:
                for valid_tag in keep_etags:
                    if  line.find(valid_tag) != -1:
                        line = line.replace(valid_tag, '')
                        return line
            else:
                if  line.find(tag) != -1:
                    line = line.replace(tag, '')
                    line = ''
                    return line
        return line

    def write(self, fname, tmpl_name, kwds):
        "Create new file from given template name and set of arguments"
        code = ""
        read_code = False
        with open(fname, 'w') as stream:
            for line in open(tmpl_name, 'r').readlines():
                line = self.parse_etags(line)
                if  not line:
                    continue
                if  line.find('#python_begin') != -1:
                    read_code = True
                    continue
                if  line.find('#python_end') != -1:
                    read_code = False
                if  read_code:
                    code += line
                if  code and not read_code:
                    res   = functor(code, kwds, self.debug)
                    stream.write(res)
                    code  = ""
                    continue
                if  not read_code:
                    for key, val in kwds.items():
                        if  isinstance(val, basestring):
                            line = line.replace(key, val)
                    stream.write(line)

    def get_kwds(self):
        "Return keyword arguments to be used in methods"
        kwds  = {'__pkgname__': self.config.get('pkgname', 'Package'),
                 '__author__': self.author,
                 '__user__': os.getlogin(),
                 '__date__': self.date,
                 '__class__': self.pname,
                 '__name__': self.pname,
                 '__rcsid__': self.rcsid,
                 '__subsys__': self.config.get('subsystem', 'Subsystem')}
        args = self.config.get('args', None)
        kwds.update(args)
        if  self.debug:
            print "Template tags:"
            pprint.pprint(kwds)
        return kwds

    def generate(self):
        "Generate package templates in a given directory"

        # keep current location, since generate will switch directories
        cdir = os.getcwd()

        # read from configutation which template files to create
        tmpl_files = self.config.get('tmpl_files', 'all')

        # setup keyword arguments which we'll pass to write method
        kwds = self.get_kwds()

        # create template package dir and cd into it
        if  tmpl_files == 'all' and self.tmpl not in self.not_in_dir:
            if  os.path.isdir(self.pname):
                msg  = "Can't create package '%s'\n" % self.pname
                msg += "Directory %s is already exists" % self.pname
                print msg
                sys.exit(1)
            os.makedirs(self.pname)
            os.chdir(self.pname)

        # read directory driver information and create file list to generate
        sdir    = os.path.join(self.tdir, self.tmpl)
        sources = [s for s in os.listdir(sdir) \
                if s != 'Driver.dir' and s.find('~') == -1]
        driver  = os.path.join(sdir, 'Driver.dir')
        if  os.path.isfile(driver):
            sources = [s.replace('\n', '') for s in open(driver, 'r').readlines()]
        if  'CVS' in sources:
            sources.remove('CVS')

        # special case of Skeleton, which requires to generate only given
        # file type if self.pname has extension of that type
        names = set([s.split('.')[0] for s in sources])
        if  names == set(['Skeleton']):
            if  self.pname.find('.') != -1:
                _, ext = os.path.splitext(self.pname)
                sources = [s for s in sources if s.rfind(ext) != -1]
                self.pname = self.pname.replace(ext, '')
                kwds = self.get_kwds()
                if  not sources:
                    msg = 'Unable to find skeleton for extension "%s"' % ext
                    print msg
                    sys.exit(1)
            bdir = os.environ.get('CMSSW_BASE', '')
            dirs = os.getcwd().replace(bdir, '').split('/')
            ldir = os.getcwd().split('/')[-1]
            idir = ''
            subsys  = kwds['__subsys__']
            pkgname = kwds['__pkgname__']
            if  sources == ['Skeleton.cc', 'Skeleton.h']:
                if  ldir == 'interface' and os.getcwd().find(bdir) != -1:
                    idir = '%s/%s/interface/' % (subsys, pkgname)
            # run within some directory of the Sybsystem/Pkg area
            # and only for mkskel <file>.cc
            elif sources == ['Skeleton.cc'] and \
                len(dirs) == 5 and dirs[0] == ''  and dirs[1] == 'src':
                idir = '%s/%s/interface/' % (subsys, pkgname)
            elif sources == ['Skeleton.h'] and ldir == 'interface' and \
                len(dirs) == 5 and dirs[0] == ''  and dirs[1] == 'src':
                idir = '%s/%s/interface/' % (subsys, pkgname)
            kwds.update({'__incdir__': idir})

        # loop over source files, create dirs as necessary and generate files
        # names for writing templates
        gen_files = []
        for src in sources:
            if  tmpl_files != 'all':
                fname, ext = os.path.splitext(src)
                if  tmpl_files != ext:
                    continue
                src = src.split('/')[-1]
            if  self.debug:
                print "Read", src
            items = src.split('/')
            if  items[-1] == '/':
                items = items[:-1]
            tname     = items[-1] # template file name
            tmpl_name = os.path.join(sdir, items[-1]) # full tmpl file name
            if  os.path.isfile(tmpl_name):
                ftype = 'file'
            else:
                ftype = 'dir'
            name2gen  = src # new file we'll create
            if  tname.split('.')[0] == self.tmpl: # need to substitute
                name2gen  = name2gen.replace(self.tmpl, self.pname)
            name2gen  = os.path.join(os.getcwd(), name2gen)
            if  self.debug:
                print "Create", name2gen
            if  ftype == 'dir':
                if  not os.path.isdir(name2gen):
                    os.makedirs(name2gen)
                continue # we're done with dir
            fdir = os.path.dirname(name2gen)
            if  not os.path.isdir(fdir):
                os.makedirs(fdir)
            self.write(name2gen, tmpl_name, kwds)
            gen_files.append(name2gen.split('/')[-1])
        if  tmpl_files == 'all' and self.tmpl not in self.not_in_dir:
            msg  = 'New package "%s" of %s type is successfully generated' \
                    % (self.pname, self.tmpl)
        else:
            msg = 'Generated %s file' % ', '.join(gen_files)
            if  len(gen_files) > 1:
                msg += 's'
        print msg
        # return back where we started
        os.chdir(cdir)
        if  msg.find('New package') != -1:
            tree(self.pname)
