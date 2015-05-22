import os
import glob

cmssw_src = '/'.join( [ os.environ['CMSSW_BASE'], 'src' ] )

pkg_pat = '/'.join([cmssw_src, '*', '*'])
pkg_dirs = glob.glob( pkg_pat )

packages = [dir for dir in pkg_dirs \
            if os.path.isdir(dir) and \
            not dir.endswith('CVS') and \
            dir.find('src/CMGTools')!=-1
            ]

pythonpath = ['.']
for pkg in packages:
    ana_dir = '/'.join( [pkg, 'python/analyzers'] )
    if pkg.endswith('CMGTools/H2TauTau'):
        ana_dir = '/'.join( [pkg, 'python/proto/analyzers'] )
    if os.path.isdir(ana_dir):
        pythonpath.append( ana_dir )
    

if __name__=='__main__':
    
    import pprint
    import sys
    
    pprint.pprint(packages)
    print
    pprint.pprint(sys.path)
    sys.path = pythonpath + sys.path 
    print 
    pprint.pprint(sys.path)

    
