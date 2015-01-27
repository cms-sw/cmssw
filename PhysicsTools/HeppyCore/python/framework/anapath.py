import os
import glob

def anapath():
    analyzer_path = []

    cmssw_src = '/'.join( [ os.environ['CMSSW_BASE'], 'src' ] )
    pkg_pat = '/'.join([cmssw_src, '*', '*'])
    pkg_dirs = glob.glob( pkg_pat )
    packages = [dir for dir in pkg_dirs \
                    if os.path.isdir(dir) and \
                    not dir.endswith('CVS') ]
    for pkg in packages:
        ana_dir = '/'.join( [pkg, 'python/analyzers'] )
        # if pkg.endswith('CMGTools/H2TauTau'):
        #    ana_dir = '/'.join( [pkg, 'python/proto/analyzers'] )
        if os.path.isdir(ana_dir):
            analyzer_path.append( ana_dir )
    return analyzer_path

analyzer_path = anapath()

if __name__ == '__main__':
    print analyzer_path    
    
