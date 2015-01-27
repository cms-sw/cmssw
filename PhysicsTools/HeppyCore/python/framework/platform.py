from ROOT import TFile

def platform(filename):
    '''Detects the platform on which heppy is running, and returns BARE, CMSSW, or FCC.  
    '''
    rootfile = TFile(filename)
    keys = rootfile.GetListOfKeys()
    cmssw_keys = ['MetaData', 'ParameterSets', 'Events', 'LuminosityBlocks', 'Runs']
    is_cmssw = True
    for key in cmssw_keys:
        if key not in keys:
            is_cmssw = False
    if is_cmssw: 
        return 'CMSSW'
    else:
        return 'BARE'

if __name__ == '__main__':
    import sys
    fnam = sys.argv[1]
    print platform(fnam)
