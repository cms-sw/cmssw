from ROOT import TFile

def file_dir_names(name):
    spl = name.split(':')
    file = spl[0]
    dir = None
    if len(spl)==2:
        file = spl[0]
        dir = spl[1]
    return file, dir


def file_dir(name):
    '''Splits file.root:dir, and returns the TFile and TDirectory objects'''
    file, dir = file_dir_names(name)
    tfile = TFile(file)
    tdir = tfile 
    if dir:
        tdir = tfile.Get(dir)
    return tfile, tdir
