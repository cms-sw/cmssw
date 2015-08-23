import ROOT, subprocess, json

def edmFileLs(fname):
    out = subprocess.check_output(['edmFileUtil','--ls','-j',fname])
    jdata = json.loads(out)
    return jdata[0]
