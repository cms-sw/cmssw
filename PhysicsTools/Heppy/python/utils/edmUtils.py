import ROOT, subprocess, json, re

def edmFileLs(fname):
    # if it's not a cms LFN and it does not have a protocol, put file: for it
    if not re.match(r"(/store|\w+:).*",fname): fname = "file:"+fname
    out = subprocess.check_output(['edmFileUtil','--ls','-j',fname])
    jdata = json.loads(out)
    return jdata[0]
