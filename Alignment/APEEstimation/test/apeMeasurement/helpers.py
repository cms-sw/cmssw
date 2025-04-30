import re
import glob
import os
import shutil
import errno

def findFiles(path, fileName):
    
    expr = fileName.format(number="*")
    return glob.glob(os.path.join(path,expr))


shortcuts = {}
# regex matching on key, replacement of groups on value
# implement any other shortcuts that you want to use
#sources
shortcuts["mp([0-9]*)"] = "sqlite_file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/MP/MPproduction/mp{0}/jobData/jobm/alignments_MP.db"
shortcuts["mp([0-9]*)_jobm([0-9]*)"] = "sqlite_file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/MP/MPproduction/mp{0}/jobData/jobm{1}/alignments_MP.db"
shortcuts["sm([0-9]*)_iter([0-9]*)"] = "sqlite_file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN2/HipPy/alignments/sm{0}/alignments_iter{1}.db"
shortcuts["um([0-9]*)"] = "sqlite_file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/MP/MPproduction/um{0}/jobData/jobm/um{0}.db"
shortcuts["um([0-9]*)_jobm([0-9]*)"] = "sqlite_file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/MP/MPproduction/um{0}/jobData/jobm{1}/um{0}.db"
shortcuts["hp([0-9]*)_iter([0-9]*)"] = "sqlite_file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN2/HipPy/alignments/hp{0}/alignments_iter{1}.db"
shortcuts["prod"] = "frontier://FrontierProd/CMS_CONDITIONS"

def replaceShortcuts(toScan):
    global shortcuts
    for key, value in shortcuts.items():
        match = re.search(key, toScan)
        if match and match.group(0) == toScan:
            return value.format(*match.groups())
    # no match
    return toScan

def ensurePathExists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

# creates new folder. if folder already exists, move that one to _old, if that one already exists, delete that
def newIterFolder(path, name, iteration):
    if os.path.isdir(os.path.join(path, name, iteration)):
            if os.path.isdir(os.path.join(path, name, iteration+"_old")):
                shutil.rmtree(os.path.join(path, name, iteration+"_old"))
            os.rename(os.path.join(path, name, iteration), os.path.join(path, name, iteration+"_old"))
    os.makedirs(os.path.join(path, name, iteration))


def parseConditions(conditions):
    conds = []
    for record in conditions:
        tag = conditions[record]["tag"]
        source = replaceShortcuts(conditions[record]["source"])
        conds.append( {"record":record, "source":source, "tag":tag} )
    return conds



if __name__ == "__main__":
    print(findFiles(".", "test_{number}.txt"))
-- dummy change --
